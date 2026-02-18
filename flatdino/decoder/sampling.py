import math
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import orbax.checkpoint as ocp
from dacite import Config as DaciteConfig, from_dict
from absl import logging

from flatdino.models.dit import DiTConfig, LightningDiT
from flatdino.decoder.dit_dh import DiTDHConfig, DiTDH


def _ensure_hw(value) -> tuple[int, int]:
    """Convert scalars or 2-tuples/lists into a 2D (h, w) tuple."""

    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Expected a 2-element tuple for spatial dimensions.")
        return (int(value[0]), int(value[1]))
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("Expected a 2-element list for spatial dimensions.")
        return (int(value[0]), int(value[1]))
    raise TypeError("Spatial dimensions must be an int, tuple, or list.")


def get_latent_metadata(generator) -> dict:
    """Infer latent grid metadata from a generator instance."""
    cfg = generator.cfg
    if hasattr(cfg, "patch_embed") and getattr(cfg, "patch_embed") is not None:
        img_h, img_w = _ensure_hw(cfg.patch_embed.img_size)
        pat_h, pat_w = _ensure_hw(cfg.patch_embed.patch_size)
        latent_dim = cfg.in_channels
    elif hasattr(generator, "x_patch_size"):
        img_h = img_w = cfg.input_size
        pat_h = pat_w = generator.x_patch_size
        latent_dim = cfg.in_channels
    else:
        raise ValueError("Unable to infer latent metadata from generator configuration.")

    if img_h % pat_h != 0 or img_w % pat_w != 0:
        raise ValueError("Patch size must evenly divide the latent grid dimensions.")

    grid_h = img_h // pat_h
    grid_w = img_w // pat_w
    latent_shape = (grid_h, grid_w, latent_dim)
    patch_tokens = grid_h * grid_w

    return {
        "image_hw": (img_h, img_w),
        "patch_hw": (pat_h, pat_w),
        "grid_hw": (grid_h, grid_w),
        "patch_tokens": patch_tokens,
        "latent_dim": latent_dim,
        "latent_shape": latent_shape,
    }


def _restore_generator(
    path: Path,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    latent_tokens: int | None,
    use_ema: bool,
    step: int | None = None,
) -> tuple[LightningDiT | DiTDH, dict]:
    """Restore a generator checkpoint (DiT or DiTDH).

    Args:
        step: Checkpoint step to restore. If None, uses the latest step.
    """
    item_names = ["model", "model_ema", "optim", "loader", "config"]
    opts = ocp.CheckpointManagerOptions(read_only=True)
    mngr = ocp.CheckpointManager(path.absolute(), options=opts, item_names=item_names)

    if step is None:
        step = mngr.latest_step()
        if step is None:
            raise ValueError(f"No checkpoint found in {path}.")
    elif step not in mngr.all_steps():
        raise ValueError(f"Step {step} not found in {path}. Available: {mngr.all_steps()}")
    print(f"Found generator checkpoint at step {step}. Restoring...")

    dacite_config = DaciteConfig(cast=[tuple])
    cfg_blob = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    cfg_dict = cfg_blob["config"]
    model_type = cfg_dict.get("model_type", "dit")
    dit_cfg = from_dict(DiTConfig, cfg_dict["dit"], config=dacite_config)
    dit_dh_cfg = (
        from_dict(DiTDHConfig, cfg_dict["dit_dh"], config=dacite_config)
        if "dit_dh" in cfg_dict
        else None
    )

    def _restore(name: str):
        match model_type:
            case "dit":
                return LightningDiT.restore(
                    mngr, step, name, mesh, cfg=dit_cfg, mp=mp, num_tokens=latent_tokens
                )
            case "dit_dh":
                if dit_dh_cfg is None:
                    raise ValueError("Checkpoint missing 'dit_dh' configuration.")
                return DiTDH.restore(mngr, step, name, mesh, cfg=dit_dh_cfg, mp=mp)
            case _:
                raise ValueError(f"Unsupported model_type: {model_type}")

    preferred_name = "model"
    if use_ema:
        try:
            generator = _restore("model_ema")
            preferred_name = "model_ema"
        except Exception as err:  # pragma: no cover - orbax raises backend-specific errors
            print(f"WARNING: Failed to restore EMA weights ({err}). Falling back to 'model'.")
            generator = _restore("model")
    else:
        generator = _restore("model")

    print(f"Restored generator weights from '{preferred_name}'.")
    mngr.close()
    return generator, cfg_dict


def _infer_time_dist_shift(
    cfg_blob: dict,
    latent_tokens: int,
    latent_dim: int,
) -> float | None:
    """Infer the time distortion shift used during training."""
    base = cfg_blob.get("time_dist_shift_base")
    if base in (None, 0):
        return None
    shift_dim = cfg_blob.get("time_dist_shift_dim")
    if shift_dim is None:
        shift_dim = latent_tokens * latent_dim
    if shift_dim <= 0:
        raise ValueError("time_dist_shift_dim must be positive.")
    shift = math.sqrt(float(shift_dim) / float(base))
    return max(1.0, shift)


def xpred_one_minus_t(
    t: jax.Array,
    *,
    steps: int,
    target_ndim: int,
    time_dist_shift: float | None,
) -> jax.Array:
    """Broadcasted (1 - t) with a schedule-aware floor for x-pred stability."""
    if steps <= 0:
        raise ValueError("steps must be positive.")

    min_t = 1.0 / float(steps)
    if time_dist_shift is not None:
        if time_dist_shift <= 0:
            raise ValueError("time_dist_shift must be positive when provided.")
        min_t /= float(time_dist_shift)

    t_broadcast = t.reshape((t.shape[0],) + (1,) * (target_ndim - 1))
    return jnp.maximum(1.0 - t_broadcast, min_t)


@nnx.jit(static_argnames=("use_cfg", "cfg_scale"))
def integration_step_upred(
    generator: LightningDiT | DiTDH,
    state,
    t,
    dts,
    labels,
    unconditional_labels,
    use_cfg,
    cfg_scale,
    cls_state: jax.Array | None = None,
):
    logging.info("Compiling u-pred integration step")
    if cls_state is not None:
        assert hasattr(generator, "in_cls_proj")
        velocities = generator(state, t, labels, train=False, cls_=cls_state)
        if use_cfg:
            velocities_uncod = generator(
                state, t, unconditional_labels, train=False, cls_=cls_state
            )
            velocities = {
                k: v_uncod + cfg_scale * (velocities[k] - v_uncod)
                for k, v_uncod in velocities_uncod.items()
            }
        new_state = state + velocities["x"] * dts
        new_cls_state = cls_state + velocities["cls"] * dts
        return {"x": new_state, "cls": new_cls_state}
    else:
        velocity = generator(state, t, labels, train=False)["x"]
        if use_cfg:
            velocity_uncond = generator(state, t, unconditional_labels, train=False)["x"]
            velocity = velocity_uncond + cfg_scale * (velocity - velocity_uncond)
        return state + velocity * dts


@nnx.jit(static_argnames=("use_cfg", "cfg_scale", "xpred_steps", "time_dist_shift"))
def integration_step_xpred(
    generator,
    state,
    t,
    dts,
    labels,
    unconditional_labels,
    use_cfg,
    cfg_scale,
    xpred_steps: int,
    time_dist_shift: float | None,
    cls_state: jax.Array | None = None,
):
    logging.info("Compiling x-pred integration step")
    if cls_state is not None:
        assert hasattr(generator, "in_cls_proj")
        x_pred = generator(state, t, labels, train=False, cls_=cls_state)
        denom = xpred_one_minus_t(
            t,
            steps=xpred_steps,
            target_ndim=state.ndim,
            time_dist_shift=time_dist_shift,
        )
        velocity = (x_pred["x"] - state) / denom

        denom_cls = xpred_one_minus_t(
            t, steps=xpred_steps, target_ndim=cls_state.ndim, time_dist_shift=time_dist_shift
        )
        velocity_cls = (x_pred["cls"] - cls_state) / denom_cls

        if use_cfg:
            x_pred_uncond = generator(state, t, unconditional_labels, train=False, cls_=cls_state)
            v_pred_uncond = (x_pred_uncond["x"] - state) / denom
            v_pred_uncond_cls = (x_pred_uncond["cls"] - cls_state) / denom_cls
            velocity = v_pred_uncond + cfg_scale * (velocity - v_pred_uncond)
            velocity_cls = v_pred_uncond_cls + cfg_scale * (velocity_cls - v_pred_uncond_cls)
        new_state = state + velocity * dts
        new_cls_state = cls_state + velocity_cls * dts
        return {"x": new_state, "cls": new_cls_state}
    else:
        x_pred = generator(state, t, labels, train=False)["x"]
        denom = xpred_one_minus_t(
            t, steps=xpred_steps, target_ndim=state.ndim, time_dist_shift=time_dist_shift
        )
        velocity = (x_pred - state) / denom

        if use_cfg:
            x_pred_uncod = generator(state, t, unconditional_labels, train=False)["x"]
            v_pred_uncod = (x_pred_uncod - state) / denom
            velocity = v_pred_uncod + cfg_scale * (velocity - v_pred_uncod)
        return state + velocity * dts


def _sample_latents(
    generator: LightningDiT | DiTDH,
    *,
    labels: jax.Array,
    steps: int,
    key: jax.Array,
    latent_shape: tuple[int, ...],
    cfg_scale: float | None = None,
    cfg_interval_min: float = 0.0,
    cfg_interval_max: float = 1.0,
    time_dist_shift: float | None = None,
    pred_type: str = "v",
) -> jax.Array | dict[str, jax.Array]:
    """Run flow-matching sampling to produce latent grid tokens (and CLS if enabled)."""
    key, key_ = jax.random.split(key)
    state = jax.random.normal(key_, (labels.shape[0],) + latent_shape, dtype=jnp.float32)

    # Generate noise tensor for the cls reconstruction, if enabled
    predict_cls = generator.cfg.in_cls_dim is not None
    if predict_cls:
        key, key_ = jax.random.split(key)
        cls_state = jax.random.normal(
            key_, (labels.shape[0], generator.cfg.in_cls_dim), dtype=jnp.float32
        )
    else:
        cls_state = None

    ts = jnp.linspace(0.0, 1.0, steps + 1, dtype=state.dtype)
    if time_dist_shift is not None and time_dist_shift > 1.0:
        shift = jnp.asarray(time_dist_shift, dtype=state.dtype)
        warped_ts = ts / (shift - (shift - 1.0) * ts)
    else:
        warped_ts = ts
    dts = warped_ts[1:] - warped_ts[:-1]

    use_cfg = cfg_scale is not None
    unconditional_labels = labels
    if use_cfg:
        if cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative.")
        if generator.cfg.class_dropout_prob <= 0:
            raise ValueError(
                "Classifier-free guidance requested but generator was trained without label dropout."
            )
        unconditional_labels = jnp.full_like(labels, generator.cfg.num_classes)

    match pred_type:
        case "v":
            integration_step = integration_step_upred
            integration_kwargs = {}
        case "x":
            integration_step = integration_step_xpred
            integration_kwargs = {
                "xpred_steps": steps,
                "time_dist_shift": time_dist_shift,
            }
        case _:
            raise ValueError("invalid pred_type")

    for idx in range(steps):
        t_val = float(warped_ts[idx])
        # Guidance interval: only apply CFG when t is within (cfg_interval_min, cfg_interval_max)
        # This follows the DDT approach which applies CFG only for t > 0.3 by default
        apply_cfg = use_cfg and (t_val > cfg_interval_min) and (t_val < cfg_interval_max)
        effective_cfg_scale = cfg_scale if apply_cfg else None

        t = jnp.full((labels.shape[0],), warped_ts[idx], dtype=state.dtype)
        state = integration_step(
            generator,
            state,
            t,
            dts[idx],
            labels,
            unconditional_labels,
            apply_cfg,
            effective_cfg_scale,
            **(integration_kwargs if pred_type == "x" else {}),
            cls_state=cls_state,
        )
        if cls_state is not None:
            cls_state = state["cls"]
            state = state["x"]

    if cls_state is not None:
        return {"cls": cls_state, "x": state}
    return state
