from pathlib import Path
from typing import TypeVar

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
from dacite import from_dict, Config as DaciteConfig

from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig, OptimConfig

T = TypeVar("T")


# =============================================================================
# Latent Extraction Utilities
# =============================================================================


def extract_mu_logvar(
    encoded: jax.Array,
    num_latents: int,
    encoder_disposable_registers: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Extract mu and logvar from encoder output, handling disposable registers.

    Args:
        encoded: Encoder output of shape (B, T, D) where D is 2*latent_dim
        num_latents: Number of actual latent tokens (excluding disposable registers)
        encoder_disposable_registers: Number of disposable registers to skip

    Returns:
        mu, logvar: Each of shape (B, num_latents, latent_dim)
    """
    if encoded.shape[-1] % 2 != 0:
        raise ValueError("Encoder output dim must be even to split into mu/logvar.")
    mu, logvar = jnp.split(encoded, 2, axis=-1)
    skip = encoder_disposable_registers
    return mu[:, skip : skip + num_latents], logvar[:, skip : skip + num_latents]


def extract_mu(
    encoded: jax.Array,
    num_latents: int,
    encoder_disposable_registers: int = 0,
) -> jax.Array:
    """Extract mu (mean) from encoder output, handling disposable registers.

    Args:
        encoded: Encoder output of shape (B, T, D) where D is 2*latent_dim
        num_latents: Number of actual latent tokens (excluding disposable registers)
        encoder_disposable_registers: Number of disposable registers to skip

    Returns:
        mu: Shape (B, num_latents, latent_dim)
    """
    mu, _ = extract_mu_logvar(encoded, num_latents, encoder_disposable_registers)
    return mu


def extract_decoder_patches(
    decoded: jax.Array,
    num_output_patches: int,
    decoder_disposable_registers: int = 0,
) -> jax.Array:
    """Extract output patches from decoder, handling disposable registers.

    Args:
        decoded: Decoder output of shape (B, T, D)
        num_output_patches: Number of actual output patches (excluding disposable registers)
        decoder_disposable_registers: Number of disposable registers to skip

    Returns:
        patches: Shape (B, num_output_patches, D)
    """
    skip = decoder_disposable_registers
    return decoded[:, skip : skip + num_output_patches]


def restore_train_config(flatdino_path: Path) -> FlatDinoConfig:
    item_names = FlatDinoAutoencoder.get_item_names() + ["optim", "loader", "config"]
    opts = ocp.CheckpointManagerOptions(read_only=True)
    mngr = ocp.CheckpointManager(flatdino_path.absolute(), options=opts, item_names=item_names)
    step = mngr.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found in path {flatdino_path.absolute()}")
    print(f"Found checkpoint at step {step}. Restoring ...")

    cfg_d = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"]
    return from_dict(FlatDinoConfig, cfg_d, DaciteConfig(cast=[tuple], strict=False))


def restore_vision_encoders(
    flatdino_path: Path, mp: jmp.Policy, mesh: jax.sharding.Mesh
) -> tuple[FlatDinoAutoencoder, FlatDinoConfig]:
    flatdino_mngr = ocp.CheckpointManager(
        flatdino_path.absolute(),
        options=ocp.CheckpointManagerOptions(read_only=True, best_mode="min"),
        item_names=FlatDinoAutoencoder.get_item_names() + ["optim", "config", "loader"],
    )
    flatdino_step = flatdino_mngr.best_step()
    cfg_d = flatdino_mngr.restore(
        flatdino_step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
    )["config"]
    flat_cfg = from_dict(FlatDinoConfig, cfg_d, config=DaciteConfig(cast=[tuple], strict=False))
    flatdino = FlatDinoAutoencoder(flat_cfg, mesh, mngr=flatdino_mngr, mp=mp, rngs=nnx.Rngs(0))

    return flatdino, flat_cfg


def build_lr_schedule(
    cfg: OptimConfig,
    total_updates: int,
    *,
    schedule: str | None = None,
    warmup_steps_override: int | None = None,
    decay_steps_override: int | None = None,
):
    """Build an Optax LR schedule keyed on optimizer update counts.

    Args:
        cfg: OptimConfig with lr_* fields.
        total_updates: total optimizer steps (after grad accumulation).
        schedule: optional override for cfg.lr_schedule.
        warmup_steps_override: optional warmup step count; defaults to cfg.warmup_epochs * (total_updates // cfg.epochs).
        decay_steps_override: optional decay step count; defaults to total_updates.
    """
    sched = schedule or cfg.lr_schedule
    steps_per_epoch = total_updates // cfg.epochs
    warmup_steps = (
        cfg.warmup_epochs * steps_per_epoch
        if warmup_steps_override is None
        else warmup_steps_override
    )
    decay_steps = decay_steps_override or total_updates

    match sched:
        case "constant":
            return optax.schedules.constant_schedule(cfg.lr_peak)
        case "warmup_cosine":
            return optax.schedules.warmup_cosine_decay_schedule(
                init_value=cfg.lr_start,
                peak_value=cfg.lr_peak,
                warmup_steps=warmup_steps,
                decay_steps=total_updates,
                end_value=cfg.lr_final,
            )
        case "wsd":
            remaining_steps = total_updates - warmup_steps
            if decay_steps_override is not None:
                assert 0 <= decay_steps_override <= remaining_steps, (
                    "decay_steps_override must be between 0 and total_updates - warmup_steps"
                )
            decay_steps = decay_steps_override or remaining_steps
            decay_start = total_updates - decay_steps
            return optax.schedules.join_schedules(
                [
                    optax.schedules.warmup_constant_schedule(
                        init_value=cfg.lr_start, peak_value=cfg.lr_peak, warmup_steps=warmup_steps
                    ),
                    optax.schedules.linear_schedule(
                        init_value=cfg.lr_peak, end_value=cfg.lr_final, transition_steps=decay_steps
                    ),
                ],
                [decay_start],
            )
        case "linear_decay":
            decay = max(total_updates - warmup_steps, 1)
            warmup_sched = (
                optax.schedules.linear_schedule(cfg.lr_start, cfg.lr_peak, warmup_steps)
                if warmup_steps > 0 and cfg.lr_start != cfg.lr_peak
                else optax.schedules.constant_schedule(cfg.lr_peak)
            )
            decay_sched = optax.schedules.linear_schedule(cfg.lr_peak, cfg.lr_final, decay)
            return (
                optax.schedules.join_schedules([warmup_sched, decay_sched], [warmup_steps])
                if warmup_steps > 0
                else decay_sched
            )
        case _:
            raise ValueError(f"Unsupported lr_schedule: {sched}")
