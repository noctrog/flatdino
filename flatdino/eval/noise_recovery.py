"""Evaluate noise recovery via per-token L2 reconstruction error.

This script measures how well the flow-matching generator can denoise latents
at various noise levels by computing the L2 distance between original and
denoised latent tokens.

Usage:
    python -m flatdino.eval.noise_recovery \
        --flatdino-path output/flatdino/vae/long-32-bl-128 \
        --generator-path output/flatdino-generator/long-32-bl-128/ditxl-kappa3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from einops import rearrange
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm import tqdm

from flatdino.data import DataConfig, DataLoaders, create_dataloaders
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import RestoredComponents, extract_mu, restore_encoder
from flatdino.distributed import prefetch_to_mesh
from flatdino.decoder.sampling import (
    _infer_time_dist_shift,
    _restore_generator,
    integration_step_upred,
    integration_step_xpred,
)


@dataclass
class Config:
    generator_path: Path
    flatdino_path: Path
    data: DataConfig = field(default_factory=lambda: DataConfig())
    gpu_batch_size: int = 128
    steps: int = 50
    """Total number of denoising steps."""
    step_interval: int = 5
    """Evaluate at every N steps (5, 10, 15, ..., steps)."""
    num_eval_images: int | None = None
    max_batches: int | None = None
    seed: int = 0
    use_ema: bool = True
    cfg_scale: float | None = None
    verbose: bool = False
    force: bool = False
    output_csv: Path | None = None
    plot_prefix: Path | None = None


@dataclass(frozen=True)
class EvalContext:
    restored: RestoredComponents
    image_sharding: NamedSharding
    label_sharding: NamedSharding
    device_count: int
    latent_tokens: int
    latent_dim: int
    model_type: str
    latent_shape: tuple[int, ...]
    time_dist_shift: float | None
    pred_type: str
    generator: nnx.Module
    mesh: jax.sharding.Mesh
    max_batches: int | None
    num_flat_tokens: int = 0
    encoder_disposable_registers: int = 0


def _prepare_context(cfg: Config) -> tuple[EvalContext, int, int | None, int, DataLoaders]:
    if cfg.step_interval <= 0:
        raise ValueError("step_interval must be positive.")
    if cfg.step_interval > cfg.steps:
        raise ValueError("step_interval must not exceed steps.")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive.")
    if cfg.gpu_batch_size <= 0:
        raise ValueError("gpu_batch_size must be positive.")
    if cfg.num_eval_images is not None and cfg.num_eval_images <= 0:
        raise ValueError("num_eval_images must be positive when provided.")
    if cfg.max_batches is not None and cfg.max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    device_count = jax.device_count()
    if device_count == 0:
        raise RuntimeError("No visible JAX devices.")

    mesh = jax.make_mesh((device_count, 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    restored = restore_encoder(cfg.flatdino_path, mesh=mesh, mp=mp, encoder=True, decoder=False)
    if restored.encoder is None:
        raise RuntimeError("FlatDINO checkpoint must contain encoder weights.")
    if restored.data_cfg is None or restored.aug_cfg is None:
        raise RuntimeError("FlatDINO checkpoint is missing data or augmentation configuration.")

    restored.dino.eval()
    restored.encoder.eval()

    latent_tokens = restored.encoder.num_reg
    if latent_tokens is None or latent_tokens <= 0:
        raise ValueError("Encoder configuration does not define a positive number of registers.")
    image_h, image_w = restored.aug_cfg.image_size
    if image_h != image_w:
        raise ValueError("Only square inputs are supported for noise recovery.")

    image_sharding = NamedSharding(mesh, P("data", None, None, None))
    label_sharding = NamedSharding(mesh, P("data"))

    collect_batch_size = cfg.gpu_batch_size * device_count
    data_cfg = replace(cfg.data, num_workers=0)
    augment = FlatDinoValAugmentations(
        replace(restored.aug_cfg, image_size=(image_h, image_w)), restored.data_cfg
    )
    data = create_dataloaders(
        data_cfg,
        collect_batch_size,
        train_epochs=1,
        val_epochs=1,
        drop_remainder_train=False,
        drop_remainder_val=False,
        train_aug=augment,
        val_aug=augment,
    )

    remaining = cfg.num_eval_images
    if remaining is not None:
        remaining = min(remaining, data.val_ds_size)
    total_images = data.val_ds_size if remaining is None else remaining
    estimated_batches = max(1, math.ceil(total_images / collect_batch_size))
    if cfg.max_batches is not None:
        estimated_batches = min(estimated_batches, cfg.max_batches)

    generator, generator_cfg = _restore_generator(
        cfg.generator_path,
        mesh=mesh,
        mp=mp,
        latent_tokens=latent_tokens,
        use_ema=cfg.use_ema,
    )
    generator.eval()

    model_type = (
        generator_cfg.get("model_type", "dit") if isinstance(generator_cfg, dict) else "dit"
    )
    latent_dim = generator.cfg.in_channels
    dit_dh_cfg = generator_cfg.get("dit_dh", {}) if isinstance(generator_cfg, dict) else {}
    patch_size = dit_dh_cfg.get("patch_size", None) if isinstance(dit_dh_cfg, dict) else None
    is_flat_dit_dh = model_type == "dit_dh" and patch_size is None
    if model_type == "dit_dh" and not is_flat_dit_dh:
        grid = int(math.sqrt(latent_tokens))
        if grid * grid != latent_tokens:
            raise ValueError(f"Latent tokens ({latent_tokens}) are not a perfect square for DiTDH.")
        latent_shape: tuple[int, ...] = (grid, grid, latent_dim)
    else:
        latent_shape = (latent_tokens, latent_dim)

    time_dist_shift = _infer_time_dist_shift(
        generator_cfg,
        latent_tokens=latent_tokens,
        latent_dim=latent_dim,
    )
    pred_type = generator_cfg.get("pred_type", "v") if isinstance(generator_cfg, dict) else "v"

    ctx = EvalContext(
        restored=restored,
        image_sharding=image_sharding,
        label_sharding=label_sharding,
        device_count=device_count,
        latent_tokens=latent_tokens,
        latent_dim=latent_dim,
        model_type=model_type,
        latent_shape=latent_shape,
        time_dist_shift=time_dist_shift,
        pred_type=pred_type,
        generator=generator,
        mesh=mesh,
        max_batches=cfg.max_batches,
        num_flat_tokens=restored.num_flat_tokens,
        encoder_disposable_registers=restored.encoder_disposable_registers,
    )
    return ctx, collect_batch_size, remaining, estimated_batches, data


@nnx.jit(static_argnames=("num_flat_tokens", "encoder_disposable_registers", "use_cls"))
def _encode_images(
    images: jax.Array,
    dino,
    encoder,
    num_flat_tokens: int,
    encoder_disposable_registers: int,
    use_cls: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Encode images to latent tokens (jitted)."""
    b, _, _, d = images.shape
    resized = jax.image.resize(
        images,
        (b, dino.resolution, dino.resolution, d),
        method="linear",
        antialias=True,
    )
    dino_outputs = dino(resized)
    patches = dino_outputs[:, 5:]
    cls_targets = dino_outputs[:, 0] if use_cls else None
    encoded = encoder(patches, deterministic=True)
    mu_tokens = extract_mu(encoded, num_flat_tokens, encoder_disposable_registers)
    return mu_tokens, cls_targets


def _denoise_step(
    images: jax.Array,
    labels: jax.Array,
    rng_noise: jax.Array,
    t_start: float,
    cfg_scale: float,
    dino,
    encoder,
    generator,
    latent_shape: tuple[int, ...],
    pred_type: str,
    use_cfg: bool,
    num_flat_tokens: int,
    encoder_disposable_registers: int,
    remaining_steps: int,
    total_steps: int,
    time_dist_shift: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Encode images, add noise, denoise, return original, noisy, and denoised latents.

    Args:
        t_start: Starting time (start_step / total_steps).
        remaining_steps: Number of integration steps (total_steps - start_step).
        total_steps: Total steps for the full schedule (used for x-pred stability).

    Returns:
        original_latents: The original encoded latents (B, T, D).
        noisy_latents: The noisy input before denoising (B, T, D).
        denoised_latents: The denoised output (B, T, D).
    """
    use_cls = generator.cfg.in_cls_dim is not None

    # Encode images (jitted)
    mu_tokens, cls_targets = _encode_images(
        images, dino, encoder, num_flat_tokens, encoder_disposable_registers, use_cls
    )

    if len(latent_shape) == 3:
        h, w = latent_shape[0], latent_shape[1]
        latents = rearrange(mu_tokens, "b (hh ww) c -> b hh ww c", hh=h, ww=w)
    else:
        latents = mu_tokens

    # Create noisy state at t_start
    if use_cls:
        rng_noise, rng_cls = jax.random.split(rng_noise)
    noise = jax.random.normal(rng_noise, latents.shape, dtype=latents.dtype)
    xt = (1.0 - t_start) * noise + t_start * latents

    # Save noisy latents for baseline comparison
    noisy_latents_flat = xt if xt.ndim == 3 else rearrange(xt, "b h w c -> b (h w) c")

    cls_state = None
    if use_cls:
        cls_noise = jax.random.normal(rng_cls, cls_targets.shape, dtype=cls_targets.dtype)
        cls_state = (1.0 - t_start) * cls_noise + t_start * cls_targets

    # Create uniform time schedule from t_start to 1.0 with remaining_steps
    ts = jnp.linspace(t_start, 1.0, remaining_steps + 1, dtype=jnp.float32)
    dts = ts[1:] - ts[:-1]

    unconditional_labels = jnp.full_like(labels, generator.cfg.num_classes) if use_cfg else labels

    # Integration loop (Python for loop, each step is jitted via integration_step_*)
    state = xt
    for i in range(remaining_steps):
        t = jnp.full((labels.shape[0],), ts[i], dtype=jnp.float32)
        dt = dts[i]
        if pred_type == "v":
            result = integration_step_upred(
                generator, state, t, dt, labels, unconditional_labels,
                use_cfg, cfg_scale, cls_state=cls_state,
            )
        else:
            result = integration_step_xpred(
                generator, state, t, dt, labels, unconditional_labels,
                use_cfg, cfg_scale, total_steps, time_dist_shift, cls_state=cls_state,
            )
        if use_cls:
            state = result["x"]
            cls_state = result["cls"]
        else:
            state = result

    denoised_tokens = state

    # Flatten spatial dims if needed for consistent output shape
    original_latents = mu_tokens  # (B, T, D)
    denoised_latents = (
        denoised_tokens
        if denoised_tokens.ndim == 3
        else rearrange(denoised_tokens, "b h w c -> b (h w) c")
    )
    return original_latents, noisy_latents_flat, denoised_latents


def _evaluate_start_step(
    ctx: EvalContext,
    *,
    data,
    remaining: int | None,
    estimated_batches: int,
    start_step: int,
    total_steps: int,
    cfg_scale: float | None,
    seed: int,
    verbose: bool,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Evaluate a single starting step and return per-token MSE stats.

    Args:
        start_step: The step at which to start denoising (0 to total_steps).
        total_steps: Total number of steps in the full schedule.

    Returns:
        processed: Number of images processed.
        token_mean: Mean MSE per token (averaged across dimensions).
        token_var: Variance of MSE per token.
        baseline_mse: Mean MSE of noisy input vs original (expected MSE without denoising).
    """
    use_cfg = cfg_scale is not None
    t_start = start_step / total_steps
    remaining_steps = total_steps - start_step
    cfg_scale_value = cfg_scale if cfg_scale is not None else 0.0

    processed = 0
    batches = 0
    rng = jax.random.PRNGKey(seed)

    # Accumulators for per-token MSE (Welford's online algorithm)
    # For each token, we track mean/var of the per-sample MSE (averaged across dims)
    token_count = 0
    token_mean: np.ndarray | None = None
    token_m2: np.ndarray | None = None  # Sum of squared differences from mean

    # Accumulator for baseline MSE (noisy vs original)
    baseline_mse_sum = 0.0
    baseline_mse_count = 0

    loader_iter = prefetch_to_mesh(iter(data.val_loader), 1, mesh=ctx.mesh, trim=True)
    progress = tqdm(
        loader_iter,
        total=estimated_batches,
        disable=not verbose,
        desc=f"step={start_step}/{total_steps}",
        ncols=80,
    )
    try:
        for batch in progress:
            if remaining is not None and processed >= remaining:
                break
            if ctx.max_batches is not None and batches >= ctx.max_batches:
                break

            images_np = batch["image"].astype(np.float32)
            labels_np = batch["label"].astype(np.int32)
            nb_samples = (images_np.shape[0] // ctx.device_count) * ctx.device_count
            if nb_samples == 0:
                continue

            images_np = images_np[:nb_samples]
            labels_np = labels_np[:nb_samples]
            images = jax.device_put(images_np, ctx.image_sharding)
            labels = jax.device_put(labels_np, ctx.label_sharding)

            rng, rng_noise = jax.random.split(rng)
            rng_noise = jax.device_put(rng_noise, NamedSharding(ctx.mesh, P()))
            orig_latents, noisy_latents, denoised_latents = _denoise_step(
                images,
                labels,
                rng_noise,
                t_start,
                cfg_scale_value,
                ctx.restored.dino,
                ctx.restored.encoder,
                ctx.generator,
                ctx.latent_shape,
                ctx.pred_type,
                use_cfg,
                ctx.num_flat_tokens,
                ctx.encoder_disposable_registers,
                remaining_steps,
                total_steps,
                ctx.time_dist_shift,
            )

            orig_np = np.asarray(jax.device_get(orig_latents), dtype=np.float64)
            noisy_np = np.asarray(jax.device_get(noisy_latents), dtype=np.float64)
            denoised_np = np.asarray(jax.device_get(denoised_latents), dtype=np.float64)

            if remaining is not None:
                take = min(remaining - processed, orig_np.shape[0])
                orig_np = orig_np[:take]
                noisy_np = noisy_np[:take]
                denoised_np = denoised_np[:take]
            else:
                take = orig_np.shape[0]

            if take <= 0:
                break

            # MSE per token (averaged across dims): (B, T)
            sq_error = (orig_np - denoised_np) ** 2
            mse_per_token = sq_error.mean(axis=-1)

            # Baseline MSE (noisy vs original)
            baseline_sq_error = (orig_np - noisy_np) ** 2
            baseline_mse_sum += baseline_sq_error.mean()  # Mean across B, T, D
            baseline_mse_count += 1

            # Welford's online algorithm for mean and variance per token
            for sample in mse_per_token:  # sample has shape (T,)
                token_count += 1
                if token_mean is None:
                    token_mean = sample.copy()
                    token_m2 = np.zeros_like(sample)
                else:
                    delta = sample - token_mean
                    token_mean += delta / token_count
                    delta2 = sample - token_mean
                    token_m2 += delta * delta2

            processed += take
            batches += 1
    finally:
        if isinstance(progress, tqdm):
            progress.close()

    if processed == 0 or token_mean is None or token_m2 is None:
        raise RuntimeError(f"No images processed for start_step={start_step}.")

    # Compute variance from M2
    token_var = token_m2 / token_count

    # Compute mean baseline MSE
    baseline_mse = baseline_mse_sum / baseline_mse_count if baseline_mse_count > 0 else 0.0

    return processed, token_mean, token_var, baseline_mse


def _save_token_mse_plot(
    token_stats_per_step: dict[int, tuple[np.ndarray, np.ndarray, float]],
    plot_prefix: Path,
    total_steps: int,
) -> None:
    """Save bar plot of per-token MSE (per dim) with std for each starting step."""
    if not token_stats_per_step:
        return

    plot_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Sort by step
    steps = sorted(token_stats_per_step.keys())
    num_steps = len(steps)

    # Create figure with subplots for each step
    fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 5), sharey=True, squeeze=False)
    axes = axes.flatten()

    for idx, step in enumerate(steps):
        ax = axes[idx]
        token_mean, token_var, baseline_mse = token_stats_per_step[step]
        token_std = np.sqrt(token_var)
        num_tokens = len(token_mean)
        token_indices = np.arange(num_tokens)

        ax.bar(token_indices, token_mean, yerr=token_std, color="steelblue", alpha=0.8, capsize=2)
        ax.axhline(y=baseline_mse, color="red", linestyle="--", linewidth=1.5, label="Noisy MSE")
        ax.set_xlabel("Token")
        if idx == 0:
            ax.set_ylabel("MSE")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"step {step}/{total_steps}")
        ax.set_xticks(token_indices[:: max(1, num_tokens // 8)])
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle("Per-Token MSE", fontsize=14)
    fig.tight_layout()

    png_path = plot_prefix.parent / f"{plot_prefix.stem}_token_mse.png"
    pdf_path = plot_prefix.parent / f"{plot_prefix.stem}_token_mse.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-token MSE plots to {png_path} and {pdf_path}")


def _save_summary_plot(
    df: pd.DataFrame,
    plot_prefix: Path,
    total_steps: int,
) -> None:
    """Save summary plot of mean MSE vs starting step."""
    plot_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values("start_step")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_sorted["start_step"], df_sorted["mean_mse"], marker="o", linewidth=2, label="Denoised")
    ax.fill_between(
        df_sorted["start_step"],
        df_sorted["mean_mse"] - df_sorted["std_mse"],
        df_sorted["mean_mse"] + df_sorted["std_mse"],
        alpha=0.3,
    )
    # Plot baseline MSE (noisy input)
    if "baseline_mse" in df_sorted.columns:
        ax.plot(
            df_sorted["start_step"], df_sorted["baseline_mse"],
            marker="x", linewidth=2, linestyle="--", color="red", label="Noisy (baseline)"
        )
    ax.set_xlabel(f"Starting Step (out of {total_steps})")
    ax.set_ylabel("Mean MSE")
    ax.set_title("Noise Recovery: Per-Token MSE")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    fig.tight_layout()

    png_path = plot_prefix.with_suffix(".png")
    pdf_path = plot_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plots to {png_path} and {pdf_path}")


def main(cfg: Config) -> None:
    if cfg.output_csv is None:
        cfg.output_csv = cfg.flatdino_path / "noise_recovery_l2.csv"
    if cfg.plot_prefix is None:
        cfg.plot_prefix = cfg.flatdino_path / "noise_recovery_l2"

    ctx, collect_batch_size, remaining, estimated_batches, data = _prepare_context(cfg)
    print(f"Using batch size {collect_batch_size} across {ctx.device_count} device(s).")
    print(f"Evaluating up to {remaining or data.val_ds_size} validation images.")
    print(f"Total steps: {cfg.steps}, evaluating at intervals of {cfg.step_interval}")

    if cfg.cfg_scale is not None and getattr(ctx.generator.cfg, "class_dropout_prob", 0.0) <= 0:
        raise ValueError(
            "Classifier-free guidance requested but generator was trained without label dropout."
        )

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate at steps: step_interval, 2*step_interval, ..., steps
    start_steps = list(range(cfg.step_interval, cfg.steps + 1, cfg.step_interval))
    rng_seed_base = cfg.seed

    # Track per-token MSE stats for plotting: {step: (mean, var, baseline_mse)}
    token_stats_per_step: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    results = []

    for idx, start_step in enumerate(start_steps):
        processed, token_mean, token_var, baseline_mse = _evaluate_start_step(
            ctx,
            data=data,
            remaining=remaining,
            estimated_batches=estimated_batches,
            start_step=start_step,
            total_steps=cfg.steps,
            cfg_scale=cfg.cfg_scale,
            seed=rng_seed_base + idx,
            verbose=cfg.verbose,
        )

        # Aggregate stats across tokens
        mean_mse = float(token_mean.mean())
        std_mse = float(np.sqrt(token_var).mean())  # Mean of per-token std
        remaining_steps = cfg.steps - start_step
        print(
            f"step {start_step}/{cfg.steps} (t={start_step / cfg.steps:.2f}, {remaining_steps} steps) -> "
            f"mean MSE: {mean_mse:.6f} (std: {std_mse:.6f}), baseline: {baseline_mse:.6f} over {processed} images"
        )

        token_stats_per_step[start_step] = (token_mean, token_var, baseline_mse)
        results.append(
            {
                "start_step": start_step,
                "t_start": start_step / cfg.steps,
                "remaining_steps": remaining_steps,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "baseline_mse": baseline_mse,
                "min_mse": float(token_mean.min()),
                "max_mse": float(token_mean.max()),
                "processed_images": processed,
                "total_steps": cfg.steps,
                "cfg_scale": cfg.cfg_scale if cfg.cfg_scale is not None else np.nan,
                "time_dist_shift": ctx.time_dist_shift
                if ctx.time_dist_shift is not None
                else np.nan,
            }
        )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(cfg.output_csv, index=False)
    print(f"Saved results to {cfg.output_csv}")

    # Save plots
    _save_summary_plot(df, cfg.plot_prefix, cfg.steps)
    _save_token_mse_plot(token_stats_per_step, cfg.plot_prefix, cfg.steps)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
