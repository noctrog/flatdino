from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import chex
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import seaborn as sns
import tyro
from absl import logging
from dacite import Config as DaciteConfig, from_dict
from einops import rearrange
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm import tqdm

import flatdino.metrics.fid as fid_eval
from flatdino.data import DataConfig, create_dataloaders
from flatdino.metrics.fid import FIDRunningStats
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.pretrained.inception import InceptionV3
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import save_eval_results
from flatdino.eval.metrics import (
    DEBUG_IMAGE_COUNT,
    DEBUG_IMAGE_PATH,
    _psnr_stats,
    _save_debug_image,
)
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig


@dataclass
class Config:
    path: Path
    data: DataConfig = field(default_factory=lambda: DataConfig())
    gpu_collect_batch_size: int = 128
    num_eval_images: int | None = None
    max_batches: int | None = None
    seed: int = 0
    num_noise_levels: int = 6
    verbose: bool = False
    debug_images: bool = False
    output_csv: Path | None = None
    plot_path: Path | None = None
    grid_dir: Path | None = None


def _decode_images(
    images: jax.Array,
    *,
    dino: Any,
    flatdino: FlatDinoAutoencoder,
    rae_decoder: Any,
    mean: jax.Array,
    std: jax.Array,
    noise_sigma_patches: float = 0.0,
    noise_sigma_latent: float = 0.0,
    rng_patches: jax.Array | None = None,
    rng_latent: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    chex.assert_rank(images, 4)
    b, _, _, d = images.shape
    dino_images = jax.image.resize(
        images,
        (b, dino.resolution, dino.resolution, d),
        method="linear",
        antialias=True,
    )
    patches = dino(dino_images)[:, 5:]

    # Encode with FlatDinoAutoencoder (tanh applied internally if configured)
    z, _ = flatdino.encode(patches)

    if noise_sigma_patches > 0.0:
        if rng_patches is None:
            raise ValueError("rng_patches is required when adding noise to DINO latents.")
        patches = (
            patches
            + jax.random.normal(rng_patches, patches.shape, dtype=patches.dtype)
            * noise_sigma_patches
        )

    recon_patches = rae_decoder(patches)
    recon_patches = rae_decoder.unpatchify(recon_patches.logits)
    recon_patches = jnp.transpose(recon_patches, (0, 2, 3, 1))

    if noise_sigma_latent > 0.0:
        if rng_latent is None:
            raise ValueError("rng_latent is required when adding noise to FlatDINO latents.")
        z = z + jax.random.normal(rng_latent, z.shape, dtype=z.dtype) * noise_sigma_latent

    # Decode with FlatDinoAutoencoder
    recon_tokens = flatdino.decode(z)
    decoder_out = rae_decoder(recon_tokens)
    recon_flatdino = rae_decoder.unpatchify(decoder_out.logits)
    recon_flatdino = jnp.transpose(recon_flatdino, (0, 2, 3, 1))

    original = jnp.clip(images * std + mean, 0.0, 1.0)
    recon_dino = jnp.clip(recon_patches * std + mean, 0.0, 1.0)
    recon_flatdino = jnp.clip(recon_flatdino * std + mean, 0.0, 1.0)
    return original, recon_dino, recon_flatdino


def _to_inception_features(inception_forward, pixels: jax.Array) -> jax.Array:
    resized = jax.image.resize(
        pixels, (pixels.shape[0], 299, 299, 3), method="bicubic", antialias=False
    )
    normalized = resized * 2.0 - 1.0
    return inception_forward(normalized)


@nnx.jit(static_argnames=("sigma", "inception_forward"))
def _metrics_step(
    dino,
    flatdino: FlatDinoAutoencoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    inception_forward,
    images: jax.Array,
    rng_patches: jax.Array,
    rng_latent: jax.Array,
    *,
    sigma: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    original, recon_dino, recon_flatdino = _decode_images(
        images,
        dino=dino,
        flatdino=flatdino,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        noise_sigma_patches=sigma,
        noise_sigma_latent=sigma,
        rng_patches=rng_patches,
        rng_latent=rng_latent,
    )

    feats_real = _to_inception_features(inception_forward, original)
    feats_dino = _to_inception_features(inception_forward, recon_dino)
    feats_flatdino = _to_inception_features(inception_forward, recon_flatdino)

    mse_dino = jnp.mean(jnp.square(recon_dino - original), axis=(1, 2, 3))
    mse_flatdino = jnp.mean(jnp.square(recon_flatdino - original), axis=(1, 2, 3))
    psnr_dino = -10.0 * jnp.log10(jnp.maximum(mse_dino, 1e-10))
    psnr_flatdino = -10.0 * jnp.log10(jnp.maximum(mse_flatdino, 1e-10))
    return feats_real, feats_dino, feats_flatdino, psnr_dino, psnr_flatdino


@dataclass
class _SetupResult:
    """Result from _setup_eval containing all evaluation components."""

    dino: Any
    flatdino: FlatDinoAutoencoder
    flatdino_cfg: FlatDinoConfig
    rae_decoder: Any
    mean: jax.Array
    std: jax.Array
    inception_forward: Callable[[jax.Array], jax.Array]
    image_sharding: NamedSharding
    device_count: int
    remaining: int | None
    estimated_batches: int
    data: Any


def _setup_eval(cfg: Config, *, val_epochs: int = 1) -> _SetupResult:
    device_count = jax.device_count()
    if device_count == 0:
        raise RuntimeError("No visible JAX devices.")
    if cfg.gpu_collect_batch_size <= 0:
        raise ValueError("gpu_collect_batch_size must be positive.")
    if cfg.num_eval_images is not None and cfg.num_eval_images <= 0:
        raise ValueError("num_eval_images must be positive when provided.")
    if cfg.max_batches is not None and cfg.max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    mesh = jax.make_mesh((device_count, 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )
    dacite_config = DaciteConfig(cast=[tuple], strict=False)

    # Load checkpoint and config
    ckpt_path = cfg.path.absolute()
    mngr = ocp.CheckpointManager(
        ckpt_path,
        item_names=FlatDinoAutoencoder.get_item_names() + ["optim", "loader", "config"],
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    step = mngr.best_step()
    assert step is not None, "Failed to load best step from checkpoint"
    print(f"Found checkpoint at step {step}. Restoring...")

    cfg_ckpt = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"]
    flatdino_cfg = from_dict(FlatDinoConfig, cfg_ckpt, config=dacite_config)

    # Create models
    dino = DinoWithRegisters(flatdino_cfg.dino_name, resolution=224, dtype=mp.param_dtype)
    flatdino = FlatDinoAutoencoder(
        flatdino_cfg, mesh, mngr=mngr, step=step, mp=mp, rngs=nnx.Rngs(cfg.seed)
    )

    dino.eval()
    flatdino.encoder.eval()
    flatdino.decoder.eval()

    num_patch_tokens = (dino.resolution // 14) ** 2
    num_output_patches = flatdino_cfg.num_output_patches
    if num_output_patches != num_patch_tokens:
        raise ValueError(
            f"Decoder expects {num_output_patches} tokens but DINO produced {num_patch_tokens}."
        )

    rae_decoder = make_rae_decoder(
        num_patches=num_patch_tokens,
        image_size=256,
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    if not hasattr(fid_eval, "InceptionV3"):
        fid_eval.InceptionV3 = InceptionV3
    inception_forward = fid_eval.create_inception_forward()

    mean = jnp.array(flatdino_cfg.data.normalization_mean, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(flatdino_cfg.data.normalization_std, dtype=jnp.float32)[None, None, None, :]
    image_sharding = NamedSharding(mesh, P("data", None, None, None))

    collect_batch_size = cfg.gpu_collect_batch_size * device_count
    data_cfg = replace(cfg.data, num_workers=0)
    data = create_dataloaders(
        data_cfg,
        collect_batch_size,
        train_epochs=1,
        val_epochs=val_epochs,
        train_aug=FlatDinoValAugmentations(
            replace(flatdino_cfg.aug, image_size=(256, 256)), flatdino_cfg.data
        ),
        val_aug=FlatDinoValAugmentations(
            replace(flatdino_cfg.aug, image_size=(256, 256)), flatdino_cfg.data
        ),
        drop_remainder_train=False,
        drop_remainder_val=False,
    )

    remaining = cfg.num_eval_images
    if remaining is not None:
        remaining = min(remaining, data.val_ds_size)

    total_images = data.val_ds_size if remaining is None else remaining
    estimated_batches = max(1, math.ceil(total_images / collect_batch_size))
    if cfg.max_batches is not None:
        estimated_batches = min(estimated_batches, cfg.max_batches)

    return _SetupResult(
        dino=dino,
        flatdino=flatdino,
        flatdino_cfg=flatdino_cfg,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        inception_forward=inception_forward,
        image_sharding=image_sharding,
        device_count=device_count,
        remaining=remaining,
        estimated_batches=estimated_batches,
        data=data,
    )


def main(cfg: Config):
    if cfg.num_noise_levels <= 1:
        raise ValueError("num_noise_levels must be greater than 1.")

    output_csv = cfg.path / "noise_levels_metrics.csv" if cfg.output_csv is None else cfg.output_csv
    plot_path = cfg.path / "noise_levels_plot.png" if cfg.plot_path is None else cfg.plot_path
    grid_dir = cfg.path / "noise_level_grids" if cfg.grid_dir is None else cfg.grid_dir

    setup = _setup_eval(cfg, val_epochs=1)

    grid_dir.mkdir(parents=True, exist_ok=True)

    def _save_image_grid(images: np.ndarray, model_name: str, sigma: float):
        if images.shape[0] == 0:
            return

        needed = 16
        if images.shape[0] < needed:
            pad = np.ones((needed - images.shape[0], *images.shape[1:]), dtype=images.dtype)
            images = np.concatenate([images, pad], axis=0)
        else:
            images = images[:needed]

        grid = rearrange(images, "(r c) h w d -> (r h) (c w) d", r=4, c=4)
        plt.figure(figsize=(6, 6))
        plt.imshow(grid)
        plt.axis("off")
        plt.title(f"{model_name}-{sigma:.2f}")
        out_path = grid_dir / f"{model_name}-{sigma:.2f}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()

    def _decode(
        images: jax.Array,
        *,
        noise_sigma_patches: float = 0.0,
        noise_sigma_latent: float = 0.0,
        rng_patches: jax.Array | None = None,
        rng_latent: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _decode_images(
            images,
            dino=setup.dino,
            flatdino=setup.flatdino,
            rae_decoder=setup.rae_decoder,
            mean=setup.mean,
            std=setup.std,
            noise_sigma_patches=noise_sigma_patches,
            noise_sigma_latent=noise_sigma_latent,
            rng_patches=rng_patches,
            rng_latent=rng_latent,
        )

    noise_levels = np.linspace(0.0, 1.0, cfg.num_noise_levels)
    records: list[dict[str, float]] = []

    # Load existing results to skip already-computed noise levels
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        existing_sigmas = set(existing_df["sigma"].tolist())
        records = existing_df.to_dict("records")
        noise_levels = np.array([s for s in noise_levels if s not in existing_sigmas])
        if len(noise_levels) == 0:
            logging.info(
                f"All {cfg.num_noise_levels} noise levels already computed in {output_csv}"
            )
            return
        logging.info(
            f"Found {len(existing_sigmas)} existing noise levels, "
            f"computing {len(noise_levels)} remaining"
        )

    rng = jax.random.PRNGKey(cfg.seed)
    for sigma in noise_levels:
        sigma_float = float(sigma)
        debug_originals: list[np.ndarray] = []
        debug_dino: list[np.ndarray] = []
        debug_flatdino: list[np.ndarray] = []

        def _run_metrics_step(
            images: jax.Array, rng_patches: jax.Array, rng_latent: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return _metrics_step(
                setup.dino,
                setup.flatdino,
                setup.rae_decoder,
                setup.mean,
                setup.std,
                setup.inception_forward,
                images,
                rng_patches,
                rng_latent,
                sigma=sigma_float,
            )

        stats_real = FIDRunningStats()
        stats_recon_dino = FIDRunningStats()
        stats_recon_flatdino = FIDRunningStats()
        psnr_dino_values: list[np.ndarray] = []
        psnr_flat_values: list[np.ndarray] = []
        processed = 0
        batches_processed = 0
        grid_dino: np.ndarray | None = None
        grid_flatdino: np.ndarray | None = None

        loader_iter = iter(setup.data.val_loader)
        progress = tqdm(
            loader_iter,
            total=setup.estimated_batches,
            disable=not cfg.verbose,
            desc=f"sigma={sigma_float:.2f}",
            ncols=80,
        )
        try:
            for batch in progress:
                if cfg.max_batches is not None and batches_processed >= cfg.max_batches:
                    break
                if batches_processed >= setup.estimated_batches:
                    break

                images_np = batch["image"].astype(np.float32)
                nb_samples = (images_np.shape[0] // setup.device_count) * setup.device_count
                if nb_samples == 0:
                    continue

                images_np = images_np[:nb_samples]
                images = jax.device_put(images_np, setup.image_sharding)

                rng, rng_patches = jax.random.split(rng)
                rng, rng_latent = jax.random.split(rng)
                feats_real, feats_dino, feats_flatdino, psnr_dino, psnr_flatdino = (
                    _run_metrics_step(images, rng_patches, rng_latent)
                )

                feats_real_np = np.asarray(jax.device_get(feats_real), dtype=np.float64)
                feats_dino_np = np.asarray(jax.device_get(feats_dino), dtype=np.float64)
                feats_flatdino_np = np.asarray(jax.device_get(feats_flatdino), dtype=np.float64)
                psnr_dino_np = np.asarray(jax.device_get(psnr_dino), dtype=np.float64)
                psnr_flatdino_np = np.asarray(jax.device_get(psnr_flatdino), dtype=np.float64)

                if grid_dino is None or grid_flatdino is None:
                    original, recon_dino, recon_flatdino = _decode(
                        images,
                        noise_sigma_patches=sigma_float,
                        noise_sigma_latent=sigma_float,
                        rng_patches=rng_patches,
                        rng_latent=rng_latent,
                    )
                    grid_dino = np.asarray(jax.device_get(recon_dino))
                    grid_flatdino = np.asarray(jax.device_get(recon_flatdino))

                if (
                    feats_real_np.shape != feats_dino_np.shape
                    or feats_real_np.shape != feats_flatdino_np.shape
                ):
                    raise RuntimeError("Feature shapes for real and reconstructed images differ.")

                if setup.remaining is not None:
                    take = min(setup.remaining - processed, feats_real_np.shape[0])
                    feats_real_np = feats_real_np[:take]
                    feats_dino_np = feats_dino_np[:take]
                    feats_flatdino_np = feats_flatdino_np[:take]
                    psnr_dino_np = psnr_dino_np[:take]
                    psnr_flatdino_np = psnr_flatdino_np[:take]
                else:
                    take = feats_real_np.shape[0]

                if take <= 0:
                    break

                stats_real.update(feats_real_np)
                stats_recon_dino.update(feats_dino_np)
                stats_recon_flatdino.update(feats_flatdino_np)
                psnr_dino_values.append(psnr_dino_np)
                psnr_flat_values.append(psnr_flatdino_np)
                processed += take
                batches_processed += 1

                if cfg.debug_images and not debug_originals:
                    original, recon_dino, recon_flatdino = _decode(
                        images,
                        noise_sigma_patches=sigma_float,
                        noise_sigma_latent=sigma_float,
                        rng_patches=rng_patches,
                        rng_latent=rng_latent,
                    )
                    original_np = np.asarray(jax.device_get(original))[:DEBUG_IMAGE_COUNT]
                    recon_dino_np = np.asarray(jax.device_get(recon_dino))[:DEBUG_IMAGE_COUNT]
                    recon_flatdino_np = np.asarray(jax.device_get(recon_flatdino))[
                        :DEBUG_IMAGE_COUNT
                    ]
                    debug_originals.append(original_np)
                    debug_dino.append(recon_dino_np)
                    debug_flatdino.append(recon_flatdino_np)

                if setup.remaining is not None and processed >= setup.remaining:
                    break
        finally:
            if isinstance(progress, tqdm):
                progress.close()

        if processed == 0:
            raise RuntimeError("No images were processed for metric computation.")
        if not psnr_dino_values or not psnr_flat_values:
            raise RuntimeError("Failed to accumulate PSNR statistics.")

        real_stats = stats_real.finalize()
        recon_dino_stats = stats_recon_dino.finalize()
        recon_flatdino_stats = stats_recon_flatdino.finalize()

        fid_dino = fid_eval.frechet_distance(real_stats, recon_dino_stats)
        fid_flatdino = fid_eval.frechet_distance(real_stats, recon_flatdino_stats)

        psnr_dino_all = np.concatenate(psnr_dino_values, axis=0)
        psnr_flatdino_all = np.concatenate(psnr_flat_values, axis=0)
        mean_dino, var_dino = _psnr_stats(psnr_dino_all)
        mean_flatdino, var_flatdino = _psnr_stats(psnr_flatdino_all)

        if grid_dino is not None:
            _save_image_grid(grid_dino, "dino", sigma_float)
        if grid_flatdino is not None:
            _save_image_grid(grid_flatdino, "flatdino", sigma_float)

        records.append(
            {
                "sigma": sigma_float,
                "fid_dino": float(fid_dino),
                "fid_flatdino": float(fid_flatdino),
                "psnr_mean_dino": float(mean_dino),
                "psnr_mean_flatdino": float(mean_flatdino),
                "psnr_var_dino": float(var_dino),
                "psnr_var_flatdino": float(var_flatdino),
                "processed_images": int(processed),
            }
        )

        df = pd.DataFrame(records)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    if cfg.debug_images:
        if debug_originals and debug_dino and debug_flatdino:
            originals = np.concatenate(debug_originals, axis=0)[:DEBUG_IMAGE_COUNT]
            recon_dino = np.concatenate(debug_dino, axis=0)[:DEBUG_IMAGE_COUNT]
            recon_flatdino = np.concatenate(debug_flatdino, axis=0)[:DEBUG_IMAGE_COUNT]
            _save_debug_image(originals, recon_dino, recon_flatdino)
            logging.info(f"Saved reconstruction debug image to {DEBUG_IMAGE_PATH}")
        else:
            logging.info("Debug images were requested but no samples were collected.")

    df = pd.DataFrame(records).sort_values("sigma").reset_index(drop=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    sns.lineplot(data=df, x="sigma", y="fid_dino", marker="o", ax=axes[0], label="DINO")
    sns.lineplot(data=df, x="sigma", y="fid_flatdino", marker="s", ax=axes[0], label="FlatDINO")
    axes[0].set_title("FID vs noise level")
    axes[0].set_ylabel("FID (Inception V3)")
    axes[0].set_yscale("log")

    sns.lineplot(data=df, x="sigma", y="psnr_mean_dino", marker="o", ax=axes[1], label="DINO")
    sns.lineplot(
        data=df, x="sigma", y="psnr_mean_flatdino", marker="s", ax=axes[1], label="FlatDINO"
    )
    axes[1].set_title("PSNR vs noise level")
    axes[1].set_ylabel("PSNR (dB)")

    for ax in axes:
        ax.set_xlabel("Noise sigma")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Save final sorted CSV
    df.to_csv(output_csv, index=False)

    # Save results summary to JSON
    zero_row = df[df["sigma"] == 0.0].iloc[0] if 0.0 in df["sigma"].values else df.iloc[0]
    results = {
        "csv_path": str(output_csv),
        "plot_path": str(plot_path),
        "num_noise_levels": cfg.num_noise_levels,
        "noise_levels": df["sigma"].tolist(),
        "fid_dino_at_zero": float(zero_row["fid_dino"]),
        "fid_flatdino_at_zero": float(zero_row["fid_flatdino"]),
        "psnr_dino_at_zero": float(zero_row["psnr_mean_dino"]),
        "psnr_flatdino_at_zero": float(zero_row["psnr_mean_flatdino"]),
    }
    save_eval_results(cfg.path, "noise_levels", results)
    logging.info(f"Saved noise levels results to {cfg.path}/eval_results.json")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
