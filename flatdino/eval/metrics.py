import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import chex
import flax.nnx as nnx
import jax
import jax.image
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import tyro
from dacite import Config as DaciteConfig, from_dict
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm import tqdm

import flatdino.metrics.fid as fid_eval
from flatdino.data import DataConfig, DataLoaders, create_dataloaders
from flatdino.metrics.fid import FIDRunningStats
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.pretrained.inception import InceptionV3
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import save_eval_results
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig


@dataclass
class Config:
    path: Path
    data: DataConfig = field(default_factory=lambda: DataConfig())
    gpu_batch_size: int = 128
    num_eval_images: int | None = None
    max_batches: int | None = None
    seed: int = 0
    verbose: bool = False
    debug_images: bool = False


DEBUG_IMAGE_PATH = Path("/tmp/metrics_debug.png")
DEBUG_IMAGE_COUNT = 6
DEBUG_IMAGE_COLS = 3


def _make_image_grid(
    images: np.ndarray,
    *,
    cols: int = DEBUG_IMAGE_COLS,
    fill_value: float = 1.0,
) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError("Expected images with shape (N, H, W, C).")

    count, height, width, channels = images.shape
    rows = math.ceil(count / cols)
    grid = np.full(
        (rows * height, cols * width, channels),
        fill_value,
        dtype=np.float32,
    )

    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        top, left = row * height, col * width
        grid[top : top + height, left : left + width] = img

    return grid


def _save_debug_image(
    originals: np.ndarray,
    recon_dino: np.ndarray,
    recon_flatdino: np.ndarray,
):
    originals = np.clip(originals, 0.0, 1.0)
    recon_dino = np.clip(recon_dino, 0.0, 1.0)
    recon_flatdino = np.clip(recon_flatdino, 0.0, 1.0)

    orig_grid = _make_image_grid(originals)
    dino_grid = _make_image_grid(recon_dino)
    flat_grid = _make_image_grid(recon_flatdino)

    spacer = np.ones((8, orig_grid.shape[1], orig_grid.shape[2]), dtype=np.float32)
    combined = np.concatenate([orig_grid, spacer, dino_grid, spacer, flat_grid], axis=0)
    plt.imsave(DEBUG_IMAGE_PATH, combined)


def _psnr_stats(psnr_values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(psnr_values))
    var = float(np.var(psnr_values))
    return mean, var


@dataclass
class EvalComponents:
    """Components needed for FlatDINO evaluation."""

    dino: nnx.Module
    flatdino: FlatDinoAutoencoder
    flatdino_cfg: FlatDinoConfig
    rae_decoder: nnx.Module
    mean: jax.Array
    std: jax.Array
    inception_forward: Callable[[jax.Array], jax.Array]
    image_sharding: NamedSharding
    device_count: int
    collect_batch_size: int
    remaining: int | None
    estimated_batches: int
    data: DataLoaders
    num_output_patches: int


def prepare_eval_components(cfg: Config, *, val_epochs: int | None = 1) -> EvalComponents:
    device_count = jax.device_count()
    if device_count == 0:
        raise RuntimeError("No visible JAX devices.")
    if cfg.gpu_batch_size <= 0:
        raise ValueError("gpu_batch_size must be positive.")
    if cfg.num_eval_images is not None and cfg.num_eval_images <= 0:
        raise ValueError("num_eval_images must be positive when provided.")
    if cfg.max_batches is not None and cfg.max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    mesh = jax.make_mesh((device_count, 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)
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
            f"Decoder expects {num_output_patches} output patches but DINO produced {num_patch_tokens}."
        )

    rae_decoder = make_rae_decoder(
        num_patches=num_output_patches,
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

    collect_batch_size = cfg.gpu_batch_size * device_count
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

    return EvalComponents(
        dino=dino,
        flatdino=flatdino,
        flatdino_cfg=flatdino_cfg,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        inception_forward=inception_forward,
        image_sharding=image_sharding,
        device_count=device_count,
        collect_batch_size=collect_batch_size,
        remaining=remaining,
        estimated_batches=estimated_batches,
        data=data,
        num_output_patches=num_output_patches,
    )


def decode_images(
    dino,
    flatdino: FlatDinoAutoencoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    images: jax.Array,
    *,
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


@nnx.jit(static_argnames=("inception_forward",))
def _metrics_step(
    dino,
    flatdino: FlatDinoAutoencoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    inception_forward: Callable[[jax.Array], jax.Array],
    images: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    def _to_inception_features(pixels: jax.Array) -> jax.Array:
        resized = jax.image.resize(
            pixels, (pixels.shape[0], 299, 299, 3), method="bicubic", antialias=False
        )
        normalized = resized * 2.0 - 1.0
        return inception_forward(normalized)

    original, recon_dino, recon_flatdino = decode_images(
        dino,
        flatdino,
        rae_decoder,
        mean,
        std,
        images,
    )

    feats_real = _to_inception_features(original)
    feats_dino = _to_inception_features(recon_dino)
    feats_flatdino = _to_inception_features(recon_flatdino)

    mse_dino = jnp.mean(jnp.square(recon_dino - original), axis=(1, 2, 3))
    mse_flatdino = jnp.mean(jnp.square(recon_flatdino - original), axis=(1, 2, 3))
    psnr_dino = -10.0 * jnp.log10(jnp.maximum(mse_dino, 1e-10))
    psnr_flatdino = -10.0 * jnp.log10(jnp.maximum(mse_flatdino, 1e-10))
    return feats_real, feats_dino, feats_flatdino, psnr_dino, psnr_flatdino


@nnx.jit
def _reconstruct_images(
    dino,
    flatdino: FlatDinoAutoencoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    images: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return decode_images(
        dino,
        flatdino,
        rae_decoder,
        mean,
        std,
        images,
    )


def main(cfg: Config):
    ec = prepare_eval_components(cfg)

    stats_real = FIDRunningStats()
    stats_recon_dino = FIDRunningStats()
    stats_recon_flatdino = FIDRunningStats()
    psnr_dino_values: list[np.ndarray] = []
    psnr_flat_values: list[np.ndarray] = []
    debug_originals: list[np.ndarray] = []
    debug_dino: list[np.ndarray] = []
    debug_flatdino: list[np.ndarray] = []
    processed = 0
    batches_processed = 0

    loader_iter = iter(ec.data.val_loader)
    progress = tqdm(
        loader_iter,
        total=ec.estimated_batches,
        disable=not cfg.verbose,
        desc="metrics",
        ncols=80,
    )
    try:
        for batch in progress:
            if cfg.max_batches is not None and batches_processed >= cfg.max_batches:
                break

            images_np = batch["image"].astype(np.float32)
            nb_samples = (images_np.shape[0] // ec.device_count) * ec.device_count
            if nb_samples == 0:
                continue

            images_np = images_np[:nb_samples]
            images = jax.device_put(images_np, ec.image_sharding)

            feats_real, feats_dino, feats_flatdino, psnr_dino, psnr_flatdino = _metrics_step(
                ec.dino,
                ec.flatdino,
                ec.rae_decoder,
                ec.mean,
                ec.std,
                ec.inception_forward,
                images,
            )

            feats_real_np = np.asarray(jax.device_get(feats_real), dtype=np.float64)
            feats_dino_np = np.asarray(jax.device_get(feats_dino), dtype=np.float64)
            feats_flatdino_np = np.asarray(jax.device_get(feats_flatdino), dtype=np.float64)
            psnr_dino_np = np.asarray(jax.device_get(psnr_dino), dtype=np.float64)
            psnr_flatdino_np = np.asarray(jax.device_get(psnr_flatdino), dtype=np.float64)

            if (
                feats_real_np.shape != feats_dino_np.shape
                or feats_real_np.shape != feats_flatdino_np.shape
            ):
                raise RuntimeError("Feature shapes for real and reconstructed images differ.")

            if ec.remaining is not None:
                take = min(ec.remaining - processed, feats_real_np.shape[0])
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

            if cfg.debug_images and len(debug_originals) < DEBUG_IMAGE_COUNT:
                original, recon_dino, recon_flatdino = _reconstruct_images(
                    ec.dino,
                    ec.flatdino,
                    ec.rae_decoder,
                    ec.mean,
                    ec.std,
                    images,
                )
                remaining_debug = DEBUG_IMAGE_COUNT - len(debug_originals)
                original_np = np.asarray(jax.device_get(original))[:remaining_debug]
                recon_dino_np = np.asarray(jax.device_get(recon_dino))[:remaining_debug]
                recon_flatdino_np = np.asarray(jax.device_get(recon_flatdino))[:remaining_debug]

                debug_originals.append(original_np)
                debug_dino.append(recon_dino_np)
                debug_flatdino.append(recon_flatdino_np)

            if ec.remaining is not None and processed >= ec.remaining:
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

    print(f"Processed images: {processed}")
    print(f"FID DINO patches (Inception V3): {fid_dino:.4f}")
    print(f"FID FlatDINO (Inception V3): {fid_flatdino:.4f}")
    print("PSNR statistics (dB)")
    print(f"{'Model':<16}{'Mean':>12}{'Variance':>12}")
    print(f"{'DINO patches':<16}{mean_dino:>12.4f}{var_dino:>12.4f}")
    print(f"{'FlatDINO':<16}{mean_flatdino:>12.4f}{var_flatdino:>12.4f}")

    if cfg.debug_images:
        if debug_originals and debug_dino and debug_flatdino:
            originals = np.concatenate(debug_originals, axis=0)[:DEBUG_IMAGE_COUNT]
            recon_dino = np.concatenate(debug_dino, axis=0)[:DEBUG_IMAGE_COUNT]
            recon_flatdino = np.concatenate(debug_flatdino, axis=0)[:DEBUG_IMAGE_COUNT]
            _save_debug_image(originals, recon_dino, recon_flatdino)
            print(f"Saved reconstruction debug image to {DEBUG_IMAGE_PATH}")
        else:
            print("Debug images were requested but no samples were collected.")

    # Save results to JSON
    results = {
        "processed_images": processed,
        "fid_dino": fid_dino,
        "fid_flatdino": fid_flatdino,
        "psnr_dino_mean": mean_dino,
        "psnr_dino_var": var_dino,
        "psnr_flatdino_mean": mean_flatdino,
        "psnr_flatdino_var": var_flatdino,
    }
    save_eval_results(cfg.path, "metrics", results)
    print(f"Saved metrics to {cfg.path}/eval_results.json")


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
