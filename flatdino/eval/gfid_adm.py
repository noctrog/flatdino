"""Generate images for FID evaluation with ADM evaluator.

This script generates images and saves them in a format compatible with OpenAI's
ADM evaluation suite (guided-diffusion). Images are saved as:
1. A single NPZ file with all samples (for ADM evaluator)
2. Optionally, individual PNG files organized by class

Output directory defaults to generator_path/samples_cfg{scale} (or samples_nocfg).

Usage:
    # Generate samples with CFG
    python -m flatdino.eval.gfid_adm \
        --generator-path output/flatdino-generator/model \
        --flatdino-path output/flatdino/vae/model \
        --cfg-scale 2.0

    # Generate without CFG
    python -m flatdino.eval.gfid_adm \
        --generator-path output/flatdino-generator/model \
        --flatdino-path output/flatdino/vae/model

After generation, run the ADM evaluator separately. See flatdino/metrics/adm.py.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import jmp
import numpy as np
from PIL import Image
import tyro
from tqdm import tqdm
from absl import logging

from flatdino.models.vit import ViTEncoder
from flatdino.eval import restore_encoder
from flatdino.decoder.sampling import (
    _infer_time_dist_shift,
    _restore_generator,
    _sample_latents,
)
from flatdino.pretrained.rae_decoder import make_rae_decoder


def _default_output_dir(generator_path: Path, cfg_scale: float | None, t_min: float, t_max: float) -> Path:
    """Compute default output directory based on generator path and CFG settings."""
    if cfg_scale is None:
        folder_name = "samples_nocfg"
    else:
        # Include interval in name only if non-default
        if t_min == 0.3 and t_max == 1.0:
            folder_name = f"samples_cfg{cfg_scale}"
        else:
            folder_name = f"samples_cfg{cfg_scale}_t{t_min}-{t_max}"
    return generator_path / folder_name


@dataclass
class Config:
    generator_path: Path
    """Directory containing checkpoints produced by train_generator.py."""

    flatdino_path: Path
    """Directory containing the FlatDINO autoencoder checkpoint."""

    output_dir: Path | None = None
    """Directory to save generated images. Defaults to generator_path/samples_cfg{scale}."""

    steps: int = 50
    """Number of flow-matching integration steps."""

    per_class: int = 50
    """Number of samples to draw per class (50 -> 50_000 for ImageNet)."""

    batch_size: int = 256
    """Batch size used for generation."""

    seed: int = 0
    use_ema: bool = True

    generator_step: int | None = None
    """Checkpoint step to restore. If None, uses the latest step."""

    cfg_scale: float | None = None
    """CFG scale. None disables CFG."""

    t_min: float = 0.3
    """Minimum timestep for applying CFG (DDT default: 0.3)."""

    t_max: float = 1.0
    """Maximum timestep for applying CFG."""

    save_npz: bool = True
    """Save all samples to a single NPZ file (for ADM evaluator)."""

    save_png: bool = False
    """Save individual PNG files organized by class."""


def _maybe_asdict(cfg) -> dict:
    if hasattr(cfg, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(cfg)
    return cfg


@nnx.jit(static_argnames=("patch_tokens", "image_h", "image_w"))
def decode_latents(
    flatdino_decoder, rae_decoder, decoder_input, std, mean, patch_tokens, image_h, image_w
):
    decoder_tokens = flatdino_decoder(decoder_input, deterministic=True)[:, :patch_tokens]
    decoder_out = rae_decoder(decoder_tokens)
    recon = rae_decoder.unpatchify(
        decoder_out.logits,
        original_image_size=(image_h, image_w),
    )
    recon = jnp.transpose(recon, (0, 2, 3, 1))
    return jnp.clip(recon * std + mean, 0.0, 1.0)


def generate_batch(
    labels: jax.Array,
    sample_key: jax.Array,
    generator: nnx.Module,
    flatdino_decoder: ViTEncoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    image_h: int,
    image_w: int,
    steps: int,
    latent_shape: tuple[int, ...],
    cfg_scale: float | None,
    cfg_interval_min: float,
    cfg_interval_max: float,
    time_dist_shift: float | None,
    pred_type: str,
) -> jax.Array:
    latents = _sample_latents(
        generator,
        labels=labels,
        steps=steps,
        key=sample_key,
        latent_shape=latent_shape,
        cfg_scale=cfg_scale,
        cfg_interval_min=cfg_interval_min,
        cfg_interval_max=cfg_interval_max,
        time_dist_shift=time_dist_shift,
        pred_type=pred_type,
    )
    if isinstance(latents, dict):
        latents = latents["x"]

    # Denormalize latents if normalization was used during training
    latents = generator.denormalize(latents)

    patch_tokens = flatdino_decoder.num_reg
    return decode_latents(
        flatdino_decoder, rae_decoder, latents, std, mean, patch_tokens, image_h, image_w
    )


def fp_to_uint8(images: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] images to uint8 [0, 255]."""
    return np.clip(images * 255 + 0.5, 0, 255).astype(np.uint8)


def main(cfg: Config):
    # Compute default output directory if not specified
    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = _default_output_dir(
            cfg.generator_path, cfg.cfg_scale, cfg.t_min, cfg.t_max
        )

    logging.info("Generating images for ADM FID evaluation")
    logging.info(f"Output directory: {output_dir}")
    logging.info(
        f"CFG scale: {cfg.cfg_scale}, interval: [{cfg.t_min}, {cfg.t_max}]"
    )

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)

    if cfg.batch_size % mesh.size != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by the number of devices ({mesh.size})"
        )
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.float32, output_dtype=jnp.float32)

    # Restore FlatDINO encoder/decoder
    restored = restore_encoder(cfg.flatdino_path, mesh=mesh, mp=mp, encoder=True, decoder=True)
    if restored.encoder is None:
        raise RuntimeError("FlatDINO checkpoint does not include encoder weights.")
    if restored.decoder is None:
        raise RuntimeError("FlatDINO checkpoint does not include decoder weights.")
    if restored.data_cfg is None or restored.aug_cfg is None:
        raise RuntimeError("FlatDINO checkpoint is missing data or augmentation configuration.")

    latent_tokens = restored.encoder.num_reg

    # Restore generator
    generator, generator_cfg = _restore_generator(
        cfg.generator_path,
        mesh=mesh,
        mp=mp,
        latent_tokens=latent_tokens,
        use_ema=cfg.use_ema,
        step=cfg.generator_step,
    )
    generator.eval()
    restored.decoder.eval()

    generator_cfg = _maybe_asdict(generator_cfg)

    # Setup dimensions
    patch_tokens = restored.decoder.num_reg
    image_h, image_w = restored.aug_cfg.image_size
    mean_vals = restored.data_cfg.normalization_mean
    std_vals = restored.data_cfg.normalization_std

    model_type = generator_cfg.get("model_type", "dit")
    latent_dim = generator.cfg.in_channels
    dit_dh_cfg = generator_cfg.get("dit_dh", {}) if isinstance(generator_cfg, dict) else {}
    patch_size = dit_dh_cfg.get("patch_size", None) if isinstance(dit_dh_cfg, dict) else None
    is_flat_dit_dh = model_type == "dit_dh" and patch_size is None

    if model_type == "dit_dh" and not is_flat_dit_dh:
        grid = int(jnp.sqrt(latent_tokens))
        if grid * grid != latent_tokens:
            raise ValueError(f"Latent tokens ({latent_tokens}) are not a perfect square for DiTDH.")
        latent_shape = (grid, grid, latent_dim)
    else:
        latent_shape = (latent_tokens, latent_dim)

    # Infer time distribution shift
    time_dist_shift = _infer_time_dist_shift(
        generator_cfg,
        latent_tokens=latent_tokens,
        latent_dim=latent_dim,
    )

    # Setup RAE decoder for image reconstruction
    rae_decoder = make_rae_decoder(
        num_patches=patch_tokens,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    mean = jnp.array(mean_vals, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(std_vals, dtype=jnp.float32)[None, None, None, :]

    shard = NamedSharding(mesh, P("data"))

    generate_fn = partial(
        generate_batch,
        generator=generator,
        flatdino_decoder=restored.decoder,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        image_h=image_h,
        image_w=image_w,
        steps=cfg.steps,
        latent_shape=latent_shape,
        cfg_scale=cfg.cfg_scale,
        cfg_interval_min=cfg.t_min,
        cfg_interval_max=cfg.t_max,
        time_dist_shift=time_dist_shift,
        pred_type=generator_cfg.get("pred_type", "v"),
    )

    num_classes = generator.cfg.num_classes
    total_samples = num_classes * cfg.per_class

    # Generate labels: [0,0,...,1,1,...,2,2,...] with per_class of each
    labels = jnp.arange(num_classes, dtype=jnp.int32)
    labels = jnp.repeat(labels, cfg.per_class)

    key = jax.random.PRNGKey(cfg.seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage for all samples and labels (for NPZ)
    all_samples = []
    all_labels = []

    # Track samples per class for PNG saving
    class_counts = {c: 0 for c in range(num_classes)}

    total_batches = (total_samples + cfg.batch_size - 1) // cfg.batch_size
    logging.info(
        f"Generating {total_samples} images ({cfg.per_class} per class, {num_classes} classes)"
    )
    logging.info(f"Total batches: {total_batches}, batch size: {cfg.batch_size}")

    for batch_idx in tqdm(range(total_batches), desc="Generating"):
        start = batch_idx * cfg.batch_size
        end = min(start + cfg.batch_size, total_samples)
        valid = end - start

        label_batch = jax.lax.dynamic_slice(labels, (start,), (valid,))
        key, sample_key = jax.random.split(key)

        # Pad if necessary
        if valid < cfg.batch_size:
            pad = cfg.batch_size - valid
            label_batch = jnp.concatenate(
                [label_batch, jnp.repeat(label_batch[-1:], pad, axis=0)], axis=0
            )

        label_batch = jax.device_put(label_batch, shard)
        images = generate_fn(label_batch, sample_key)

        # Get images as numpy (only valid samples)
        images_np = np.asarray(jax.device_get(images))[:valid]
        labels_np = np.asarray(jax.device_get(label_batch))[:valid]

        # Convert to uint8 for ADM evaluator
        images_uint8 = fp_to_uint8(images_np)

        # Store for NPZ
        if cfg.save_npz:
            all_samples.append(images_uint8)
            all_labels.append(labels_np)

        # Save individual PNGs if requested
        if cfg.save_png:
            for img, lbl in zip(images_uint8, labels_np):
                class_dir = output_dir / "images" / f"class_{lbl:04d}"
                class_dir.mkdir(parents=True, exist_ok=True)

                img_idx = class_counts[int(lbl)]
                img_path = class_dir / f"sample_{img_idx:04d}.png"
                Image.fromarray(img).save(img_path)
                class_counts[int(lbl)] += 1

    # Save NPZ file for ADM evaluator
    if cfg.save_npz:
        all_samples = np.concatenate(all_samples, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        npz_path = output_dir / "samples.npz"
        np.savez(npz_path, arr_0=all_samples, labels=all_labels)
        logging.info(f"Saved {all_samples.shape[0]} samples to {npz_path}")
        logging.info(f"Sample shape: {all_samples.shape}, dtype: {all_samples.dtype}")
        logging.info(f"Labels shape: {all_labels.shape}, dtype: {all_labels.dtype}")

    if cfg.save_png:
        logging.info(f"Saved individual PNGs to {output_dir / 'images'}")

    # Save generation config for reference
    config_path = output_dir / "generation_config.txt"
    with open(config_path, "w") as f:
        f.write(f"generator_path: {cfg.generator_path}\n")
        f.write(f"flatdino_path: {cfg.flatdino_path}\n")
        f.write(f"steps: {cfg.steps}\n")
        f.write(f"per_class: {cfg.per_class}\n")
        f.write(f"total_samples: {total_samples}\n")
        f.write(f"cfg_scale: {cfg.cfg_scale}\n")
        f.write(f"cfg_interval_min: {cfg.t_min}\n")
        f.write(f"cfg_interval_max: {cfg.t_max}\n")
        f.write(f"time_dist_shift: {time_dist_shift}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"use_ema: {cfg.use_ema}\n")
        f.write(f"image_size: {image_h}x{image_w}\n")

    logging.info(f"Generation complete. Config saved to {config_path}")
    logging.info("Run ADM evaluator with one of the following commands:")
    logging.info(f"  CUDA: uv run --no-project --python=3.12 --with='tensorflow[and-cuda],scipy,requests,tqdm' flatdino/metrics/adm.py {npz_path}")
    logging.info(f"  TPU:  uv run --no-project --python=3.12 --with='tensorflow,scipy,requests,tqdm' flatdino/metrics/adm.py {npz_path}")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    cfg = tyro.cli(Config)
    main(cfg)
