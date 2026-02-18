"""Generate reconstructed images for rFID evaluation.

This script reconstructs images through the FlatDINO VAE pipeline and saves them
in a format compatible with OpenAI's ADM evaluation suite (guided-diffusion).

Pipeline: validation image (uint8) → normalize → DINOv2 → FlatDINO encoder → sample z → FlatDINO decoder → RAE decoder → reconstruction

Images are loaded from the ImageNet validation set (50k images), center-cropped to
256x256 using ADM preprocessing. The validation NPZ is auto-created from tfds
if it doesn't exist in the cache (~/.cache/adm/imagenet256_val.npz).

IMPORTANT: rFID compares reconstructions to the ORIGINAL validation images,
not the VIRTUAL reference batch (which is for gFID).

Output directory defaults to flatdino_path/recon_samples.

Usage:
    python -m flatdino.eval.rfid_adm \
        --flatdino-path output/flatdino/vae/model

After running, use the printed command to compute rFID with the ADM evaluator.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax

# Use full float32 precision for matmuls to avoid TF32 numerical errors on GPU
jax.config.update("jax_default_matmul_precision", "float32")

import flax.nnx as nnx
import jax.numpy as jnp
import jmp
import numpy as np
import orbax.checkpoint as ocp
import tyro
from absl import logging
from dacite import Config as DaciteConfig, from_dict
from jax.sharding import NamedSharding, PartitionSpec as P
from PIL import Image
from tqdm import tqdm

from flatdino.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flatdino.metrics.adm import get_imagenet_256_val
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig


@dataclass
class Config:
    flatdino_path: Path
    """Directory containing the FlatDINO checkpoint (encoder and decoder required)."""

    output_dir: Path | None = None
    """Directory to save reconstructed images. Defaults to flatdino_path/recon_samples."""

    num_samples: int | None = None
    """Number of samples to reconstruct. Defaults to all images in reference."""

    batch_size: int = 256
    """Batch size for reconstruction."""

    seed: int = 0

    use_mean: bool = True
    """If True, use the mean (mu) instead of sampling from the latent distribution."""

    save_npz: bool = True
    """Save all samples to a single NPZ file (for ADM evaluator)."""

    save_png: bool = False
    """Save individual PNG files."""


@nnx.jit(static_argnames=("image_h", "image_w", "use_mean"))
def reconstruct_batch(
    images_uint8: jax.Array,
    key: jax.Array,
    dino,
    flatdino: FlatDinoAutoencoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    image_h: int,
    image_w: int,
    use_mean: bool,
) -> jax.Array:
    """Reconstruct a batch of images through the FlatDINO VAE pipeline.

    Args:
        images_uint8: Input images in uint8 format [0, 255], shape (B, H, W, C)
    """
    b = images_uint8.shape[0]

    # Convert uint8 [0, 255] to float [0, 1] then normalize
    images = images_uint8.astype(jnp.float32) / 255.0
    images_norm = (images - mean) / std

    # Resize to DINO resolution and extract patches
    dino_input = jax.image.resize(
        images_norm,
        (b, dino.resolution, dino.resolution, images_norm.shape[-1]),
        method="linear",
        antialias=True,
    )
    patches = dino(dino_input)[:, 5:]  # Remove CLS + register tokens

    # Encode to latent space using FlatDinoAutoencoder
    # If use_mean, don't pass key; otherwise pass key for sampling
    if use_mean:
        z, _ = flatdino.encode(patches)
    else:
        z, _ = flatdino.encode(patches, key=key)

    # Decode back to DINO patch space
    recon_tokens = flatdino.decode(z)

    # Decode to image space using RAE decoder
    decoder_out = rae_decoder(recon_tokens)
    recon = rae_decoder.unpatchify(decoder_out.logits, original_image_size=(image_h, image_w))
    recon = jnp.transpose(recon, (0, 2, 3, 1))

    # Denormalize
    return jnp.clip(recon * std + mean, 0.0, 1.0)


def fp_to_uint8(images: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] images to uint8 [0, 255]."""
    return np.clip(images * 255 + 0.5, 0, 255).astype(np.uint8)


def main(cfg: Config):
    # Compute default output directory if not specified
    output_dir = cfg.output_dir
    if output_dir is None:
        suffix = "_mean" if cfg.use_mean else ""
        output_dir = cfg.flatdino_path / f"recon_samples{suffix}"

    logging.info("Generating reconstructed images for ADM FID evaluation")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Use mean (no sampling): {cfg.use_mean}")

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)

    if cfg.batch_size % mesh.size != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by the number of devices ({mesh.size})"
        )
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.float32, output_dtype=jnp.float32)
    dacite_config = DaciteConfig(cast=[tuple], strict=False)

    # Load validation images from NPZ (auto-created from ImageNet val if needed)
    reference_path = get_imagenet_256_val()
    logging.info(f"Loading reference images from {reference_path}")
    ref_data = np.load(reference_path)
    ref_images = ref_data["arr_0"]  # Shape: (N, H, W, C), dtype: uint8
    logging.info(f"Loaded {ref_images.shape[0]} reference images, shape: {ref_images.shape}")

    total_available = ref_images.shape[0]
    num_samples = cfg.num_samples if cfg.num_samples is not None else total_available
    num_samples = min(num_samples, total_available)

    # Load checkpoint and config
    ckpt_path = cfg.flatdino_path.absolute()
    mngr = ocp.CheckpointManager(
        ckpt_path,
        item_names=FlatDinoAutoencoder.get_item_names() + ["optim", "loader", "config"],
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    step = mngr.best_step()
    assert step is not None, "Failed to load best step from checkpoint"
    logging.info(f"Found checkpoint at step {step}. Restoring...")

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

    # Setup dimensions - infer from reference images
    image_h, image_w = ref_images.shape[1], ref_images.shape[2]

    num_output_patches = flatdino_cfg.num_output_patches

    # Setup RAE decoder for image reconstruction
    rae_decoder = make_rae_decoder(
        num_patches=num_output_patches,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    # Normalization constants
    mean = jnp.array(IMAGENET_DEFAULT_MEAN, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(IMAGENET_DEFAULT_STD, dtype=jnp.float32)[None, None, None, :]

    shard = NamedSharding(mesh, P("data"))

    reconstruct_fn = partial(
        reconstruct_batch,
        dino=dino,
        flatdino=flatdino,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        image_h=image_h,
        image_w=image_w,
        use_mean=cfg.use_mean,
    )

    key = jax.random.PRNGKey(cfg.seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage for all samples (for NPZ)
    all_samples = []
    total_collected = 0

    logging.info(f"Reconstructing {num_samples} images from reference batch")
    logging.info(f"Batch size: {cfg.batch_size}")

    pbar = tqdm(total=num_samples, desc="Reconstructing")

    for batch_start in range(0, num_samples, cfg.batch_size):
        batch_end = min(batch_start + cfg.batch_size, num_samples)
        images = ref_images[batch_start:batch_end]
        batch_len = images.shape[0]

        # Pad batch to be divisible by number of devices
        if batch_len % mesh.size != 0:
            pad_size = mesh.size - (batch_len % mesh.size)
            images = np.concatenate([images, np.repeat(images[-1:], pad_size, axis=0)], axis=0)
        else:
            pad_size = 0

        images = jax.device_put(jnp.array(images), shard)
        key, sample_key = jax.random.split(key)

        recon = reconstruct_fn(images, sample_key)

        # Get images as numpy, removing padding
        recon_np = np.asarray(jax.device_get(recon))
        if pad_size > 0:
            recon_np = recon_np[:-pad_size]

        # Convert to uint8 for ADM evaluator
        recon_uint8 = fp_to_uint8(recon_np)

        # Store for NPZ
        if cfg.save_npz:
            all_samples.append(recon_uint8)

        # Save individual PNGs if requested
        if cfg.save_png:
            for i, img in enumerate(recon_uint8):
                img_idx = total_collected + i
                img_path = output_dir / "images" / f"sample_{img_idx:06d}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(img_path)

        total_collected += batch_len - pad_size if pad_size > 0 else batch_len
        pbar.update(batch_len - pad_size if pad_size > 0 else batch_len)

    pbar.close()

    # Save NPZ file for ADM evaluator
    if cfg.save_npz:
        all_samples = np.concatenate(all_samples, axis=0)
        npz_path = output_dir / "samples.npz"
        np.savez(npz_path, arr_0=all_samples)
        logging.info(f"Saved {all_samples.shape[0]} samples to {npz_path}")
        logging.info(f"Sample shape: {all_samples.shape}, dtype: {all_samples.dtype}")

    if cfg.save_png:
        logging.info(f"Saved individual PNGs to {output_dir / 'images'}")

    # Save generation config for reference
    config_path = output_dir / "reconstruction_config.txt"
    with open(config_path, "w") as f:
        f.write(f"reference_path: {reference_path}\n")
        f.write(f"flatdino_path: {cfg.flatdino_path}\n")
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"total_reconstructed: {total_collected}\n")
        f.write(f"use_mean: {cfg.use_mean}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"image_size: {image_h}x{image_w}\n")
        f.write(f"num_flat_tokens: {flatdino_cfg.num_latents}\n")
        f.write(f"num_patch_tokens: {num_output_patches}\n")

    logging.info(f"Reconstruction complete. Config saved to {config_path}")

    # Print the command to compute rFID
    logging.info("")
    logging.info("To compute rFID, run one of the following commands:")
    logging.info(f"  CUDA: uv run --no-project --python=3.12 --with='tensorflow[and-cuda],scipy,requests,tqdm' flatdino/metrics/adm.py {npz_path} --ref-batch {reference_path}")
    logging.info(f"  TPU:  uv run --no-project --python=3.12 --with='tensorflow,scipy,requests,tqdm' flatdino/metrics/adm.py {npz_path} --ref-batch {reference_path}")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    cfg = tyro.cli(Config)
    main(cfg)
