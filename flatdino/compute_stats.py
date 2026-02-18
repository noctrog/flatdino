"""Compute mean and variance of FlatDINO latent features over ImageNet.

This script computes per-token-position statistics of FlatDINO encoder latents
over the ImageNet dataset using JAX distributed computation. The statistics
can be used for normalizing latent features in downstream tasks.

Unlike the RAE version (which computes raw DINO patch statistics), this script
computes statistics over the FlatDINO encoder's latent tokens.

Usage:
    # Compute mean and variance (mean is computed by default)
    python -m flatdino.compute_stats --checkpoint output/flatdino/vae/my_model

    # Compute only variance (skip mean)
    python -m flatdino.compute_stats --checkpoint output/flatdino/vae/my_model --skip-mean

    # Custom output path
    python -m flatdino.compute_stats --checkpoint output/flatdino/vae/my_model --output ./my_stats.npz

    # Use validation split
    python -m flatdino.compute_stats --checkpoint output/flatdino/vae/my_model --split validation
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import tyro
from tqdm import tqdm
from absl import logging

from flatdino.data import create_dataloaders
from flatdino.eval import restore_encoder
from flatdino.decoder.augmentations import ADMCenterCropAugmentations, ADMCenterCropConfig
from flatdino.distributed import prefetch_to_mesh, is_primary_host, init_distributed, update_running_stats


# JAX configuration
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@partial(jax.jit, static_argnames=("num_register_tokens", "encoder_disposable_registers", "num_flat_tokens", "tanh"))
def encode_to_latents(
    dino,
    encoder,
    num_register_tokens: int,
    encoder_disposable_registers: int,
    num_flat_tokens: int,
    tanh: bool,
    images: jax.Array,
) -> jax.Array:
    """Encode images to FlatDINO latents (mu only, deterministic)."""
    # Extract DINO features
    features = dino.encode(images, deterministic=True)
    # features shape: (B, 1 + num_register + num_patches, hidden_size)
    # Remove CLS and register tokens, keep only patch tokens
    patch_features = features[:, 1 + num_register_tokens :, :]

    # Encode through FlatDINO encoder
    encoded = encoder(patch_features)
    # encoded shape: (B, num_registers, output_dim) where output_dim = 2 * latent_dim

    # Split encoder output into mu and logvar, take only mu
    mu, _ = jnp.split(encoded, 2, axis=-1)
    # Remove disposable registers
    mu = mu[:, encoder_disposable_registers : encoder_disposable_registers + num_flat_tokens]

    # Apply tanh if configured
    if tanh:
        mu = jnp.tanh(mu)

    return mu


@dataclass
class Args:
    checkpoint: Path
    """Path to the FlatDINO VAE checkpoint to use for encoding."""
    output: Path | None = None
    """Output path for the npz file containing mean and variance.
    If None, saves to {checkpoint}/stats.npz."""
    skip_mean: bool = False
    """If True, skip computing the mean. By default, mean is computed and saved."""
    batch_size: int = 256
    """Per-device batch size."""
    num_workers: int = 8
    """Number of data loading workers."""
    distributed: bool = False
    """Enable distributed training for multi-host setups (e.g., TPU pods)."""
    split: Literal["train", "validation"] = "train"
    """Dataset split to use for computing statistics."""
    max_samples: int | None = None
    """Maximum number of samples to process. If None, uses full dataset."""


def main(args: Args) -> None:
    if args.distributed:
        init_distributed()

    # Setup mesh for distributed computation (2D mesh with model axis size 1 for sharding compatibility)
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices, 1), ("data", "model"))
    jax.set_mesh(mesh)

    # Mixed precision policy
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )

    if is_primary_host():
        logging.info(f"Running on {num_devices} devices")
        logging.info(f"Checkpoint: {args.checkpoint}")

    # Restore FlatDINO encoder and DINO
    components = restore_encoder(
        args.checkpoint,
        mesh=mesh,
        mp=mp,
        encoder=True,
        decoder=False,
    )
    dino = components.dino
    encoder = components.encoder
    data_cfg = components.data_cfg
    num_flat_tokens = components.num_flat_tokens
    encoder_disposable_registers = components.encoder_disposable_registers
    tanh = components.tanh

    # Get encoder output dimension (mu is half of the full output)
    latent_dim = encoder.cfg.output_dim // 2

    if is_primary_host():
        logging.info(f"DINO: {dino.config.name_or_path}")
        logging.info(f"FlatDINO encoder: {num_flat_tokens} tokens, {latent_dim} latent dim")
        logging.info(f"Skip mean: {args.skip_mean}")
        logging.info(f"Tanh: {tanh}")

    # Determine output path
    output_path = args.output if args.output is not None else args.checkpoint / "stats.npz"

    # Override data config workers if specified
    if args.num_workers:
        data_cfg.num_workers = args.num_workers

    # Determine which split to use
    if args.split == "train":
        train_split = "train"
        val_split = None
    else:
        train_split = None
        val_split = "validation"

    # ADM center crop to 256, then resize to 224 for DINO (no horizontal flip)
    adm_aug_cfg = ADMCenterCropConfig(crop_size=256, output_size=224, horizontal_flip=False)

    # Create dataloader
    loaders = create_dataloaders(
        data_cfg,
        batch_size=args.batch_size * num_devices,
        train_epochs=1 if train_split else None,
        val_epochs=1 if val_split else None,
        train_aug=ADMCenterCropAugmentations(adm_aug_cfg, data_cfg) if train_split else None,
        val_aug=ADMCenterCropAugmentations(adm_aug_cfg, data_cfg) if val_split else None,
        drop_remainder_train=False,
        drop_remainder_val=False,
        val_shuffle=False,
    )

    loader = loaders.train_loader if train_split else loaders.val_loader
    ds_size = loaders.train_ds_size if train_split else loaders.val_ds_size

    if args.max_samples is not None:
        ds_size = min(ds_size, args.max_samples)

    if is_primary_host():
        logging.info(f"Dataset size: {ds_size} samples")

    # Get DINO feature dimensions
    num_register_tokens = dino.config.num_register_tokens

    # Use Welford's parallel algorithm for numerically stable running statistics
    # This avoids accumulating large sums which can lose precision
    total_count = 0
    running_mean = jnp.zeros((num_flat_tokens, latent_dim), dtype=jnp.float64)
    running_m2 = jnp.zeros((num_flat_tokens, latent_dim), dtype=jnp.float64)

    # Process dataset
    data_iter = iter(loader)
    num_batches = (ds_size + args.batch_size * num_devices - 1) // (args.batch_size * num_devices)

    for batch in tqdm(
        prefetch_to_mesh(data_iter, 1, mesh, trim=True),
        desc="Computing stats",
        total=num_batches,
        disable=not is_primary_host(),
    ):
        images = batch["image"]  # (B, H, W, C)

        if args.max_samples is not None and total_count >= args.max_samples:
            break

        # Get FlatDINO latents
        latents = encode_to_latents(
            dino,
            encoder,
            num_register_tokens,
            encoder_disposable_registers,
            num_flat_tokens,
            tanh,
            images,
        )  # (B, num_tokens, latent_dim)
        latents = latents.astype(jnp.float64)

        # Update running statistics using Welford's parallel algorithm
        total_count, running_mean, running_m2 = update_running_stats(
            total_count, running_mean, running_m2, latents
        )

    # Compute final variance from M2
    mean = running_mean
    variance = running_m2 / total_count

    # Ensure variance is non-negative (can be slightly negative due to numerical precision)
    variance = jnp.maximum(variance, 0.0)

    # Convert to numpy
    mean_np = np.array(mean, dtype=np.float32)
    variance_np = np.array(variance, dtype=np.float32)

    # Save results (only on primary host)
    if is_primary_host():
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {"var": variance_np}
        if not args.skip_mean:
            save_dict["mean"] = mean_np

        np.savez(output_path, **save_dict)

        logging.info(f"Saved statistics to {output_path}")
        logging.info(f"  Samples processed: {total_count}")
        logging.info(f"  Shape: ({num_flat_tokens}, {latent_dim})")
        if not args.skip_mean:
            logging.info(f"  Mean range: [{mean_np.min():.6f}, {mean_np.max():.6f}]")
            logging.info(f"  Mean mean: {mean_np.mean():.6f}")
        logging.info(f"  Variance range: [{variance_np.min():.6f}, {variance_np.max():.6f}]")
        logging.info(f"  Variance mean: {variance_np.mean():.6f}")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    args = tyro.cli(Args)
    main(args)
