from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import matplotlib.pyplot as plt
import tyro
from tqdm import tqdm
from absl import logging

# Specific project imports
from flatdino.data import DataConfig, create_dataloaders
from flatdino.augmentations import FlatDinoTrainAugmentations, FlatDinoValAugmentations
from flatdino.eval import extract_mu, restore_encoder
from flatdino.distributed import prefetch_to_mesh

# JAX Configuration
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)


@dataclass
class Config:
    flatdino_path: Path
    """Checkpoint directory produced by flatdino/train.py."""

    batch_size: int = 64
    max_batches: int | None = None
    output_dir: Path | None = None
    log_every: int = 20


def _get_radial_indices(h: int, w: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Precomputes the integer radius (distance from center) for every pixel in an HxW grid.
    Returns: (radius_map, sorted_indices, max_radius)
    """
    y, x = np.indices((h, w))
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0

    # Distance from center
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r_int = r.astype(int)

    # We only care about frequencies up to the edge (Nyquist)
    max_r = min(h, w) // 2

    return r_int, max_r


def _compute_radial_profile(batch_images: np.ndarray) -> np.ndarray:
    """
    Computes the radially averaged power spectrum for a batch of 2D images.
    Input: (B, H, W, C)
    Output: (Max_Radius,)
    """
    # 1. FFT over spatial dims (axes 1, 2)
    # Output shape: (B, H, W, C)
    fft = np.fft.fft2(batch_images, axes=(1, 2), norm="ortho")

    # 2. Shift zero freq to center
    fft_shifted = np.fft.fftshift(fft, axes=(1, 2))
    magnitude = np.abs(fft_shifted)

    # Average over batch and channels -> (H, W)
    avg_mag = magnitude.mean(axis=(0, 3))

    h, w = avg_mag.shape
    r_int, max_r = _get_radial_indices(h, w)

    # 3. Radial average
    # Sum amplitudes falling into each integer radius bin
    tbin = np.bincount(r_int.ravel(), weights=avg_mag.ravel())
    nr = np.bincount(r_int.ravel())

    # Avoid division by zero
    radial_profile = tbin / np.maximum(nr, 1)

    # Crop to relevant Nyquist limit
    return radial_profile[:max_r]


def _compute_1d_profile(batch_seq: np.ndarray) -> np.ndarray:
    """
    Computes the spectrum for 1D sequences.
    Input: (B, T, C)
    Output: (T // 2,)
    """
    # 1. FFT over sequence dim (axis 1)
    fft = np.fft.fft(batch_seq, axis=1, norm="ortho")
    magnitude = np.abs(fft)

    # Average over batch and channels -> (T,)
    avg_mag = magnitude.mean(axis=(0, 2))

    # 2. Keep only positive frequencies (first half)
    half_n = avg_mag.shape[0] // 2
    return avg_mag[:half_n]


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    mx = curve.max() if curve.size else 0.0
    if mx <= 1e-9:
        return curve
    return curve / mx


def main(cfg: Config):
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)

    restored = restore_encoder(
        cfg.flatdino_path,
        mesh=mesh,
        mp=mp,
        encoder=True,
        decoder=True,
    )
    if restored.encoder is None or restored.decoder is None:
        raise RuntimeError("Checkpoint must include encoder and decoder weights.")

    data_cfg: DataConfig = restored.data_cfg or DataConfig()
    aug_cfg = restored.aug_cfg
    if aug_cfg is None:
        raise RuntimeError("Augmentation config missing.")

    loaders = create_dataloaders(
        data_cfg,
        cfg.batch_size,
        train_aug=FlatDinoTrainAugmentations(aug_cfg, data_cfg),
        val_aug=FlatDinoValAugmentations(aug_cfg, data_cfg),
        val_epochs=1,
        drop_remainder_train=True,
        drop_remainder_val=True,
    )
    val_iter = iter(loaders.val_loader)
    stream = prefetch_to_mesh(val_iter, 1, mesh)

    # Accumulators for the profiles
    # We initialize them as None and allocate on first batch to know sizes
    acc_rgb = None
    acc_dino = None
    acc_flat2d = None
    acc_flat1d = None

    count_batches = 0
    mean = np.array(data_cfg.normalization_mean, dtype=np.float32)[None, None, None, :]
    std = np.array(data_cfg.normalization_std, dtype=np.float32)[None, None, None, :]

    for i, batch in enumerate(tqdm(stream, desc="val")):
        if cfg.max_batches is not None and i >= cfg.max_batches:
            break

        imgs = batch["image"]
        bsz, _, _, channels = imgs.shape

        # ---------------------------------------------------------
        # 1. RGB Images (2D)
        # ---------------------------------------------------------
        rgb = np.asarray(jax.device_get(imgs * std + mean))
        rgb_prof = _compute_radial_profile(rgb)

        if acc_rgb is None:
            acc_rgb = np.zeros_like(rgb_prof)
        acc_rgb += rgb_prof

        # ---------------------------------------------------------
        # 2. DINO Features (2D spatial grid)
        # ---------------------------------------------------------
        resized = jax.image.resize(imgs, (bsz, 224, 224, channels), method="bicubic")
        feats = restored.dino(resized)
        patch_tokens = feats[:, 5:]  # drop cls + registers

        num_tokens = patch_tokens.shape[1]
        grid_dim = int(math.sqrt(num_tokens))

        # Reshape to (B, H, W, C)
        patch_grid = jnp.reshape(patch_tokens, (bsz, grid_dim, grid_dim, patch_tokens.shape[-1]))
        patch_grid_np = np.asarray(jax.device_get(patch_grid))

        dino_prof = _compute_radial_profile(patch_grid_np)

        if acc_dino is None:
            acc_dino = np.zeros_like(dino_prof)
        acc_dino += dino_prof

        # ---------------------------------------------------------
        # 3. FlatDINO Latents (1D structure)
        # ---------------------------------------------------------
        # Get latents (mu)
        enc_out = restored.encoder(patch_tokens, deterministic=True)
        mu_latent = extract_mu(enc_out, restored.num_flat_tokens, restored.encoder_disposable_registers)
        flat_latents = np.asarray(jax.device_get(mu_latent))  # (B, T, D)

        flat1d_prof = _compute_1d_profile(flat_latents)

        if acc_flat1d is None:
            acc_flat1d = np.zeros_like(flat1d_prof)
        acc_flat1d += flat1d_prof

        # ---------------------------------------------------------
        # 4. Decoded Patches (2D structure check)
        # ---------------------------------------------------------
        decoded_tokens = restored.decoder(mu_latent, deterministic=True)[:, :num_tokens]
        decoded_grid = jnp.reshape(
            decoded_tokens, (bsz, grid_dim, grid_dim, decoded_tokens.shape[-1])
        )
        decoded_grid_np = np.asarray(jax.device_get(decoded_grid))

        flat2d_prof = _compute_radial_profile(decoded_grid_np)

        if acc_flat2d is None:
            acc_flat2d = np.zeros_like(flat2d_prof)
        acc_flat2d += flat2d_prof

        count_batches += 1
        if count_batches % cfg.log_every == 0:
            logging.info(f"Processed {count_batches} batches")

    if count_batches == 0:
        raise RuntimeError("No batches processed.")

    # Normalize accumulated sums to averages
    avg_rgb = acc_rgb / count_batches
    avg_dino = acc_dino / count_batches
    avg_flat2d = acc_flat2d / count_batches
    avg_flat1d = acc_flat1d / count_batches

    # Normalize amplitude to [0, 1] for comparison
    norm_rgb = _normalize_curve(avg_rgb)
    norm_dino = _normalize_curve(avg_dino)
    norm_flat2d = _normalize_curve(avg_flat2d)
    norm_flat1d = _normalize_curve(avg_flat1d)

    # Generate x-axes (Normalized Frequency 0.0 -> 1.0)
    x_rgb = np.linspace(0, 1, len(norm_rgb))
    x_dino = np.linspace(0, 1, len(norm_dino))
    x_flat2d = np.linspace(0, 1, len(norm_flat2d))
    x_flat1d = np.linspace(0, 1, len(norm_flat1d))

    # Plotting
    out_dir = cfg.output_dir or (cfg.flatdino_path / "diffusability")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # 2D baselines
    plt.plot(x_rgb, norm_rgb, label="RGB (Radial Profile)", color="red", alpha=0.7)
    plt.plot(x_dino, norm_dino, label="DINO Feats (Radial Profile)", color="green", alpha=0.7)

    # Your model's outputs
    plt.plot(x_flat2d, norm_flat2d, label="Decoded (Radial Profile)", color="blue", linestyle="--")
    plt.plot(x_flat1d, norm_flat1d, label="Latent (1D FFT)", color="black", linewidth=2.5)

    plt.xlabel("Normalized Frequency (0 = DC, 1 = Nyquist)")
    plt.ylabel("Normalized Amplitude")
    plt.title("Spectral Comparison: 2D Radial vs 1D Linear")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = out_dir / "spectral_analysis.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Saved spectral analysis to {png_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
