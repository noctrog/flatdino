"""Compute intrinsic dimensionality of DINO features vs raw image patches.

This script computes the number of PCA components needed to explain 95% and 99%
of variance for both DINO patch features and raw image patches, to understand
the compressibility of each representation.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from tqdm import tqdm
from einops import rearrange
from sklearn.decomposition import PCA

from flatdino.data import DataConfig, create_dataloaders
from flatdino.pretrained import DinoWithRegisters
from flatdino.decoder.augmentations import DINOValAugmentations


@dataclass
class Config:
    data: DataConfig = field(default_factory=lambda: DataConfig())
    dino_name: str = "facebook/dinov2-with-registers-base"
    """DINO model to use for feature extraction."""
    batch_size: int = 64
    """Batch size for processing images."""
    max_samples: int | None = 10_000
    """Maximum number of images to process. None for full validation set."""
    seed: int = 0


def extract_image_patches(images: jax.Array, patch_size: int = 14) -> jax.Array:
    """Extract non-overlapping patches from images.

    Args:
        images: (B, H, W, C) images
        patch_size: Size of each patch (default 14 to match DINO)

    Returns:
        (B, num_patches, patch_size * patch_size * C) flattened patches
    """
    return rearrange(
        images,
        "b (h ph) (w pw) c -> b (h w) (ph pw c)",
        ph=patch_size,
        pw=patch_size,
    )


def compute_intrinsic_dim(features: np.ndarray, name: str) -> dict:
    """Compute PCA and find dimensions for 95% and 99% variance.

    Args:
        features: (N, D) array of features
        name: Name for logging

    Returns:
        Dictionary with dim_95, dim_99, and full explained variance ratios
    """
    print(f"\nComputing PCA for {name}...")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean():.4f}, Std: {features.std():.4f}")

    # Fit PCA (use min of samples and features for n_components)
    n_components = min(features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(features)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    dim_99 = int(np.searchsorted(cumvar, 0.99) + 1)

    print(f"  Dimensions for 95% variance: {dim_95}")
    print(f"  Dimensions for 99% variance: {dim_99}")
    print(f"  Total dimensions: {features.shape[1]}")
    print(f"  Compression ratio at 95%: {features.shape[1] / dim_95:.1f}x")
    print(f"  Compression ratio at 99%: {features.shape[1] / dim_99:.1f}x")

    return {
        "dim_95": dim_95,
        "dim_99": dim_99,
        "cumvar": cumvar,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def main(cfg: Config):
    print("=" * 60)
    print("Intrinsic Dimensionality Analysis: DINO vs Image Patches")
    print("=" * 60)

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)

    # Load DINO model
    print(f"\nLoading DINO model: {cfg.dino_name}")
    dino = DinoWithRegisters(cfg.dino_name, resolution=224, dtype=jnp.float32)
    patch_size = dino.config.patch_size  # Should be 14 for DINOv2
    hidden_size = dino.config.hidden_size  # 768 for ViT-B
    num_registers = dino.config.num_register_tokens  # 4 for DINOv2 with registers
    print(f"  Patch size: {patch_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num registers: {num_registers}")

    # Create validation dataloader
    print("\nLoading ImageNet validation set...")
    cfg.data.num_workers = 4
    data = create_dataloaders(
        cfg.data,
        batch_size=cfg.batch_size,
        val_epochs=1,
        val_aug=DINOValAugmentations(cfg.data),
        drop_remainder_val=True,
    )
    val_loader = data.val_loader

    # Collect features
    all_dino_feats = []
    all_image_patches = []
    num_samples = 0

    @jax.jit
    def get_dino_features(images: jax.Array) -> jax.Array:
        # Resize to 224x224 for DINO
        b = images.shape[0]
        resized = jax.image.resize(images, (b, 224, 224, 3), method="bilinear")
        # Get DINO features, skip CLS (1) and registers (4) -> [:, 5:]
        features = dino(resized)[:, 1 + num_registers :]
        return features

    print("\nCollecting features from validation set...")
    for batch in tqdm(val_loader, desc="Processing"):
        images = jnp.array(batch["image"])
        b = images.shape[0]

        # Get DINO features
        dino_feats = get_dino_features(images)
        # Reshape to (B * num_patches, hidden_size)
        dino_feats = dino_feats.reshape(-1, hidden_size)
        all_dino_feats.append(np.array(dino_feats))

        # Get raw image patches (resize to 224x224 first to match DINO input)
        resized = jax.image.resize(images, (b, 224, 224, 3), method="bilinear")
        patches = extract_image_patches(resized, patch_size)
        # Reshape to (B * num_patches, patch_dim)
        patch_dim = patch_size * patch_size * 3
        patches = patches.reshape(-1, patch_dim)
        all_image_patches.append(np.array(patches))

        num_samples += b
        if cfg.max_samples is not None and num_samples >= cfg.max_samples:
            break

    # Concatenate all features
    dino_feats = np.concatenate(all_dino_feats, axis=0)
    image_patches = np.concatenate(all_image_patches, axis=0)

    print(f"\nTotal patches collected: {dino_feats.shape[0]}")
    print(f"From {num_samples} images")

    # Compute intrinsic dimensionality for DINO features
    dino_results = compute_intrinsic_dim(dino_feats, "DINO features")

    # Compute intrinsic dimensionality for image patches
    patch_results = compute_intrinsic_dim(image_patches, "Image patches")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'DINO':>12} {'Pixels':>12}")
    print("-" * 60)
    print(f"{'Feature dimension':<30} {hidden_size:>12} {patch_size**2 * 3:>12}")
    print(f"{'Dims for 95% variance':<30} {dino_results['dim_95']:>12} {patch_results['dim_95']:>12}")
    print(f"{'Dims for 99% variance':<30} {dino_results['dim_99']:>12} {patch_results['dim_99']:>12}")
    print(f"{'Compression @ 95%':<30} {hidden_size / dino_results['dim_95']:>11.1f}x {patch_size**2 * 3 / patch_results['dim_95']:>11.1f}x")
    print(f"{'Compression @ 99%':<30} {hidden_size / dino_results['dim_99']:>11.1f}x {patch_size**2 * 3 / patch_results['dim_99']:>11.1f}x")

    # Implications for VAE compression
    print("\n" + "=" * 60)
    print("IMPLICATIONS FOR VAE COMPRESSION")
    print("=" * 60)
    total_dino_dim = 256 * hidden_size  # 256 patches * 768 dims
    total_pixel_dim = 256 * patch_size**2 * 3  # 256 patches * 588 dims

    print("\nTotal representation size:")
    print(f"  DINO: 256 patches x {hidden_size} dims = {total_dino_dim:,} values")
    print(f"  Pixels: 256 patches x {patch_size**2 * 3} dims = {total_pixel_dim:,} values")

    # Estimate minimal latent size based on PCA
    # For 32 tokens, what dimension do we need?
    print("\nFor 32 latent tokens:")
    for target_var in [0.95, 0.99]:
        dino_dim = dino_results['dim_95'] if target_var == 0.95 else dino_results['dim_99']
        pixel_dim = patch_results['dim_95'] if target_var == 0.95 else patch_results['dim_99']

        # Total information to preserve (per image, across all patches)
        # This is a rough estimate assuming patches are independent
        dino_total_info = 256 * dino_dim
        pixel_total_info = 256 * pixel_dim

        # If we have 32 tokens, what dim per token?
        dino_per_token = dino_total_info / 32
        pixel_per_token = pixel_total_info / 32

        print(f"\n  At {int(target_var * 100)}% variance:")
        print(f"    DINO:   ~{dino_per_token:.0f} dims/token needed (32 x {dino_per_token:.0f} = {32 * dino_per_token:.0f})")
        print(f"    Pixels: ~{pixel_per_token:.0f} dims/token needed (32 x {pixel_per_token:.0f} = {32 * pixel_per_token:.0f})")


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
