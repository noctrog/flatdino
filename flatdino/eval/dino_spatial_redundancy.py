"""Analyze spatial redundancy in DINO features.

This script computes pairwise cosine similarity between all DINO patches in images,
then plots similarity as a function of spatial distance (in patch units). If spatial
redundancy exists, nearby patches should be significantly more similar than distant ones.

Usage:
    python -m flatdino.eval.dino_spatial_redundancy
    python -m flatdino.eval.dino_spatial_redundancy --max-samples 1000
"""

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm

from flatdino.data import DataConfig, create_dataloaders
from flatdino.pretrained import DinoWithRegisters
from flatdino.decoder.augmentations import DINOValAugmentations

# Configure matplotlib for publication-quality figures
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["STIXGeneral"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.labelsize"] = 22
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 16
mpl.rcParams["figure.dpi"] = 150

# Color palette
CONTRAST_PALETTE = [
    "#3B9AB2",  # Teal
    "#F21A00",  # Red
    "#00A08A",  # Green
    "#F98400",  # Orange
    "#5BBCD6",  # Light blue
]


@dataclass
class Config:
    data: DataConfig = field(default_factory=lambda: DataConfig())
    dino_name: str = "facebook/dinov2-with-registers-base"
    """DINO model to use for feature extraction."""
    batch_size: int = 64
    """Batch size for processing images."""
    max_samples: int | None = 5000
    """Maximum number of images to process. None for full validation set."""
    max_distance: float = 10.0
    """Maximum spatial distance to show in plot (in patch units)."""
    output_dir: Path | None = None
    """Directory to save plot. Defaults to current directory."""
    seed: int = 0


def compute_spatial_distances(grid_h: int, grid_w: int) -> np.ndarray:
    """Compute pairwise Euclidean distances between all patch positions.

    Args:
        grid_h: Number of patches in height
        grid_w: Number of patches in width

    Returns:
        (num_patches, num_patches) array of distances in patch units
    """
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[:grid_h, :grid_w]
    positions = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)  # (N, 2)

    # Compute pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    distances = np.sqrt((diff**2).sum(axis=-1))  # (N, N)

    return distances


def compute_cosine_similarity_matrix(features: jax.Array) -> jax.Array:
    """Compute pairwise cosine similarity between all patches.

    Args:
        features: (B, N, D) array of patch features

    Returns:
        (B, N, N) array of cosine similarities
    """
    # Normalize features
    norms = jnp.linalg.norm(features, axis=-1, keepdims=True)
    normalized = features / (norms + 1e-8)

    # Compute pairwise cosine similarity: (B, N, D) @ (B, D, N) -> (B, N, N)
    similarity = jnp.einsum("bnd,bmd->bnm", normalized, normalized)

    return similarity


def aggregate_similarity_by_distance(
    similarities: np.ndarray,
    distances: np.ndarray,
    num_bins: int = 50,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate cosine similarities by spatial distance.

    Args:
        similarities: (B, N, N) array of cosine similarities
        distances: (N, N) array of spatial distances
        num_bins: Number of distance bins
        max_distance: Maximum distance to include. None for all distances.

    Returns:
        bin_centers: array of distance bin centers (starting from 0)
        mean_similarities: array of mean similarities per bin
        std_similarities: array of std similarities per bin
        counts: array of sample counts per bin
    """
    # Flatten similarities across batch and pairs
    # Exclude diagonal (self-similarity = 1) for non-zero distances
    diag_mask = ~np.eye(distances.shape[0], dtype=bool)
    flat_distances = np.tile(distances[diag_mask], similarities.shape[0])
    flat_similarities = similarities[:, diag_mask].flatten()

    # Determine max distance for binning
    if max_distance is not None:
        actual_max = min(max_distance, distances[diag_mask].max())
    else:
        actual_max = distances[diag_mask].max()

    # Create distance bins from 0 to max_distance
    # First bin centered at 0 will have self-similarity (1.0)
    bin_edges = np.linspace(0, actual_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute mean and std per bin
    mean_similarities = np.full(num_bins, np.nan)
    std_similarities = np.full(num_bins, np.nan)
    counts = np.zeros(num_bins)

    for i in range(num_bins):
        bin_mask = (flat_distances >= bin_edges[i]) & (flat_distances < bin_edges[i + 1])
        if bin_mask.sum() > 0:
            mean_similarities[i] = flat_similarities[bin_mask].mean()
            std_similarities[i] = flat_similarities[bin_mask].std()
            counts[i] = bin_mask.sum()

    # Add distance 0 point (self-similarity = 1.0, std = 0)
    # Prepend to arrays
    bin_centers = np.concatenate([[0.0], bin_centers])
    mean_similarities = np.concatenate([[1.0], mean_similarities])
    std_similarities = np.concatenate([[0.0], std_similarities])
    counts = np.concatenate([[similarities.shape[0] * distances.shape[0]], counts])  # B * N diagonal elements

    # Interpolate over any empty bins
    valid_mask = ~np.isnan(mean_similarities)
    if not valid_mask.all() and valid_mask.any():
        mean_similarities = np.interp(
            bin_centers, bin_centers[valid_mask], mean_similarities[valid_mask]
        )
        std_similarities = np.interp(
            bin_centers, bin_centers[valid_mask], std_similarities[valid_mask]
        )

    return bin_centers, mean_similarities, std_similarities, counts


def plot_similarity_vs_distance(
    bin_centers: np.ndarray,
    mean_similarities: np.ndarray,
    std_similarities: np.ndarray,
    output_path: Path,
    dino_name: str,
    num_images: int,
):
    """Plot cosine similarity as a function of spatial distance."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    color = CONTRAST_PALETTE[0]

    # Plot mean with std shading
    ax.plot(
        bin_centers,
        mean_similarities,
        color=color,
        linewidth=2,
        label="Mean similarity",
        zorder=3,
    )
    ax.fill_between(
        bin_centers,
        mean_similarities - std_similarities,
        mean_similarities + std_similarities,
        alpha=0.3,
        color=color,
        label="$\\pm$ 1 std",
        zorder=2,
    )

    ax.set_xlabel("Spatial Distance (patch units)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0.2, 1.0)

    # Legend
    ax.legend(loc="upper right", framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save to both PDF and PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path.with_suffix('.pdf')}")
    print(f"Saved plot to {output_path.with_suffix('.png')}")


def main(cfg: Config):
    print("=" * 60)
    print("Spatial Redundancy Analysis: DINO Patch Similarity vs Distance")
    print("=" * 60)

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)

    # Load DINO model
    print(f"\nLoading DINO model: {cfg.dino_name}")
    dino = DinoWithRegisters(cfg.dino_name, resolution=224, dtype=jnp.float32)
    patch_size = dino.config.patch_size
    hidden_size = dino.config.hidden_size
    num_registers = dino.config.num_register_tokens

    # Compute grid dimensions (224 / 14 = 16 for DINOv2)
    grid_size = 224 // patch_size
    num_patches = grid_size * grid_size

    print(f"  Patch size: {patch_size}")
    print(f"  Grid size: {grid_size}x{grid_size} = {num_patches} patches")
    print(f"  Hidden size: {hidden_size}")

    # Precompute spatial distances
    distances = compute_spatial_distances(grid_size, grid_size)
    print(f"\nSpatial distance range: {distances.min():.2f} to {distances.max():.2f} patch units")

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

    @jax.jit
    def get_dino_features(images: jax.Array) -> jax.Array:
        b = images.shape[0]
        resized = jax.image.resize(images, (b, 224, 224, 3), method="bilinear")
        # Get DINO features, skip CLS and registers
        features = dino(resized)[:, 1 + num_registers :]
        return features

    @jax.jit
    def compute_batch_similarities(features: jax.Array) -> jax.Array:
        return compute_cosine_similarity_matrix(features)

    # Collect similarity statistics
    all_similarities = []
    num_samples = 0

    print("\nComputing pairwise similarities...")
    for batch in tqdm(val_loader, desc="Processing"):
        images = jnp.array(batch["image"])
        b = images.shape[0]

        # Get DINO features: (B, num_patches, hidden_size)
        features = get_dino_features(images)

        # Compute pairwise similarities: (B, num_patches, num_patches)
        similarities = compute_batch_similarities(features)
        all_similarities.append(np.array(similarities))

        num_samples += b
        if cfg.max_samples is not None and num_samples >= cfg.max_samples:
            break

    # Concatenate all similarities
    all_similarities = np.concatenate(all_similarities, axis=0)
    print(f"\nProcessed {num_samples} images")
    print(f"Similarity matrix shape: {all_similarities.shape}")

    # Aggregate by distance
    print(f"\nAggregating similarities by spatial distance (max_distance={cfg.max_distance})...")
    bin_centers, mean_sims, std_sims, counts = aggregate_similarity_by_distance(
        all_similarities, distances, num_bins=50, max_distance=cfg.max_distance
    )

    # Print statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Distance (patches)':<20} {'Mean Sim':>12} {'Std':>12}")
    print("-" * 60)

    # Print key distance points (filter to max_distance)
    key_distances = [0.0, 1.0, 2.0, 5.0, 10.0]
    key_distances = [d for d in key_distances if d <= cfg.max_distance]
    for d in key_distances:
        idx = np.argmin(np.abs(bin_centers - d))
        if counts[idx] > 0:
            print(f"{bin_centers[idx]:<20.1f} {mean_sims[idx]:>12.4f} {std_sims[idx]:>12.4f}")

    # Compute correlation decay statistics
    self_sim = mean_sims[0]  # Distance 0 = 1.0
    nearest_sim = mean_sims[np.argmin(np.abs(bin_centers - 1.0))]
    farthest_idx = np.argmin(np.abs(bin_centers - cfg.max_distance))
    farthest_sim = mean_sims[farthest_idx]
    farthest_dist = bin_centers[farthest_idx]

    # Half-decay: distance where similarity drops to halfway between self (1.0) and farthest
    target_sim = (self_sim + farthest_sim) / 2
    half_decay_idx = np.argmin(np.abs(mean_sims - target_sim))
    half_decay_dist = bin_centers[half_decay_idx]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Similarity at distance 0 (self):       {self_sim:.4f}")
    print(f"Similarity at distance ~1 (adjacent):  {nearest_sim:.4f}")
    print(f"Similarity at distance ~{farthest_dist:.0f}:            {farthest_sim:.4f}")
    print(f"Total similarity drop (0 to {farthest_dist:.0f}):       {self_sim - farthest_sim:.4f}")
    print(f"Half-decay distance:                   {half_decay_dist:.1f} patches")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    drop = self_sim - farthest_sim
    if drop > 0.3:
        print("Strong spatial redundancy detected!")
        print(f"  - Self-similarity is {self_sim:.1%}")
        print(f"  - Adjacent patches are {nearest_sim:.1%} similar")
        print(f"  - Distant patches ({farthest_dist:.0f} units) are {farthest_sim:.1%} similar")
        print("  - This suggests significant compressibility via spatial aggregation")
    else:
        print("Moderate spatial redundancy detected.")
        print(f"  - Similarity drops by {drop:.1%} from self to {farthest_dist:.0f} patches")
        print("  - DINO features maintain high global coherence across the image")

    # Save plot
    output_dir = cfg.output_dir or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = cfg.dino_name.split("/")[-1]
    plot_path = output_dir / f"spatial_redundancy_{model_short}"
    plot_similarity_vs_distance(
        bin_centers, mean_sims, std_sims, plot_path, cfg.dino_name, num_samples
    )


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
