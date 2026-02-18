"""Estimate Mutual Information (MI) between FlatDINO latent tokens.

Computes pairwise MI between all tokens to understand token dependencies:
    MI(X, Y) = H(X) + H(Y) - H(X, Y)

Under Gaussian assumption:
    MI(X, Y) = 0.5 * (log|Σ_X| + log|Σ_Y| - log|Σ_{X,Y}|)

Lower off-diagonal MI means more independent tokens = better compression.

Example usage:
    python -m flatdino.eval.mutual_information --checkpoint output/flatdino/vae/med-32-bl-64

    # With k-NN estimator (more accurate for non-Gaussian)
    python -m flatdino.eval.mutual_information --checkpoint ... --method knn
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
import matplotlib.pyplot as plt
import numpy as np
import tyro
from absl import logging
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from flatdino.data import DataLoaders, create_dataloaders
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import restore_encoder, save_eval_results
from flatdino.distributed import prefetch_to_mesh


@dataclass
class Args:
    checkpoint: Path
    """Path to FlatDINO checkpoint."""
    method: Literal["gaussian", "knn"] = "gaussian"
    """Estimation method: 'gaussian' (fast) or 'knn' (more accurate for non-Gaussian)."""
    num_images: int = 50000
    """Number of images to encode (max 50k for ImageNet val)."""
    batch_size: int = 256
    """Batch size for encoding."""
    k_neighbors: int = 5
    """Number of neighbors for k-NN MI estimator."""
    seed: int = 42


def estimate_mi_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate MI between two multivariate variables using Gaussian approximation.

    MI(X, Y) = 0.5 * (log|Σ_X| + log|Σ_Y| - log|Σ_{X,Y}|)

    Args:
        x: (N, D_x) samples from X
        y: (N, D_y) samples from Y

    Returns:
        Estimated MI in nats
    """
    len(x)
    d_x, d_y = x.shape[1], y.shape[1]

    # Compute covariances with regularization
    eps = 1e-6
    cov_x = np.cov(x.T) + eps * np.eye(d_x)
    cov_y = np.cov(y.T) + eps * np.eye(d_y)

    # Joint covariance
    xy = np.concatenate([x, y], axis=1)
    cov_xy = np.cov(xy.T) + eps * np.eye(d_x + d_y)

    # Log determinants
    sign_x, logdet_x = np.linalg.slogdet(cov_x)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    sign_xy, logdet_xy = np.linalg.slogdet(cov_xy)

    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        logging.warning("Covariance matrix is not positive definite")
        return float("nan")

    # MI = 0.5 * (log|Σ_X| + log|Σ_Y| - log|Σ_{X,Y}|)
    mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return max(0.0, float(mi))  # MI is non-negative


def estimate_mi_knn(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Estimate MI using k-NN based estimator (KSG estimator).

    Based on Kraskov et al. "Estimating Mutual Information" (2004).

    Args:
        x: (N, D_x) samples from X
        y: (N, D_y) samples from Y
        k: Number of nearest neighbors

    Returns:
        Estimated MI in nats
    """
    n = len(x)

    # Joint space
    xy = np.concatenate([x, y], axis=1)

    # Find k-th neighbor distance in joint space (using Chebyshev/max norm)
    nn_xy = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn_xy.fit(xy)
    distances_xy, _ = nn_xy.kneighbors(xy)
    eps_xy = distances_xy[:, k]  # k-th neighbor distance (0-indexed, so k+1 neighbors)

    # Count neighbors within eps in marginal spaces
    nn_x = NearestNeighbors(metric="chebyshev")
    nn_x.fit(x)
    nn_y = NearestNeighbors(metric="chebyshev")
    nn_y.fit(y)

    # Count points within eps distance (excluding self)
    n_x = np.zeros(n)
    n_y = np.zeros(n)

    for i in range(n):
        # Points within eps_xy[i] in x space
        indices_x = nn_x.radius_neighbors([x[i]], radius=eps_xy[i], return_distance=False)[0]
        n_x[i] = len(indices_x) - 1  # Exclude self

        # Points within eps_xy[i] in y space
        indices_y = nn_y.radius_neighbors([y[i]], radius=eps_xy[i], return_distance=False)[0]
        n_y[i] = len(indices_y) - 1  # Exclude self

    # KSG estimator: MI = psi(k) - <psi(n_x + 1) + psi(n_y + 1)> + psi(n)
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return max(0.0, float(mi))  # MI is non-negative


def compute_pairwise_mi(
    latents: np.ndarray,
    num_tokens: int,
    method: str = "gaussian",
    k_neighbors: int = 5,
) -> np.ndarray:
    """Compute pairwise MI between all tokens.

    Args:
        latents: (N, T*F) flattened latent codes
        num_tokens: Number of tokens T
        method: 'gaussian' or 'knn'
        k_neighbors: k for k-NN estimator

    Returns:
        (T, T) matrix of pairwise MI values
    """
    n, d = latents.shape
    features_per_token = d // num_tokens

    # Reshape to (N, T, F)
    tokens = latents.reshape(n, num_tokens, features_per_token)

    # Compute MI matrix
    mi_matrix = np.zeros((num_tokens, num_tokens))

    estimate_fn = estimate_mi_gaussian if method == "gaussian" else lambda x, y: estimate_mi_knn(x, y, k_neighbors)

    total_pairs = num_tokens * (num_tokens + 1) // 2
    with tqdm(total=total_pairs, desc=f"Computing MI ({method})") as pbar:
        for i in range(num_tokens):
            for j in range(i, num_tokens):
                token_i = tokens[:, i, :]  # (N, F)
                token_j = tokens[:, j, :]  # (N, F)

                if i == j:
                    # Self-information = entropy, but we'll compute MI(X,X) = H(X)
                    # For visualization purposes, compute entropy using Gaussian approx
                    eps = 1e-6
                    cov = np.cov(token_i.T) + eps * np.eye(features_per_token)
                    _, logdet = np.linalg.slogdet(cov)
                    # H(X) for Gaussian = 0.5 * (d * log(2*pi*e) + log|Σ|)
                    entropy = 0.5 * (features_per_token * np.log(2 * np.pi * np.e) + logdet)
                    mi_matrix[i, j] = entropy
                else:
                    mi = estimate_fn(token_i, token_j)
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi  # Symmetric

                pbar.update(1)

    return mi_matrix


def plot_mi_heatmap(
    mi_matrix: np.ndarray,
    token_entropies: np.ndarray,
    output_dir: Path,
    title: str = "Pairwise Mutual Information Between Tokens",
) -> tuple[Path, Path]:
    """Plot MI matrix as heatmap (with masked diagonal) and entropy bar chart.

    Args:
        mi_matrix: (T, T) MI matrix (off-diagonal values, diagonal will be masked)
        token_entropies: (T,) entropy of each token
        output_dir: Directory to save plots
        title: Plot title

    Returns:
        Paths to PDF and PNG files
    """
    num_tokens = mi_matrix.shape[0]

    # Configure matplotlib for publication-quality plots
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    })

    # Create figure with two subplots
    fig, (ax_heatmap, ax_entropy) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # Mask diagonal for heatmap
    mi_masked = mi_matrix.copy()
    np.fill_diagonal(mi_masked, np.nan)

    # Plot heatmap with masked diagonal
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")  # Color for NaN (diagonal)
    im = ax_heatmap.imshow(mi_masked, cmap=cmap, aspect="equal")

    # Add colorbar
    plt.colorbar(im, ax=ax_heatmap, label="MI (nats)")

    # Set ticks
    ax_heatmap.set_xticks(range(num_tokens))
    ax_heatmap.set_yticks(range(num_tokens))
    ax_heatmap.set_xticklabels([f"{i}" for i in range(num_tokens)])
    ax_heatmap.set_yticklabels([f"{i}" for i in range(num_tokens)])

    # Labels
    ax_heatmap.set_xlabel("Token Index")
    ax_heatmap.set_ylabel("Token Index")
    ax_heatmap.set_title("Pairwise MI (diagonal masked)")

    # Add text annotations for small matrices
    if num_tokens <= 8:
        vmax = np.nanmax(mi_masked)
        for i in range(num_tokens):
            for j in range(num_tokens):
                if i == j:
                    continue  # Skip diagonal
                ax_heatmap.text(
                    j, i, f"{mi_matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if mi_matrix[i, j] < vmax * 0.7 else "black",
                    fontsize=8,
                )

    # Plot entropy bar chart
    x_pos = np.arange(num_tokens)
    bars = ax_entropy.bar(x_pos, token_entropies, color="steelblue", edgecolor="black", linewidth=0.5)
    ax_entropy.set_xlabel("Token Index")
    ax_entropy.set_ylabel("Entropy H(X) (nats)")
    ax_entropy.set_title("Token Entropies")
    ax_entropy.set_xticks(x_pos)
    ax_entropy.set_xticklabels([f"{i}" for i in range(num_tokens)])

    # Add value labels on bars for small number of tokens
    if num_tokens <= 16:
        for bar, val in zip(bars, token_entropies):
            ax_entropy.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=8,
            )

    # Add mean line
    mean_entropy = np.mean(token_entropies)
    ax_entropy.axhline(mean_entropy, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_entropy:.2f}")
    ax_entropy.legend(loc="upper right")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    # Save as PDF and PNG
    pdf_path = output_dir / "mutual_information.pdf"
    png_path = output_dir / "mutual_information.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=150)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    return pdf_path, png_path


def main(args: Args):
    logging.set_verbosity(logging.INFO)
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    # Load encoder
    logging.info(f"Loading checkpoint from {args.checkpoint}")
    components = restore_encoder(args.checkpoint, mesh=mesh, mp=mp, encoder=True, decoder=False)
    encoder = components.encoder
    dino = components.dino
    num_tokens = components.num_flat_tokens

    # Create validation dataloader
    logging.info("Creating validation dataloader")
    data: DataLoaders = create_dataloaders(
        components.data_cfg,
        args.batch_size,
        val_aug=FlatDinoValAugmentations(components.aug_cfg, components.data_cfg),
        val_epochs=1,
        drop_remainder_val=False,
    )

    num_images = min(args.num_images, data.val_ds_size)
    logging.info(f"Encoding {num_images} images...")

    # Encode all images
    all_latents = []
    num_encoded = 0

    @nnx.jit
    def encode_batch(imgs: jax.Array) -> jax.Array:
        b, h, w, c = imgs.shape
        imgs = jax.image.resize(imgs, (b, 224, 224, c), method="bilinear")
        dino_patches = dino(imgs)[:, 5:]  # Remove CLS and registers
        encoded = encoder(dino_patches)
        # Extract mu (first half of output dim)
        mu = encoded[:, :, :encoded.shape[-1] // 2]
        return mu

    val_iter = iter(data.val_loader)
    for batch in tqdm(
        prefetch_to_mesh(val_iter, 1, mesh),
        total=(num_images + args.batch_size - 1) // args.batch_size,
        desc="Encoding",
    ):
        if num_encoded >= num_images:
            break

        imgs = batch["image"]
        mu = encode_batch(imgs)  # (B, T, F)

        # Flatten tokens: (B, T, F) -> (B, T*F)
        flat = mu.reshape(mu.shape[0], -1)
        all_latents.append(np.array(flat))
        num_encoded += len(imgs)

    latents = np.concatenate(all_latents, axis=0)[:num_images]
    logging.info(f"Encoded {len(latents)} images, latent shape: {latents.shape}")

    n, d = latents.shape
    features_per_token = d // num_tokens
    logging.info(f"Tokens: {num_tokens}, features per token: {features_per_token}")

    # Compute pairwise MI
    logging.info(f"Computing pairwise MI using {args.method} method...")
    mi_matrix = compute_pairwise_mi(
        latents,
        num_tokens,
        method=args.method,
        k_neighbors=args.k_neighbors,
    )

    # Extract diagonal (entropies) separately
    token_entropies = np.diag(mi_matrix).copy()

    # Compute summary statistics (off-diagonal only)
    mask = ~np.eye(num_tokens, dtype=bool)
    off_diag_mi = mi_matrix[mask]
    mean_mi = float(np.mean(off_diag_mi))
    max_mi = float(np.max(off_diag_mi))
    min_mi = float(np.min(off_diag_mi))
    mean_entropy = float(np.mean(token_entropies))
    min_entropy = float(np.min(token_entropies))
    max_entropy = float(np.max(token_entropies))

    # Log results
    logging.info(f"Mean off-diagonal MI: {mean_mi:.4f} nats")
    logging.info(f"Max off-diagonal MI: {max_mi:.4f} nats")
    logging.info(f"Min off-diagonal MI: {min_mi:.4f} nats")
    logging.info(f"Mean token entropy: {mean_entropy:.4f} nats")

    # Plot heatmap (with masked diagonal) and entropy bar chart
    logging.info("Generating plots...")
    pdf_path, png_path = plot_mi_heatmap(
        mi_matrix,
        token_entropies,
        args.checkpoint,
        title=f"Pairwise MI ({args.method}) - {args.checkpoint.name}",
    )
    logging.info(f"Saved plots to {pdf_path} and {png_path}")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Method: {args.method}")
    print(f"Latent shape: {num_tokens} tokens x {features_per_token} features = {d} dims")
    print("-" * 50)
    print("Mutual Information (off-diagonal):")
    print(f"  Mean: {mean_mi:.4f} nats")
    print(f"  Max:  {max_mi:.4f} nats")
    print(f"  Min:  {min_mi:.4f} nats")
    print("-" * 50)
    print("Token Entropies H(X):")
    print(f"  Mean: {mean_entropy:.4f} nats")
    print(f"  Max:  {max_entropy:.4f} nats")
    print(f"  Min:  {min_entropy:.4f} nats")
    print("=" * 50)

    # Print token entropies
    print("\nToken Entropies (nats):")
    print("-" * 50)
    for i in range(num_tokens):
        print(f"  Token {i:2d}: {token_entropies[i]:.4f}")

    # Print MI matrix (off-diagonal only)
    print("\nMI Matrix (nats, diagonal = entropy shown separately):")
    print("-" * 50)
    header = "     " + "".join(f"  T{i:02d}" for i in range(num_tokens))
    print(header)
    for i in range(num_tokens):
        row_vals = []
        for j in range(num_tokens):
            if i == j:
                row_vals.append("   -- ")  # Mask diagonal in text output
            else:
                row_vals.append(f"{mi_matrix[i, j]:6.2f}")
        row = f"T{i:02d}: " + "".join(row_vals)
        print(row)

    # Save results to JSON
    results = {
        "method": args.method,
        "num_images": len(latents),
        "num_tokens": num_tokens,
        "features_per_token": features_per_token,
        "latent_dim": d,
        "mutual_information": {
            "mean": mean_mi,
            "max": max_mi,
            "min": min_mi,
        },
        "token_entropies": {
            "values": token_entropies.tolist(),
            "mean": mean_entropy,
            "max": max_entropy,
            "min": min_entropy,
        },
        "mi_matrix": mi_matrix.tolist(),
    }
    save_eval_results(args.checkpoint, "mutual_information", results)
    print(f"\nSaved mutual information results to {args.checkpoint}/eval_results.json")
    print(f"Plots saved to:\n  - {pdf_path}\n  - {png_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
