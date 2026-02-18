"""Token ablation analysis for FlatDINO.

Measures what visual information each latent token encodes by:
1. Encoding images to get latent tokens
2. For each token, zero it out and measure reconstruction change
3. Visualize which spatial regions are affected by each token

Optionally computes pixel-level ablation using the RAE decoder (--rae flag).

Example usage:
    python -m flatdino.eval.token_ablation --flatdino-path output/flatdino/vae/med-32-ss-64
    python -m flatdino.eval.token_ablation --flatdino-path output/flatdino/vae/med-32-ss-64 --rae
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tyro
from tqdm import tqdm

# Configure matplotlib for publication-quality figures
# Use STIX fonts (bundled with matplotlib, similar to Times New Roman)
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["STIXGeneral"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

from flatdino.data import create_dataloaders
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import restore_encoder, save_eval_results
from flatdino.distributed import prefetch_to_mesh


@dataclass
class Args:
    flatdino_path: Path
    """Path to FlatDINO checkpoint."""
    num_images: int = 10_000
    """Number of images to use for computing ablation heatmaps."""
    batch_size: int = 128
    """Batch size for encoding."""
    ablation_batch_size: int = 256
    """Batch size for ablation computation (reduce if OOM)."""
    rae: bool = False
    """Enable pixel-level ablation using RAE decoder."""
    seed: int = 42


def compute_ablation_heatmaps(
    decoder: nnx.Module,
    latents: jax.Array,
    num_tokens: int,
    num_patches: int,
    batch_size: int = 64,
    rae_decoder: nnx.Module | None = None,
    image_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute reconstruction error heatmaps for each token ablation.

    Computes patch-level heatmaps, and optionally pixel-level heatmaps if
    rae_decoder is provided. Both are computed in a single pass.

    Args:
        decoder: FlatDINO decoder
        latents: Encoded latent tokens (N, T, F) - mu values
        num_tokens: Number of latent tokens
        num_patches: Number of DINO patches
        batch_size: Batch size for processing (to avoid OOM)
        rae_decoder: Optional RAE decoder (DINO patches -> pixels)
        image_size: Output image size (required if rae_decoder is provided)

    Returns:
        patch_heatmaps: (T, P) array of L2 distance per patch when token is ablated
        pixel_heatmaps: (T, H, W) array of pixel L2 distance, or None if rae_decoder not provided
    """
    if rae_decoder is not None and image_size is None:
        raise ValueError("image_size must be provided when rae_decoder is specified")

    num_samples = latents.shape[0]
    compute_pixels = rae_decoder is not None

    patch_heatmaps = []
    pixel_heatmaps = [] if compute_pixels else None

    desc = "Ablating tokens" + (" (with pixel)" if compute_pixels else "")
    for token_idx in tqdm(range(num_tokens), desc=desc):
        ablation_mask = jnp.ones(num_tokens).at[token_idx].set(0.0)

        # Accumulators for patch-level
        total_patch_delta = jnp.zeros(num_patches)
        # Accumulators for pixel-level (if enabled)
        total_pixel_delta = jnp.zeros((image_size, image_size)) if compute_pixels else None
        total_count = 0

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            latent_batch = latents[start_idx:end_idx]

            # Get original reconstruction in DINO space
            original_recon = decoder(latent_batch, deterministic=True)[:, :num_patches]

            # Apply ablation and reconstruct
            ablated_latents = latent_batch * ablation_mask[None, :, None]
            ablated_recon = decoder(ablated_latents, deterministic=True)[:, :num_patches]

            # Compute patch-level L2 distance
            batch_patch_delta = jnp.linalg.norm(ablated_recon - original_recon, axis=-1)
            total_patch_delta = total_patch_delta + jnp.sum(batch_patch_delta, axis=0)

            # Compute pixel-level L2 distance (if enabled)
            if compute_pixels:
                original_rae_out = rae_decoder(original_recon)
                original_pixels = rae_decoder.unpatchify(
                    original_rae_out.logits, original_image_size=(image_size, image_size)
                )

                ablated_rae_out = rae_decoder(ablated_recon)
                ablated_pixels = rae_decoder.unpatchify(
                    ablated_rae_out.logits, original_image_size=(image_size, image_size)
                )

                batch_pixel_delta = jnp.linalg.norm(ablated_pixels - original_pixels, axis=1)
                total_pixel_delta = total_pixel_delta + jnp.sum(batch_pixel_delta, axis=0)

            total_count += latent_batch.shape[0]

        # Compute means
        patch_heatmaps.append(np.array(total_patch_delta / total_count))
        if compute_pixels:
            pixel_heatmaps.append(np.array(total_pixel_delta / total_count))

    patch_heatmaps = np.stack(patch_heatmaps, axis=0)
    pixel_heatmaps = np.stack(pixel_heatmaps, axis=0) if compute_pixels else None

    return patch_heatmaps, pixel_heatmaps


def plot_pixel_heatmap(
    heatmap: np.ndarray,
    vmin: float,
    vmax: float,
    save_path: Path,
):
    """Plot and save a single token's pixel ablation heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    im = ax.imshow(heatmap, cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Pixel $\Delta$ (L2)")

    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pixel_heatmap_grid(
    heatmaps: np.ndarray,
    vmin: float,
    vmax: float,
    save_path: Path,
):
    """Plot all pixel ablation heatmaps in a grid."""
    num_tokens = heatmaps.shape[0]

    ncols = int(np.ceil(np.sqrt(num_tokens)))
    nrows = int(np.ceil(num_tokens / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2 * ncols, 2 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for idx in range(num_tokens):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        im = ax.imshow(heatmaps[idx], cmap="hot", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(num_tokens, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.colorbar(im, ax=axes, label=r"Pixel $\Delta$ (L2)", shrink=0.8)

    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    heatmap: np.ndarray,
    grid_size: int,
    vmin: float,
    vmax: float,
    save_path: Path,
):
    """Plot and save a single token's ablation heatmap."""
    # Reshape to spatial grid
    heatmap_2d = heatmap.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    im = ax.imshow(heatmap_2d, cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\Delta$ (L2)")

    # Save both formats
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_grid(
    heatmaps: np.ndarray,
    grid_size: int,
    vmin: float,
    vmax: float,
    save_path: Path,
):
    """Plot all token heatmaps in a grid."""
    num_tokens = heatmaps.shape[0]

    # Compute grid dimensions (as square as possible)
    ncols = int(np.ceil(np.sqrt(num_tokens)))
    nrows = int(np.ceil(num_tokens / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2 * ncols, 2 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for idx in range(num_tokens):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        heatmap_2d = heatmaps[idx].reshape(grid_size, grid_size)
        im = ax.imshow(heatmap_2d, cmap="hot", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(num_tokens, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    # Add colorbar
    fig.colorbar(im, ax=axes, label=r"Reconstruction $\Delta$ (L2)", shrink=0.8)

    # Save both formats
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(args: Args):
    # Create output directory
    output_dir = args.flatdino_path / "token_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Check if pre-computed heatmaps exist
    heatmaps_path = output_dir / "heatmaps.npy"
    pixel_heatmaps_path = args.flatdino_path / "token_ablation_pixel" / "pixel_heatmaps.npy"

    use_precomputed = heatmaps_path.exists()
    use_precomputed_pixel = args.rae and pixel_heatmaps_path.exists()

    if use_precomputed:
        logging.warning(
            f"Using pre-computed heatmaps from {heatmaps_path}. Delete this file to recompute."
        )
        heatmaps = np.load(heatmaps_path)
        num_tokens = heatmaps.shape[0]
        num_patches = heatmaps.shape[1]
        grid_size = int(np.sqrt(num_patches))

        # Try to load num_images from existing results
        from flatdino.eval import load_eval_results

        existing_results = load_eval_results(args.flatdino_path)
        num_images = existing_results.get("token_ablation", {}).get("num_images")

        pixel_heatmaps = None
        image_size = None
        if use_precomputed_pixel:
            logging.warning(
                f"Using pre-computed pixel heatmaps from {pixel_heatmaps_path}. "
                "Delete this file to recompute."
            )
            pixel_heatmaps = np.load(pixel_heatmaps_path)
            image_size = pixel_heatmaps.shape[1]
        elif args.rae:
            logging.warning(
                "Pixel heatmaps not found. Run without --rae first, then with --rae to compute pixel heatmaps."
            )
    else:
        mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
        jax.set_mesh(mesh)
        mp = jmp.Policy(
            param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32
        )

        # Load encoder and decoder
        print(f"Loading checkpoint from {args.flatdino_path}")
        components = restore_encoder(
            args.flatdino_path, mesh=mesh, mp=mp, encoder=True, decoder=True
        )
        encoder = components.encoder
        decoder = components.decoder
        dino = components.dino
        num_tokens = components.num_flat_tokens

        assert encoder is not None
        assert decoder is not None

        # Create validation dataloader
        print("Creating validation dataloader")
        data = create_dataloaders(
            components.data_cfg,
            args.batch_size,
            val_aug=FlatDinoValAugmentations(components.aug_cfg, components.data_cfg),
            val_epochs=1,
            drop_remainder_val=False,
        )

        num_images = min(args.num_images, data.val_ds_size)
        print(f"Processing {num_images} images...")

        # Collect latents and DINO patches
        all_latents = []
        all_patches = []
        num_collected = 0

        @nnx.jit
        def encode_batch(imgs: jax.Array) -> tuple[jax.Array, jax.Array]:
            b, h, w, c = imgs.shape
            imgs = jax.image.resize(imgs, (b, 224, 224, c), method="bilinear")
            dino_patches = dino(imgs)[:, 5:]  # Remove CLS and registers
            encoded = encoder(dino_patches)
            # Extract mu (first half of output dim)
            mu = encoded[:, :, : encoded.shape[-1] // 2]
            return mu[:, :num_tokens], dino_patches

        val_iter = iter(data.val_loader)
        for batch in tqdm(
            prefetch_to_mesh(val_iter, 1, mesh, trim=True),
            total=(num_images + args.batch_size - 1) // args.batch_size,
            desc="Encoding",
        ):
            if num_collected >= num_images:
                break

            imgs = batch["image"]
            mu, patches = encode_batch(imgs)

            all_latents.append(np.array(mu))
            all_patches.append(np.array(patches))
            num_collected += len(imgs)

        latents = jnp.array(np.concatenate(all_latents, axis=0)[:num_images])
        dino_patches = jnp.array(np.concatenate(all_patches, axis=0)[:num_images])

        print(f"Latents shape: {latents.shape}")
        print(f"DINO patches shape: {dino_patches.shape}")

        num_patches = dino_patches.shape[1]
        grid_size = int(np.sqrt(num_patches))
        assert grid_size * grid_size == num_patches, (
            f"Patches ({num_patches}) must form a square grid"
        )

        # Create RAE decoder if pixel ablation is enabled
        rae_decoder = None
        image_size = None
        if args.rae:
            image_size = components.aug_cfg.image_size[0]
            rae_decoder = make_rae_decoder(
                num_patches=num_patches,
                image_size=image_size,
                dtype=mp.param_dtype,
                seed=args.seed,
            )

        # Compute ablation heatmaps (and optionally pixel heatmaps in same pass)
        print("Computing ablation heatmaps..." + (" (with pixel ablation)" if args.rae else ""))
        heatmaps, pixel_heatmaps = compute_ablation_heatmaps(
            decoder,
            latents,
            num_tokens,
            num_patches,
            batch_size=args.ablation_batch_size,
            rae_decoder=rae_decoder,
            image_size=image_size,
        )

    # Global normalization
    vmin = heatmaps.min()
    vmax = heatmaps.max()
    print(f"Heatmap range: [{vmin:.6f}, {vmax:.6f}]")

    # Save individual token heatmaps
    print("Saving individual token heatmaps...")
    for token_idx in tqdm(range(num_tokens), desc="Saving plots"):
        plot_heatmap(
            heatmaps[token_idx],
            grid_size,
            vmin=vmin,
            vmax=vmax,
            save_path=output_dir / f"token_{token_idx:02d}",
        )

    # Save grid of all heatmaps
    print("Saving heatmap grid...")
    plot_heatmap_grid(
        heatmaps,
        grid_size,
        vmin=vmin,
        vmax=vmax,
        save_path=output_dir / "all_tokens_grid",
    )

    # Compute summary statistics
    token_importance = heatmaps.sum(axis=1)  # Total impact per token
    token_importance_normalized = token_importance / token_importance.sum()

    # Spatial specialization: entropy of each token's heatmap
    def spatial_entropy(heatmap):
        p = heatmap / (heatmap.sum() + 1e-10)
        return -np.sum(p * np.log(p + 1e-10))

    token_entropies = np.array([spatial_entropy(h) for h in heatmaps])
    max_entropy = np.log(num_patches)  # Uniform distribution entropy

    # Save results to JSON (only when not using precomputed)
    if not use_precomputed:
        results = {
            "num_images": num_images,
            "num_tokens": num_tokens,
            "num_patches": num_patches,
            "grid_size": grid_size,
            "heatmap_min": float(vmin),
            "heatmap_max": float(vmax),
            "token_importance": token_importance.tolist(),
            "token_importance_normalized": token_importance_normalized.tolist(),
            "token_spatial_entropy": token_entropies.tolist(),
            "max_spatial_entropy": float(max_entropy),
            "mean_spatial_entropy": float(token_entropies.mean()),
        }
        save_eval_results(args.flatdino_path, "token_ablation", results)

        # Save raw heatmaps as numpy array
        np.save(output_dir / "heatmaps.npy", heatmaps)

    # Process pixel ablation results (if enabled)
    pixel_token_importance = None
    pixel_token_importance_normalized = None
    pixel_output_dir = None
    if args.rae and pixel_heatmaps is not None:
        pixel_output_dir = args.flatdino_path / "token_ablation_pixel"
        pixel_output_dir.mkdir(parents=True, exist_ok=True)

        pixel_vmin = pixel_heatmaps.min()
        pixel_vmax = pixel_heatmaps.max()
        print(f"Pixel heatmap range: [{pixel_vmin:.6f}, {pixel_vmax:.6f}]")

        # Save individual pixel heatmaps
        print("Saving individual pixel heatmaps...")
        for token_idx in tqdm(range(num_tokens), desc="Saving pixel plots"):
            plot_pixel_heatmap(
                pixel_heatmaps[token_idx],
                vmin=pixel_vmin,
                vmax=pixel_vmax,
                save_path=pixel_output_dir / f"token_{token_idx:02d}_pixel",
            )

        # Save grid of all pixel heatmaps
        print("Saving pixel heatmap grid...")
        plot_pixel_heatmap_grid(
            pixel_heatmaps,
            vmin=pixel_vmin,
            vmax=pixel_vmax,
            save_path=pixel_output_dir / "all_tokens_pixel_grid",
        )

        # Compute pixel-level statistics
        pixel_token_importance = pixel_heatmaps.sum(axis=(1, 2))
        pixel_token_importance_normalized = pixel_token_importance / pixel_token_importance.sum()

        # Save raw pixel heatmaps and results (only when not using precomputed)
        if not use_precomputed_pixel:
            np.save(pixel_output_dir / "pixel_heatmaps.npy", pixel_heatmaps)

            pixel_results = {
                "pixel_heatmap_min": float(pixel_vmin),
                "pixel_heatmap_max": float(pixel_vmax),
                "pixel_token_importance": pixel_token_importance.tolist(),
                "pixel_token_importance_normalized": pixel_token_importance_normalized.tolist(),
                "image_size": image_size,
            }
            save_eval_results(args.flatdino_path, "token_ablation_pixel", pixel_results)

    # Print summary
    print("\n" + "=" * 50)
    print("TOKEN ABLATION SUMMARY")
    print("=" * 50)
    print(f"Checkpoint: {args.flatdino_path}")
    if num_images is not None:
        print(f"Images analyzed: {num_images}")
    print(f"Tokens: {num_tokens}, Patches: {num_patches} ({grid_size}x{grid_size})")
    print("\nTop 5 most important tokens (by total reconstruction impact):")
    top_tokens = np.argsort(token_importance)[::-1][:5]
    for rank, idx in enumerate(top_tokens):
        print(
            f"  {rank + 1}. Token {idx}: {token_importance[idx]:.4f} ({token_importance_normalized[idx] * 100:.1f}%)"
        )
    print(
        f"\nSpatial entropy: {token_entropies.mean():.2f} / {max_entropy:.2f} (higher = more distributed)"
    )
    print(f"\nResults saved to: {output_dir}")

    if args.rae and pixel_token_importance is not None:
        print("\n--- Pixel Ablation (RAE) ---")
        print(f"Pixel results saved to: {pixel_output_dir}")
        pixel_top_tokens = np.argsort(pixel_token_importance)[::-1][:5]
        print("Top 5 most important tokens (by pixel impact):")
        for rank, idx in enumerate(pixel_top_tokens):
            print(
                f"  {rank + 1}. Token {idx}: {pixel_token_importance[idx]:.4f} ({pixel_token_importance_normalized[idx] * 100:.1f}%)"
            )

    print("=" * 50)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
