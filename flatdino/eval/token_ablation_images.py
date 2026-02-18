"""Per-image token ablation analysis for FlatDINO.

Visualizes what visual information each latent token encodes for individual images by:
1. Encoding an image to get latent tokens
2. For each token, zero it out and measure reconstruction change per patch
3. Overlay the heatmap on a semi-transparent version of the original image

Example usage:
    python -m flatdino.eval.token_ablation_images --flatdino-path output/flatdino/vae/med-32-ss-64
    python -m flatdino.eval.token_ablation_images --flatdino-path output/flatdino/vae/med-32-ss-64 --images 32
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import tyro
from tqdm import tqdm

from flatdino.pretrained.rae_decoder import make_rae_decoder

# Configure matplotlib for publication-quality figures
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["STIXGeneral"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["figure.dpi"] = 150

from flatdino.data import create_dataloaders
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import restore_encoder

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize an ImageNet-normalized image back to [0, 1] range."""
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0, 1)


@dataclass
class Args:
    flatdino_path: Path
    """Path to FlatDINO checkpoint."""
    images: int = 16
    """Number of images to process."""
    batch_size: int = 16
    """Batch size for processing."""
    image_alpha: float = 0.3
    """Alpha (transparency) for the background image (0=invisible, 1=fully visible)."""
    seed: int = 42


def compute_single_image_ablation(
    decoder: nnx.Module,
    latent: jax.Array,
    num_tokens: int,
    num_patches: int,
) -> tuple[np.ndarray, jax.Array, list[jax.Array]]:
    """Compute reconstruction error heatmap for each token ablation on a single image.

    Args:
        decoder: FlatDINO decoder
        latent: Encoded latent tokens (T, F) - mu values for a single image
        num_tokens: Number of latent tokens
        num_patches: Number of DINO patches

    Returns:
        heatmaps: (T, P) array of L2 distance per patch when token is ablated
        original_recon: (P, D) original reconstructed DINO patches
        ablated_recons: list of (P, D) ablated reconstructions per token
    """
    # Add batch dimension
    latent_batch = latent[None, :, :]  # (1, T, F)

    # Get original reconstruction
    original_recon = decoder(latent_batch, deterministic=True)[:, :num_patches]  # (1, P, D)

    heatmaps = []
    ablated_recons = []
    for token_idx in range(num_tokens):
        # Create ablation mask
        ablation_mask = jnp.ones(num_tokens).at[token_idx].set(0.0)

        # Apply ablation and reconstruct
        ablated_latents = latent_batch * ablation_mask[None, :, None]
        ablated_recon = decoder(ablated_latents, deterministic=True)[:, :num_patches]

        # Compute L2 distance per patch
        delta = jnp.linalg.norm(ablated_recon - original_recon, axis=-1)  # (1, P)
        heatmaps.append(np.array(delta[0]))
        ablated_recons.append(ablated_recon[0])  # (P, D)

    return np.stack(heatmaps, axis=0), original_recon[0], ablated_recons  # (T, P), (P, D), list of (P, D)


@nnx.jit(static_argnames=("image_h", "image_w"))
def decode_patches_to_rgb(
    rae_decoder,
    patches: jax.Array,
    mean: jax.Array,
    std: jax.Array,
    image_h: int,
    image_w: int,
) -> jax.Array:
    """Decode DINO patches to RGB image using RAE decoder.

    Args:
        rae_decoder: RAE decoder module
        patches: (B, P, D) DINO patch features
        mean: Normalization mean
        std: Normalization std
        image_h: Output image height
        image_w: Output image width

    Returns:
        RGB images (B, H, W, 3) in [0, 1] range
    """
    decoder_out = rae_decoder(patches)
    recon = rae_decoder.unpatchify(
        decoder_out.logits,
        original_image_size=(image_h, image_w),
    )
    recon = jnp.transpose(recon, (0, 2, 3, 1))  # (B, H, W, C)
    return jnp.clip(recon * std + mean, 0.0, 1.0)


def save_rgb_image(image: np.ndarray, path: Path):
    """Save a [0, 1] float image as PNG."""
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def plot_single_token_overlay(
    heatmap: np.ndarray,
    image: np.ndarray,
    grid_size: int,
    token_idx: int,
    vmin: float,
    vmax: float,
    image_alpha: float,
    save_path: Path,
):
    """Plot a single token's ablation heatmap overlaid on the original image."""
    # Reshape heatmap to spatial grid
    heatmap_2d = heatmap.reshape(grid_size, grid_size)

    # Resize heatmap to match image size
    from scipy.ndimage import zoom

    scale_factor = image.shape[0] / grid_size
    heatmap_resized = zoom(heatmap_2d, scale_factor, order=1)

    # Denormalize image for display
    image_display = denormalize_image(image)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    # Show original image with reduced opacity
    ax.imshow(image_display, alpha=image_alpha)

    # Overlay heatmap
    ax.imshow(
        heatmap_resized,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Save both formats
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_all_tokens_grid_overlay(
    heatmaps: np.ndarray,
    image: np.ndarray,
    grid_size: int,
    vmin: float,
    vmax: float,
    image_alpha: float,
    save_path: Path,
):
    """Plot all token ablation heatmaps in a grid, each overlaid on the original image."""
    from scipy.ndimage import zoom

    num_tokens = heatmaps.shape[0]

    # Compute grid dimensions (as square as possible)
    ncols = int(np.ceil(np.sqrt(num_tokens)))
    nrows = int(np.ceil(num_tokens / ncols))

    # Scale factor for resizing heatmaps
    scale_factor = image.shape[0] / grid_size

    # Denormalize image for display
    image_display = denormalize_image(image)

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

        # Reshape and resize heatmap
        heatmap_2d = heatmaps[idx].reshape(grid_size, grid_size)
        heatmap_resized = zoom(heatmap_2d, scale_factor, order=1)

        # Show original image with reduced opacity
        ax.imshow(image_display, alpha=image_alpha)

        # Overlay heatmap
        im = ax.imshow(
            heatmap_resized,
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # Hide empty subplots
    for idx in range(num_tokens, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\Delta$ (L2)", fontsize=10)

    # Save both formats
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(args: Args):
    # Create output directory
    output_dir = args.flatdino_path / "token_ablation_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    # Load encoder and decoder
    print(f"Loading checkpoint from {args.flatdino_path}")
    components = restore_encoder(args.flatdino_path, mesh=mesh, mp=mp, encoder=True, decoder=True)
    encoder = components.encoder
    decoder = components.decoder
    dino = components.dino
    num_tokens = components.num_flat_tokens

    assert encoder is not None
    assert decoder is not None

    # Get image size and normalization from config
    image_h, image_w = components.aug_cfg.image_size
    mean_vals = components.data_cfg.normalization_mean
    std_vals = components.data_cfg.normalization_std
    mean = jnp.array(mean_vals, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(std_vals, dtype=jnp.float32)[None, None, None, :]

    # Initialize RAE decoder for RGB reconstruction
    print("Loading RAE decoder for RGB reconstruction")
    patch_tokens = decoder.num_reg
    rae_decoder = make_rae_decoder(
        num_patches=patch_tokens,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=args.seed,
    )

    # Create validation dataloader
    print("Creating validation dataloader")
    data = create_dataloaders(
        components.data_cfg,
        args.batch_size,
        val_aug=FlatDinoValAugmentations(components.aug_cfg, components.data_cfg),
        val_epochs=1,
        drop_remainder_val=False,
    )

    num_images = min(args.images, data.val_ds_size)
    print(f"Processing {num_images} images...")

    # Collect images, latents, and patch info
    all_images = []
    all_latents = []
    num_collected = 0

    @nnx.jit
    def encode_batch(imgs: jax.Array) -> tuple[jax.Array, jax.Array]:
        b, h, w, c = imgs.shape
        imgs_resized = jax.image.resize(imgs, (b, 224, 224, c), method="bilinear")
        dino_patches = dino(imgs_resized)[:, 5:]  # Remove CLS and registers
        encoded = encoder(dino_patches)
        # Extract mu (first half of output dim)
        mu = encoded[:, :, : encoded.shape[-1] // 2]
        return mu[:, :num_tokens], dino_patches

    num_patches = None
    val_iter = iter(data.val_loader)
    for batch in tqdm(
        val_iter,
        total=(num_images + args.batch_size - 1) // args.batch_size,
        desc="Loading images",
    ):
        if num_collected >= num_images:
            break

        imgs = jnp.array(batch["image"])
        mu, patches = encode_batch(imgs)

        # Get num_patches from first batch
        if num_patches is None:
            num_patches = patches.shape[1]

        # Store original images (before resize, normalized to [0, 1])
        # Images come in as float32 in [0, 1] range from augmentations
        all_images.append(np.array(imgs))
        all_latents.append(np.array(mu))
        num_collected += len(imgs)

    # Concatenate and truncate to exact number
    images = np.concatenate(all_images, axis=0)[:num_images]
    latents = np.concatenate(all_latents, axis=0)[:num_images]

    # Get patch grid size (assuming square)
    grid_size = int(np.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, f"Patches ({num_patches}) must form a square grid"

    print(f"Images shape: {images.shape}")
    print(f"Latents shape: {latents.shape}")
    print(f"Num tokens: {num_tokens}, Num patches: {num_patches}, Grid: {grid_size}x{grid_size}")

    # Process each image
    for img_idx in tqdm(range(num_images), desc="Processing images"):
        img_dir = output_dir / f"image{img_idx:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Get image and latent for this sample
        image = images[img_idx]  # (H, W, C)
        latent = jnp.array(latents[img_idx])  # (T, F)

        # Compute ablation heatmaps for this image
        heatmaps, original_patches, ablated_patches_list = compute_single_image_ablation(
            decoder,
            latent,
            num_tokens,
            num_patches,
        )

        # Global normalization for this image
        vmin = heatmaps.min()
        vmax = heatmaps.max()

        # Save original image (denormalized for display)
        image_display = denormalize_image(image)
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        ax.imshow(image_display)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        fig.savefig(img_dir / "original.png", dpi=300, bbox_inches="tight", pad_inches=0)
        fig.savefig(img_dir / "original.pdf", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Decode and save RGB reconstruction from original latents
        original_rgb = decode_patches_to_rgb(
            rae_decoder, original_patches[None], mean, std, image_h, image_w
        )
        save_rgb_image(np.array(original_rgb[0]), img_dir / "recon_original.png")

        # Save individual token heatmaps overlaid on image, and ablated RGB reconstructions
        for token_idx in range(num_tokens):
            plot_single_token_overlay(
                heatmaps[token_idx],
                image,
                grid_size,
                token_idx,
                vmin=vmin,
                vmax=vmax,
                image_alpha=args.image_alpha,
                save_path=img_dir / f"token_{token_idx:02d}",
            )

            # Decode and save RGB reconstruction with this token ablated
            ablated_rgb = decode_patches_to_rgb(
                rae_decoder, ablated_patches_list[token_idx][None], mean, std, image_h, image_w
            )
            save_rgb_image(np.array(ablated_rgb[0]), img_dir / f"recon_ablated_{token_idx:02d}.png")

        # Save grid of all tokens
        plot_all_tokens_grid_overlay(
            heatmaps,
            image,
            grid_size,
            vmin=vmin,
            vmax=vmax,
            image_alpha=args.image_alpha,
            save_path=img_dir / "all_tokens_grid",
        )

    # Print summary
    print("\n" + "=" * 50)
    print("TOKEN ABLATION IMAGES SUMMARY")
    print("=" * 50)
    print(f"Checkpoint: {args.flatdino_path}")
    print(f"Images processed: {num_images}")
    print(f"Tokens: {num_tokens}")
    print(f"Output directory: {output_dir}")
    print()
    print("Files saved per image:")
    print("  - original.png/pdf: Original input image")
    print("  - recon_original.png: RGB reconstruction from all tokens")
    print("  - recon_ablated_XX.png: RGB reconstruction with token XX ablated")
    print("  - token_XX.png/pdf: Ablation heatmap for token XX")
    print("  - all_tokens_grid.png/pdf: Grid of all token heatmaps")
    print("=" * 50)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
