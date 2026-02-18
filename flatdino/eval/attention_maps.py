"""Visualize attention maps from the FlatDINO encoder and decoder.

This script extracts and visualizes attention weights from the FlatDINO model:
- Encoder: How each latent token attends to input DINO patches (compression)
- Decoder: How each output patch attends to latent tokens (reconstruction)

Example usage:
    # Both encoder and decoder (default)
    python -m flatdino.eval.attention_maps \
        --checkpoint output/flatdino/vae/fast-32-sb-128 \
        --num-images 4

    # Encoder only
    python -m flatdino.eval.attention_maps \
        --checkpoint output/flatdino/vae/fast-32-sb-128 \
        --mode encoder

    # Decoder only, specific layer and head
    python -m flatdino.eval.attention_maps \
        --checkpoint output/flatdino/vae/fast-32-sb-128 \
        --mode decoder --layer -1 --head 0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import jmp
import tyro
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from flatdino.data import DataConfig, create_dataloaders
from flatdino.distributed import prefetch_to_mesh
from flatdino.eval import extract_mu, restore_encoder
from flatdino.augmentations import FlatDinoValAugmentations


@dataclass
class Config:
    checkpoint: Path
    """Path to FlatDINO checkpoint."""

    mode: Literal["encoder", "decoder", "both"] = "both"
    """Which attention maps to visualize."""

    num_images: int = 4
    """Number of images to visualize."""

    layer: int = -1
    """Layer index to visualize (-1 for last layer)."""

    head: int | None = None
    """Head index to visualize (None for mean across heads)."""

    start_token: int = 0
    """Starting token index for visualization (shows tokens start_token to start_token+7)."""

    include_disposable: bool = False
    """Include disposable registers in visualization. When enabled, shows all tokens including
    disposable registers instead of just the latent tokens."""

    gpu_batch_size: int = 8
    """Batch size per GPU."""

    output_stem: str = "attention_maps"
    """Output filename stem."""

    data: DataConfig = field(default_factory=lambda: DataConfig())


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """Denormalize ImageNet-normalized image to [0, 1]."""
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


def visualize_encoder_attention(
    images: np.ndarray,
    attention_weights: np.ndarray,
    num_latents: int,
    layer_idx: int,
    head_idx: int | None,
    output_path: Path,
    output_stem: str,
    start_token: int = 0,
    encoder_disposable_registers: int = 0,
    include_disposable: bool = False,
):
    """Visualize encoder attention: how latent tokens attend to input DINO patches.

    Args:
        images: Original images, shape (N, H, W, C)
        attention_weights: Attention weights from one layer, shape (N, num_heads, T, T)
            where T = disposable + num_latents + num_patches
        num_latents: Number of latent tokens (excluding disposable registers)
        layer_idx: Layer index (for title)
        head_idx: Head index or None for mean
        output_path: Directory to save outputs
        output_stem: Filename stem
        start_token: Starting token index for visualization
        encoder_disposable_registers: Number of disposable registers at the start of sequence
        include_disposable: If True, include disposable registers in visualization
    """
    n_images = images.shape[0]
    total_tokens = attention_weights.shape[2]
    num_registers = encoder_disposable_registers + num_latents
    num_patches = total_tokens - num_registers

    # Compute patch grid size (assuming square)
    patch_grid = int(np.sqrt(num_patches))
    assert patch_grid * patch_grid == num_patches, f"Expected square grid, got {num_patches} patches"

    # Extract attention from register tokens to patches
    # Sequence order: [disposable, latents, patches]
    if include_disposable:
        # Include all register tokens (disposable + latents)
        reg_to_patch_attn = attention_weights[:, :, :num_registers, num_registers:]
        num_tokens_total = num_registers
        token_labels = [f"Disp {i}" for i in range(encoder_disposable_registers)] + \
                       [f"Latent {i}" for i in range(num_latents)]
    else:
        # Only latent tokens (skip disposable)
        latent_start = encoder_disposable_registers
        latent_end = latent_start + num_latents
        reg_to_patch_attn = attention_weights[:, :, latent_start:latent_end, num_registers:]
        num_tokens_total = num_latents
        token_labels = [f"Latent {i}" for i in range(num_latents)]

    if head_idx is not None:
        attn = reg_to_patch_attn[:, head_idx]  # (N, num_tokens, num_patches)
        head_str = f"head_{head_idx}"
    else:
        attn = np.mean(reg_to_patch_attn, axis=1)  # (N, num_tokens, num_patches)
        head_str = "mean_heads"

    # Create visualization: rows = images, cols = tokens + original image
    n_tokens_to_show = min(8, num_tokens_total - start_token)
    n_cols = n_tokens_to_show + 1
    n_rows = n_images

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for img_idx in range(n_images):
        img = denormalize_image(images[img_idx])

        # Original image
        ax = axes[img_idx, 0]
        ax.imshow(img)
        ax.set_title("Image" if img_idx == 0 else "")
        ax.axis("off")

        # Attention maps for each token
        for i in range(n_tokens_to_show):
            tok_idx = start_token + i
            ax = axes[img_idx, i + 1]
            attn_map = attn[img_idx, tok_idx].reshape(patch_grid, patch_grid)

            ax.imshow(img)
            ax.imshow(
                attn_map,
                cmap="hot",
                alpha=0.6,
                interpolation="bilinear",
                extent=[0, img.shape[1], img.shape[0], 0],
                norm=Normalize(vmin=0, vmax=attn_map.max()),
            )
            ax.set_title(token_labels[tok_idx] if img_idx == 0 else "")
            ax.axis("off")

    end_token = start_token + n_tokens_to_show - 1
    token_type = "All tokens" if include_disposable else "Latent tokens"
    plt.suptitle(f"Encoder Attention (Layer {layer_idx}, {head_str})\n{token_type} {start_token}-{end_token} attending to DINO patches", fontsize=12)
    plt.tight_layout()

    suffix = "_with_disp" if include_disposable else ""
    for ext in ("png", "pdf"):
        fig.savefig(output_path / f"{output_stem}_encoder_layer{layer_idx}_{head_str}{suffix}.{ext}", dpi=150)
    plt.close(fig)

    # Attention matrix for first image
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(attn[0], aspect="auto", cmap="viridis")
    ax.set_xlabel("Input patch index")
    ylabel = "Token index (disp + latent)" if include_disposable else "Latent token index"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Encoder Attention Matrix (Layer {layer_idx}, {head_str})")
    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_path / f"{output_stem}_encoder_matrix_layer{layer_idx}_{head_str}{suffix}.{ext}", dpi=150)
    plt.close(fig)


def visualize_decoder_attention(
    images: np.ndarray,
    attention_weights: np.ndarray,
    num_latent_tokens: int,
    num_output_patches: int,
    layer_idx: int,
    head_idx: int | None,
    output_path: Path,
    output_stem: str,
    start_token: int = 0,
    decoder_disposable_registers: int = 0,
    include_disposable: bool = False,
):
    """Visualize decoder attention: how output patches attend to latent tokens.

    Args:
        images: Original images, shape (N, H, W, C)
        attention_weights: Attention weights from one layer, shape (N, num_heads, T, T)
            where T = disposable + num_output_patches + num_latent_tokens
        num_latent_tokens: Number of latent tokens (input to decoder)
        num_output_patches: Number of output patches (excluding disposable registers)
        layer_idx: Layer index (for title)
        head_idx: Head index or None for mean
        output_path: Directory to save outputs
        output_stem: Filename stem
        start_token: Starting latent token index for visualization
        decoder_disposable_registers: Number of disposable registers at the start of sequence
        include_disposable: If True, include disposable registers in visualization
    """
    n_images = images.shape[0]

    # Compute patch grid size (assuming square)
    patch_grid = int(np.sqrt(num_output_patches))
    assert patch_grid * patch_grid == num_output_patches, f"Expected square grid, got {num_output_patches} patches"

    # In the decoder:
    # Sequence order: [disposable, output_patches, latent_input]
    latent_start = decoder_disposable_registers + num_output_patches

    if include_disposable:
        # Include disposable registers: show attention from (disposable + patches) to latents
        num_queries = decoder_disposable_registers + num_output_patches
        query_to_latent_attn = attention_weights[:, :, :num_queries, latent_start:]
        [f"Disp {i}" for i in range(decoder_disposable_registers)] + \
                       [f"Patch {i}" for i in range(num_output_patches)]
    else:
        # Only output patches (skip disposable)
        output_start = decoder_disposable_registers
        output_end = output_start + num_output_patches
        query_to_latent_attn = attention_weights[:, :, output_start:output_end, latent_start:]
        num_queries = num_output_patches
        [f"Patch {i}" for i in range(num_output_patches)]

    if head_idx is not None:
        attn = query_to_latent_attn[:, head_idx]  # (N, num_queries, num_latent_tokens)
        head_str = f"head_{head_idx}"
    else:
        attn = np.mean(query_to_latent_attn, axis=1)  # (N, num_queries, num_latent_tokens)
        head_str = "mean_heads"

    # Visualization 1: For each latent token, show which queries attend to it
    # For disposable queries, we can't reshape to patch grid, so we show a different view
    if include_disposable and decoder_disposable_registers > 0:
        # Show disposable registers' attention as a separate bar plot or matrix
        # For now, just show the attention matrix
        pass
    else:
        n_latents_to_show = min(8, num_latent_tokens - start_token)
        n_cols = n_latents_to_show + 1
        n_rows = n_images

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for img_idx in range(n_images):
            img = denormalize_image(images[img_idx])

            # Original image
            ax = axes[img_idx, 0]
            ax.imshow(img)
            ax.set_title("Image" if img_idx == 0 else "")
            ax.axis("off")

            # For each latent token, show how much each output patch attends to it
            for i in range(n_latents_to_show):
                lat_idx = start_token + i
                ax = axes[img_idx, i + 1]
                # attn shape: (N, num_output_patches, num_latent_tokens)
                # Get attention TO this latent token FROM all patches
                attn_to_latent = attn[img_idx, :, lat_idx].reshape(patch_grid, patch_grid)

                ax.imshow(img)
                ax.imshow(
                    attn_to_latent,
                    cmap="hot",
                    alpha=0.6,
                    interpolation="bilinear",
                    extent=[0, img.shape[1], img.shape[0], 0],
                    norm=Normalize(vmin=0, vmax=attn_to_latent.max()),
                )
                ax.set_title(f"Latent {lat_idx}" if img_idx == 0 else "")
                ax.axis("off")

        end_token = start_token + n_latents_to_show - 1
        plt.suptitle(f"Decoder Attention (Layer {layer_idx}, {head_str})\nOutput patches attending to latent tokens {start_token}-{end_token}", fontsize=12)
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(output_path / f"{output_stem}_decoder_layer{layer_idx}_{head_str}.{ext}", dpi=150)
        plt.close(fig)

    # Attention matrix for first image (always shown, includes disposable if requested)
    suffix = "_with_disp" if include_disposable else ""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn[0], aspect="auto", cmap="viridis")
    ax.set_xlabel("Latent token index")
    ylabel = "Query index (disp + patches)" if include_disposable else "Output patch index"
    ax.set_ylabel(ylabel)

    # Add horizontal line to separate disposable from patches if applicable
    if include_disposable and decoder_disposable_registers > 0:
        ax.axhline(y=decoder_disposable_registers - 0.5, color="white", linewidth=2, linestyle="--")
        ax.text(0.02, decoder_disposable_registers / 2 / num_queries, "Disp",
                transform=ax.get_yaxis_transform(), color="white", fontsize=10, va="center")

    ax.set_title(f"Decoder Attention Matrix (Layer {layer_idx}, {head_str})")
    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_path / f"{output_stem}_decoder_matrix_layer{layer_idx}_{head_str}{suffix}.{ext}", dpi=150)
    plt.close(fig)


def main(cfg: Config):
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    load_encoder = cfg.mode in ("encoder", "both")
    load_decoder = cfg.mode in ("decoder", "both")

    r = restore_encoder(cfg.checkpoint, mesh=mesh, mp=mp, encoder=load_encoder, decoder=load_decoder)
    r.dino.eval()
    if r.encoder is not None:
        r.encoder.eval()
    if r.decoder is not None:
        r.decoder.eval()

    if load_encoder:
        num_enc_layers = len(r.encoder.blocks)
        print(f"Encoder: {r.num_flat_tokens} latent tokens, {num_enc_layers} layers")
        if r.encoder_disposable_registers > 0:
            print(f"  (with {r.encoder_disposable_registers} disposable registers)")

    if load_decoder:
        num_output_patches = r.decoder.num_reg - r.decoder_disposable_registers
        num_dec_layers = len(r.decoder.blocks)
        print(f"Decoder: {num_output_patches} output patches, {num_dec_layers} layers")
        if r.decoder_disposable_registers > 0:
            print(f"  (with {r.decoder_disposable_registers} disposable registers)")

    batch_size = cfg.gpu_batch_size * jax.device_count()
    data = create_dataloaders(
        cfg.data,
        batch_size=batch_size,
        val_epochs=1,
        val_aug=FlatDinoValAugmentations(r.aug_cfg, r.data_cfg),
    )

    @nnx.jit
    def forward_encoder_with_attention(imgs: jax.Array) -> tuple[jax.Array, jax.Array, list[jax.Array]]:
        patches = r.dino(imgs)[:, 5:]
        encoded, attn_weights = r.encoder(patches, deterministic=True, return_attention_weights=True)
        mu = extract_mu(encoded, r.num_flat_tokens, r.encoder_disposable_registers)
        return imgs, mu, attn_weights

    @nnx.jit
    def forward_decoder_with_attention(mu: jax.Array) -> list[jax.Array]:
        _, attn_weights = r.decoder(mu, deterministic=True, return_attention_weights=True)
        return attn_weights

    @nnx.jit
    def forward_both_with_attention(imgs: jax.Array) -> tuple[jax.Array, list[jax.Array], list[jax.Array]]:
        patches = r.dino(imgs)[:, 5:]
        encoded, enc_attn = r.encoder(patches, deterministic=True, return_attention_weights=True)
        mu = extract_mu(encoded, r.num_flat_tokens, r.encoder_disposable_registers)
        _, dec_attn = r.decoder(mu, deterministic=True, return_attention_weights=True)
        return imgs, enc_attn, dec_attn

    # Get first batch
    val_iter = iter(data.val_loader)
    batch = next(prefetch_to_mesh(val_iter, 1, mesh, trim=True))
    imgs = batch["image"][:cfg.num_images]

    print(f"Processing {imgs.shape[0]} images...")

    output_dir = cfg.checkpoint if cfg.checkpoint.is_dir() else cfg.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "encoder":
        imgs_out, _, enc_attn = forward_encoder_with_attention(imgs)
        imgs_np = np.asarray(jax.device_get(imgs_out))
        enc_attn_np = [np.asarray(jax.device_get(w)) for w in enc_attn]

        layer_idx = cfg.layer if cfg.layer >= 0 else num_enc_layers + cfg.layer
        print(f"Visualizing encoder layer {layer_idx}, tokens {cfg.start_token}-{cfg.start_token + 7}...")
        visualize_encoder_attention(
            imgs_np, enc_attn_np[layer_idx], r.num_flat_tokens,
            layer_idx, cfg.head, output_dir, cfg.output_stem, cfg.start_token,
            r.encoder_disposable_registers, cfg.include_disposable,
        )

    elif cfg.mode == "decoder":
        imgs_out, mu, _ = forward_encoder_with_attention(imgs)
        dec_attn = forward_decoder_with_attention(mu)
        imgs_np = np.asarray(jax.device_get(imgs_out))
        dec_attn_np = [np.asarray(jax.device_get(w)) for w in dec_attn]

        layer_idx = cfg.layer if cfg.layer >= 0 else num_dec_layers + cfg.layer
        print(f"Visualizing decoder layer {layer_idx}, tokens {cfg.start_token}-{cfg.start_token + 7}...")
        visualize_decoder_attention(
            imgs_np, dec_attn_np[layer_idx], r.num_flat_tokens, num_output_patches,
            layer_idx, cfg.head, output_dir, cfg.output_stem, cfg.start_token,
            r.decoder_disposable_registers, cfg.include_disposable,
        )

    else:  # both
        imgs_out, enc_attn, dec_attn = forward_both_with_attention(imgs)
        imgs_np = np.asarray(jax.device_get(imgs_out))
        enc_attn_np = [np.asarray(jax.device_get(w)) for w in enc_attn]
        dec_attn_np = [np.asarray(jax.device_get(w)) for w in dec_attn]

        enc_layer_idx = cfg.layer if cfg.layer >= 0 else num_enc_layers + cfg.layer
        dec_layer_idx = cfg.layer if cfg.layer >= 0 else num_dec_layers + cfg.layer

        print(f"Visualizing encoder layer {enc_layer_idx}, tokens {cfg.start_token}-{cfg.start_token + 7}...")
        visualize_encoder_attention(
            imgs_np, enc_attn_np[enc_layer_idx], r.num_flat_tokens,
            enc_layer_idx, cfg.head, output_dir, cfg.output_stem, cfg.start_token,
            r.encoder_disposable_registers, cfg.include_disposable,
        )

        print(f"Visualizing decoder layer {dec_layer_idx}, tokens {cfg.start_token}-{cfg.start_token + 7}...")
        visualize_decoder_attention(
            imgs_np, dec_attn_np[dec_layer_idx], r.num_flat_tokens, num_output_patches,
            dec_layer_idx, cfg.head, output_dir, cfg.output_stem, cfg.start_token,
            r.decoder_disposable_registers, cfg.include_disposable,
        )

    print(f"Saved attention maps to {output_dir}")


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
