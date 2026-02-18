from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import chex
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import tyro
from dacite import Config as DaciteConfig, from_dict

from flatdino.data import DataConfig, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_dataloaders
try:
    from ssljax.data.pusht import PushTDataset, TrajSlicerDataset
except ImportError:
    PushTDataset = TrajSlicerDataset = None
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.augmentations import FlatDinoAugConfig, FlatDinoValAugmentations
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig


@dataclass
class Config:
    flatdino_path: Path
    """Directory containing the FlatDINO checkpoint."""

    output: Path = Path("/tmp/flatdino_reconstruction.png")
    samples: int = 4
    data: Literal["imagenet", "pusht"] = "imagenet"
    seed: int = 0


def _fetch_images(
    cfg: Config,
    *,
    restored,
    mu: jnp.ndarray,
    std: jnp.ndarray,
) -> jax.Array:
    restored.data_cfg.num_workers = 0
    match cfg.data:
        case "imagenet":
            batch_size = max(1, cfg.samples)
            data = create_dataloaders(
                restored.data_cfg,
                batch_size=batch_size,
                val_epochs=1,
                # val_aug=FlatDinoValAugmentations(restored.aug_cfg, restored.data_cfg),
                # TODO: future models should be trained with 256 in the data augmentation, i guess
                val_aug=FlatDinoValAugmentations(
                    replace(restored.aug_cfg, image_size=(256, 256)), restored.data_cfg
                ),
            )
            batch = next(iter(data.val_loader))
            images = jax.device_put(batch["image"][: cfg.samples])
        case "pusht":
            dataset = TrajSlicerDataset(
                PushTDataset("val"),
                num_frames=1,
                frameskip=1,
                shuffle=True,
                seed=cfg.seed,
            )
            if cfg.samples > len(dataset):
                raise ValueError(
                    f"Requested {cfg.samples} samples but dataset only has {len(dataset)}."
                )

            mu_vec = mu[0, 0, 0]
            std_vec = std[0, 0, 0]
            frames = []
            for idx in range(cfg.samples):
                sample = dataset[idx]["visual"][0]  # (H, W, C) in [0, 1]
                frame = jnp.asarray(sample, dtype=jnp.float32)
                frame = (frame - mu_vec) / std_vec
                frames.append(frame)

            images = jnp.stack(frames, axis=0)
        case _:
            raise ValueError(f"Unsupported dataset {cfg.data}")
    return images


def main(cfg: Config):
    assert jax.device_count() == 1, "This script expects a single accelerator."
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)
    dacite_config = DaciteConfig(cast=[tuple], strict=False)

    # Load checkpoint and config
    ckpt_path = cfg.flatdino_path.absolute()
    mngr = ocp.CheckpointManager(
        ckpt_path,
        item_names=FlatDinoAutoencoder.get_item_names() + ["optim", "loader", "config"],
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    step = mngr.best_step()
    assert step is not None, "Failed to load best step from checkpoint"
    print(f"Found checkpoint at step {step}. Restoring...")

    cfg_ckpt = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"]
    flatdino_cfg = from_dict(FlatDinoConfig, cfg_ckpt, config=dacite_config)

    # Create models
    dino = DinoWithRegisters(flatdino_cfg.dino_name, resolution=224, dtype=mp.param_dtype)
    flatdino = FlatDinoAutoencoder(
        flatdino_cfg, mesh, mngr=mngr, step=step, mp=mp, rngs=nnx.Rngs(cfg.seed)
    )

    dino_params = jax.tree.leaves(nnx.state(dino, nnx.Param))
    dino_params = sum(jax.tree.map(lambda x: jnp.size(x), dino_params))
    encoder_params = jax.tree.leaves(nnx.state(flatdino.encoder, nnx.Param))
    encoder_params = sum(jax.tree.map(lambda x: jnp.size(x), encoder_params))
    decoder_params = jax.tree.leaves(nnx.state(flatdino.decoder, nnx.Param))
    decoder_params = sum(jax.tree.map(lambda x: jnp.size(x), decoder_params))
    print(f"DINO nb. params: {dino_params / 1_000_000:.2f}M params")
    print(f"Encoder nb. params: {encoder_params / 1_000_000:.2f}M params")
    print(f"Decoder nb. params: {decoder_params / 1_000_000:.2f}M params")

    dino.eval()
    flatdino.encoder.eval()
    flatdino.decoder.eval()

    mu = jnp.array(IMAGENET_DEFAULT_MEAN, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(IMAGENET_DEFAULT_STD, dtype=jnp.float32)[None, None, None, :]

    # Create a simple object to pass to _fetch_images (matching expected interface)
    @dataclass
    class _Restored:
        data_cfg: DataConfig
        aug_cfg: FlatDinoAugConfig

    restored_compat = _Restored(data_cfg=flatdino_cfg.data, aug_cfg=flatdino_cfg.aug)

    images = _fetch_images(cfg, restored=restored_compat, mu=mu, std=std).astype(jnp.float32)
    if images.shape[0] == 0:
        raise ValueError("No images were loaded for reconstruction.")

    image_size = images.shape[1]
    if images.shape[1] != images.shape[2]:
        raise ValueError("Expected square inputs; got shape {images.shape}.")

    sample_patches = dino(
        jax.image.resize(
            images[:1], (1, 224, 224, images.shape[-1]), method="linear", antialias=True
        )
    )[:, 5:]
    num_patch_tokens = sample_patches.shape[1]

    num_output_patches = flatdino_cfg.num_output_patches
    if num_output_patches != num_patch_tokens:
        raise ValueError(
            f"Decoder expects {num_output_patches} output patches but DINO produced {num_patch_tokens}."
        )

    rae_decoder = make_rae_decoder(
        num_patches=num_output_patches,
        image_size=image_size,
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    def reconstruct(batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        chex.assert_rank(batch, 4)
        b, _, _, d = batch.shape
        batch = jax.image.resize(
            batch, (b, dino.resolution, dino.resolution, d), method="linear", antialias=True
        )
        patches = dino(batch)[:, 5:]

        recon_patches = rae_decoder(patches)
        recon_patches = rae_decoder.unpatchify(
            recon_patches.logits,  # ty: ignore[possibly-unbound-attribute]
            original_image_size=(image_size, image_size),
        )
        recon_patches = jnp.transpose(recon_patches, (0, 2, 3, 1))

        # Use FlatDinoAutoencoder encode/decode methods (tanh is applied internally if configured)
        z, _ = flatdino.encode(patches)
        recon_tokens = flatdino.decode(z)

        decoder_out = rae_decoder(recon_tokens)
        recon_flatdino = rae_decoder.unpatchify(
            decoder_out.logits,  # ty: ignore[possibly-unbound-attribute]
            original_image_size=(image_size, image_size),
        )
        recon_flatdino = jnp.transpose(recon_flatdino, (0, 2, 3, 1))
        return recon_patches, recon_flatdino

    recon_dino, recon_flatdino = reconstruct(images)

    original = jnp.clip(images * std + mu, 0.0, 1.0)
    recon_dino = jnp.clip(recon_dino * std + mu, 0.0, 1.0)
    recon_flatdino = jnp.clip(recon_flatdino * std + mu, 0.0, 1.0)

    original_np = np.asarray(jax.device_get(original))
    recon_dino_np = np.asarray(jax.device_get(recon_dino))
    recon_flatdino_np = np.asarray(jax.device_get(recon_flatdino))

    mse_dino = np.mean((recon_dino_np - original_np) ** 2, axis=(1, 2, 3))
    psnr_dino = -10.0 * np.log10(np.maximum(mse_dino, 1e-10))
    mse_flatdino = np.mean((recon_flatdino_np - original_np) ** 2, axis=(1, 2, 3))
    psnr_flatdino = -10.0 * np.log10(np.maximum(mse_flatdino, 1e-10))

    # Assuming plt, np, original_np, recon_dino_np, recon_flatdino_np,
    # psnr_dino, psnr_flatdino, and cfg are defined

    cols = original_np.shape[0]
    # constrained_layout=True is good, keep it
    fig, axes = plt.subplots(3, cols, figsize=(cols * 3, 9), constrained_layout=True)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "serif",
            "font.serif": "Times New Roman",
            "axes.titlesize": 20,
            "axes.titlepad": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    row_titles = ["Original", "DINO", "FlatDINO"]

    for row_idx in range(3):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            # -----------------------

            if row_idx == 0:
                ax.imshow(original_np[col_idx])
            elif row_idx == 1:
                ax.imshow(recon_dino_np[col_idx])
                # This label will now be visible
                ax.set_xlabel(f"PSNR: {psnr_dino[col_idx]:.1f} dB", fontsize=18)
            else:
                ax.imshow(recon_flatdino_np[col_idx])
                # This label will now be visible
                ax.set_xlabel(f"PSNR: {psnr_flatdino[col_idx]:.2f} dB", fontsize=18)

        # This label will now be visible
        axes[row_idx, 0].set_ylabel(
            row_titles[row_idx],
            rotation=0,
            fontsize=11,
            fontweight="semibold",
            labelpad=40,
            va="center",
        )

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    mean_dino_mse = float(np.mean(mse_dino))
    mean_dino_mse = float(np.mean(psnr_dino))
    mean_flatdino_mse = float(np.mean(mse_flatdino))
    mean_flatdino_mse = float(np.mean(psnr_flatdino))
    print(f"Saved reconstructions to {cfg.output}")
    print(f"DINO: Average MSE: {mean_dino_mse:.6f} | Average PSNR: {mean_dino_mse:.2f} dB")
    print(
        f"FlatDINO: Average MSE: {mean_flatdino_mse:.6f} | Average PSNR: {mean_flatdino_mse:.2f} dB"
    )


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
