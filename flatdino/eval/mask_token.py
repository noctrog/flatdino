from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import orbax.checkpoint as ocp
import tyro
from absl import logging
from dacite import Config as DaciteConfig, from_dict

from flatdino.data import create_dataloaders
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig


@dataclass
class Config:
    flatdino_path: Path
    """Directory containing the FlatDINO autoencoder checkpoint."""

    output_dir: Path | None = None
    """Where to write the grids; defaults to <flatdino_path>/masked_token_grids."""

    grid_size: int = 8
    """Grid rows/cols (grid_size x grid_size images)."""

    batch_size: int = 64
    """Batch size used for pulling validation images."""

    mask_mode: Literal["zero", "normal"] = "zero"
    """How to mask a token: zero it out or replace with random normal noise."""

    seed: int = 0


def _tile_grid(images: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Arrange (N, H, W, C) images into a single (rows*H, cols*W, C) grid."""
    h, w, c = images.shape[1:]
    grid = np.zeros((rows * h, cols * w, c), dtype=images.dtype)
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        r, c_idx = divmod(idx, cols)
        grid[r * h : (r + 1) * h, c_idx * w : (c_idx + 1) * w] = img
    return grid


def _denormalize(x: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return jnp.clip(x * std + mean, 0.0, 1.0)


def main(cfg: Config):
    if cfg.grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    num_devices = jax.device_count()
    if num_devices != 1:
        raise RuntimeError("This script expects a single accelerator.")

    mesh = jax.make_mesh((num_devices, 1), ("data", "model"))
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

    dino.eval()
    flatdino.encoder.eval()
    flatdino.decoder.eval()

    data = create_dataloaders(
        flatdino_cfg.data,
        batch_size=cfg.batch_size,
        val_aug=FlatDinoValAugmentations(flatdino_cfg.aug, flatdino_cfg.data),
        val_epochs=1,
        drop_remainder_val=False,
        train_epochs=1,
        drop_remainder_train=False,
    )
    batch = next(iter(data.val_loader))

    total_needed = cfg.grid_size * cfg.grid_size
    if batch["image"].shape[0] < total_needed:
        raise ValueError(
            f"Validation batch smaller than grid ({batch['image'].shape[0]} < {total_needed})."
        )

    imgs = jnp.asarray(batch["image"][:total_needed])

    mean = jnp.array(flatdino_cfg.data.normalization_mean, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(flatdino_cfg.data.normalization_std, dtype=jnp.float32)[None, None, None, :]

    resized = jax.image.resize(imgs, (imgs.shape[0], 224, 224, imgs.shape[-1]), method="bilinear")
    patches = dino(resized)[:, 5:]
    # Use FlatDinoAutoencoder encode (tanh applied internally if configured)
    z, _ = flatdino.encode(patches)
    num_tokens = z.shape[1]

    num_output_patches = flatdino_cfg.num_output_patches

    rae_decoder = make_rae_decoder(
        num_patches=num_output_patches,
        image_size=flatdino_cfg.aug.image_size[0],
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    @nnx.jit
    def decode_tokens(tokens: jax.Array) -> jax.Array:
        recon_tokens = flatdino.decode(tokens)
        rae_out = rae_decoder(recon_tokens)
        recon = rae_decoder.unpatchify(
            rae_out.logits,  # ty: ignore[possibly-unbound-attribute]
            original_image_size=flatdino_cfg.aug.image_size,
        )
        recon = jnp.transpose(recon, (0, 2, 3, 1))
        return _denormalize(recon, mean, std)

    rng = jax.random.PRNGKey(cfg.seed)

    originals = _denormalize(imgs, mean, std)
    decoded = decode_tokens(z)

    masked_recons = []
    for token_id in range(num_tokens):
        tokens = jnp.array(z)
        if cfg.mask_mode == "zero":
            mask_values = jnp.zeros_like(tokens[:, token_id, :])
        else:
            rng, sub = jax.random.split(rng)
            mask_values = jax.random.normal(sub, tokens[:, token_id, :].shape, dtype=tokens.dtype)
        tokens = tokens.at[:, token_id, :].set(mask_values)
        masked_recons.append(decode_tokens(tokens))

    rows = cols = cfg.grid_size

    output_dir = cfg.output_dir or (cfg.flatdino_path / "masked_token_grids")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_grid(array: jax.Array, name: str):
        np_img = np.asarray(jax.device_get(array))
        grid = _tile_grid(np_img, rows, cols)
        img_uint8 = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
        (output_dir / name).parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        imageio.imwrite(output_dir / name, img_uint8)

    _save_grid(originals, "original.png")
    _save_grid(decoded, "decoded.png")

    for idx, recon in enumerate(masked_recons):
        _save_grid(recon, f"masked_{idx:03d}.png")

    print(f"Saved masked token grids to {output_dir}")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    cfg = tyro.cli(Config)
    main(cfg)
