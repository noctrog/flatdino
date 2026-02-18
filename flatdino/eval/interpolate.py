from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import math

import flax.nnx as nnx
import imageio_ffmpeg as ffmpeg
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import orbax.checkpoint as ocp
import tyro
from dacite import Config as DaciteConfig, from_dict

from flatdino.data import create_dataloaders
try:
    from ssljax.data.pusht import PushTDataset, TrajSlicerDataset
except ImportError:
    PushTDataset = TrajSlicerDataset = None
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig


@dataclass
class Config:
    flatdino_path: Path
    """Directory containing the FlatDINO autoencoder checkpoint."""

    output: Path | None = None
    """Path where the interpolation video will be written. Defaults to the checkpoint folder."""

    num_pairs: int = 16
    """Number of image pairs to interpolate."""

    interp_steps: int = 60
    """Number of interpolation steps (includes endpoints)."""

    interp_type: Literal["linear", "spherical"] = "linear"
    """Interpolation strategy between latent endpoints."""

    same_class_pairs: bool = False
    """If True, both images in a pair share the same label."""

    batch_size: int = 32

    seed: int = 0
    data: Literal["imagenet", "pusht"] = "imagenet"
    """Dataset to draw interpolation endpoints from."""
    fps: int = 6


def _tile_grid(images: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w, c = images.shape[1:]
    grid = np.zeros((rows * h, cols * w, c), dtype=images.dtype)
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        r, c_idx = divmod(idx, cols)
        grid[r * h : (r + 1) * h, c_idx * w : (c_idx + 1) * w] = img
    return grid


def _normalize_images(images: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (images - mean) / std


def _gather_pairs_imagenet(
    val_iter, num_pairs: int, same_class: bool
) -> list[tuple[jax.Array, jax.Array]]:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")

    pairs: list[tuple[jax.Array, jax.Array]] = []
    pending: dict[int, jax.Array] = {}
    pending_any: jax.Array | None = None

    for batch in val_iter:
        imgs = jnp.asarray(batch["image"])
        labels = np.asarray(batch["label"])
        for img, lbl in zip(imgs, labels):
            label_int = int(lbl)
            if same_class:
                if label_int in pending:
                    pairs.append((pending.pop(label_int), img))
                else:
                    pending[label_int] = img
            else:
                if pending_any is None:
                    pending_any = img
                else:
                    pairs.append((pending_any, img))
                    pending_any = None

            if len(pairs) >= num_pairs:
                return pairs

    raise RuntimeError("Unable to collect the requested number of pairs from the dataset.")


def _gather_pairs_pusht(
    num_pairs: int, *, mean: jax.Array, std: jax.Array, seed: int
) -> tuple[jax.Array, jax.Array]:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")

    ds = TrajSlicerDataset(
        PushTDataset("val"),
        num_frames=1,
        frameskip=1,
        shuffle=True,
        seed=seed,
    )

    required = num_pairs * 2
    if len(ds) < required:
        raise RuntimeError(
            f"PushT dataset does not contain enough samples ({len(ds)} found, {required} needed)."
        )

    frames = []
    for idx in range(required):
        sample = ds[idx]["visual"][0]  # (H, W, C) in [0, 1]
        frames.append(jnp.asarray(sample, dtype=jnp.float32))

    stacked = jnp.stack(frames, axis=0)
    normalized = _normalize_images(stacked, mean, std)
    return normalized[0::2], normalized[1::2]


def _encode_latents(
    images: jax.Array,
    *,
    dino,
    flatdino: FlatDinoAutoencoder,
) -> jax.Array:
    """Encode images to FlatDINO latents (tanh applied internally if configured)."""
    b, _, _, c = images.shape
    resized = jax.image.resize(images, (b, 224, 224, c), method="bilinear")
    patches = dino(resized)[:, 5:]
    z, _ = flatdino.encode(patches)
    return z


def _interpolate(
    a: jax.Array,
    b: jax.Array,
    steps: int,
    kind: Literal["linear", "spherical"],
) -> jax.Array:
    t = jnp.linspace(0.0, 1.0, steps, dtype=a.dtype)
    t = t[:, None, None]
    if kind == "linear":
        return (1.0 - t) * a + t * b

    eps = jnp.asarray(1e-6, dtype=a.dtype)
    a_norm = jnp.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = jnp.linalg.norm(b, axis=-1, keepdims=True)
    a_dir = a / jnp.clip(a_norm, eps, None)
    b_dir = b / jnp.clip(b_norm, eps, None)

    dot = jnp.clip(jnp.sum(a_dir * b_dir, axis=-1, keepdims=True), -1.0 + eps, 1.0 - eps)
    omega = jnp.arccos(dot)
    sin_omega = jnp.sin(omega)

    coeff1 = jnp.sin((1.0 - t) * omega) / jnp.clip(sin_omega, eps, None)
    coeff2 = jnp.sin(t * omega) / jnp.clip(sin_omega, eps, None)
    dir_interp = coeff1 * a_dir + coeff2 * b_dir
    scale = (1.0 - t) * a_norm + t * b_norm
    return dir_interp * scale


def main(cfg: Config):
    if cfg.interp_steps <= 1:
        raise ValueError("interp_steps must be greater than 1.")

    device_count = jax.device_count()
    if device_count != 1:
        raise RuntimeError("This script currently expects a single accelerator.")

    mesh = jax.make_mesh((device_count, 1), ("data", "model"))
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

    mean_vec = jnp.array(flatdino_cfg.data.normalization_mean, dtype=jnp.float32)
    std_vec = jnp.array(flatdino_cfg.data.normalization_std, dtype=jnp.float32)
    mean = mean_vec[None, None, None, :]
    std = std_vec[None, None, None, :]

    dino.eval()
    flatdino.encoder.eval()
    flatdino.decoder.eval()

    match cfg.data:
        case "imagenet":
            data = create_dataloaders(
                flatdino_cfg.data,
                batch_size=cfg.batch_size,
                val_aug=FlatDinoValAugmentations(flatdino_cfg.aug, flatdino_cfg.data),
                val_epochs=1,
                drop_remainder_val=False,
                train_epochs=1,
                drop_remainder_train=False,
            )
            val_iter = iter(data.val_loader)
            pairs = _gather_pairs_imagenet(val_iter, cfg.num_pairs, cfg.same_class_pairs)
            start_imgs = jnp.stack([p[0] for p in pairs])
            end_imgs = jnp.stack([p[1] for p in pairs])
        case "pusht":
            if cfg.same_class_pairs:
                raise ValueError("same_class_pairs requires labels; PushT samples are unlabeled.")
            start_imgs, end_imgs = _gather_pairs_pusht(
                cfg.num_pairs,
                mean=mean_vec.reshape((1, 1, 1, -1)),
                std=std_vec.reshape((1, 1, 1, -1)),
                seed=cfg.seed,
            )
        case _:
            raise ValueError(f"Unsupported dataset {cfg.data}")

    start_latents = _encode_latents(start_imgs, dino=dino, flatdino=flatdino)
    end_latents = _encode_latents(end_imgs, dino=dino, flatdino=flatdino)

    num_output_patches = flatdino_cfg.num_output_patches

    rae_decoder = make_rae_decoder(
        num_patches=num_output_patches,
        image_size=flatdino_cfg.aug.image_size[0],
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    @nnx.jit
    def decode(decoder_input: jax.Array) -> jax.Array:
        recon_tokens = flatdino.decode(decoder_input)
        rae_out = rae_decoder(recon_tokens)
        recon = rae_decoder.unpatchify(
            rae_out.logits,  # ty: ignore[possibly-unbound-attribute]
            original_image_size=flatdino_cfg.aug.image_size,
        )
        recon = jnp.transpose(recon, (0, 2, 3, 1))
        return jnp.clip(recon * std + mean, 0.0, 1.0)

    trajectories = []
    for s_lat, e_lat in zip(start_latents, end_latents):
        traj = _interpolate(s_lat, e_lat, cfg.interp_steps, cfg.interp_type)
        trajectories.append(traj)

    rows = int(math.ceil(cfg.num_pairs**0.5))
    cols = int(math.ceil(cfg.num_pairs / rows))

    output_path = cfg.output or (cfg.flatdino_path / "interpolations.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for step in range(cfg.interp_steps):
            step_latents = jnp.stack([traj[step] for traj in trajectories])
            step_imgs = decode(step_latents)
            step_np = np.asarray(jax.device_get(step_imgs))
            frame = _tile_grid(step_np, rows, cols)
            frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

            if writer is None:
                height, width, _ = frame_uint8.shape
                writer = ffmpeg.write_frames(
                    str(output_path),
                    size=(width, height),
                    fps=cfg.fps,
                    codec="libx264",
                    pix_fmt_out="yuv420p",
                )
                writer.send(None)  # prime generator

            writer.send(frame_uint8)
    finally:
        if writer is not None:
            writer.close()

    print(f"Saved interpolation video to {output_path}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
