"""Generate reconstructed images for rFID evaluation (VQDino variant).

Pipeline: validation image (uint8) → normalize → DINOv2 → VQDino encoder → FSQ quantize → VQDino decoder → RAE decoder → reconstruction

Images are loaded from the ImageNet validation set (50k images), center-cropped to
256x256 using ADM preprocessing.

Output directory defaults to vqdino_path/recon_samples.

Usage:
    python -m flatdino.eval.rfid_adm_vq \
        --vqdino-path output/flatdino/vq/model
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "float32")

import flax.nnx as nnx
import jax.numpy as jnp
import jmp
import numpy as np
import orbax.checkpoint as ocp
import tyro
from absl import logging
from dacite import Config as DaciteConfig, from_dict
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from PIL import Image
from tqdm import tqdm

from flatdino.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flatdino.metrics.adm import get_imagenet_256_val
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.vq_autoencoder import VQDinoAutoencoder, VQDinoConfig


@dataclass
class Config:
    vqdino_path: Path
    """Directory containing the VQDino checkpoint."""

    output_dir: Path | None = None
    """Directory to save reconstructed images. Defaults to vqdino_path/recon_samples."""

    num_samples: int | None = None
    """Number of samples to reconstruct. Defaults to all images in reference."""

    batch_size: int = 256

    seed: int = 0

    save_npz: bool = True
    """Save all samples to a single NPZ file (for ADM evaluator)."""

    save_png: bool = False
    """Save individual PNG files."""


@nnx.jit(static_argnames=("image_h", "image_w"))
def reconstruct_batch(
    images_uint8: jax.Array,
    dino,
    vqdino: VQDinoAutoencoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    image_h: int,
    image_w: int,
) -> jax.Array:
    b = images_uint8.shape[0]

    images = images_uint8.astype(jnp.float32) / 255.0
    images_norm = (images - mean) / std

    dino_input = jax.image.resize(
        images_norm,
        (b, dino.resolution, dino.resolution, images_norm.shape[-1]),
        method="linear",
        antialias=True,
    )
    patches = dino(dino_input)[:, 5:]

    z, _ = vqdino.encode(patches)
    recon_tokens = vqdino.decode(z)

    decoder_out = rae_decoder(recon_tokens)
    recon = rae_decoder.unpatchify(decoder_out.logits, original_image_size=(image_h, image_w))
    recon = jnp.transpose(recon, (0, 2, 3, 1))

    return jnp.clip(recon * std + mean, 0.0, 1.0)


def fp_to_uint8(images: np.ndarray) -> np.ndarray:
    return np.clip(images * 255 + 0.5, 0, 255).astype(np.uint8)


def restore_model_state(
    restore_mngr: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    mesh: Mesh,
) -> None:
    model_state = nnx.state(model)
    pure_state = nnx.to_pure_dict(model_state)
    restore_args = jax.tree.map(
        lambda x: ocp.ArrayRestoreArgs(sharding=NamedSharding(mesh, P())),
        pure_state,
    )
    pure_state = restore_mngr.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeRestore(pure_state, restore_args=restore_args)
        ),
    )["model"]
    nnx.replace_by_pure_dict(model_state, pure_state)
    nnx.update(model, model_state)


def main(cfg: Config):
    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = cfg.vqdino_path / "recon_samples"

    logging.info("Generating reconstructed images for ADM rFID evaluation (VQDino)")
    logging.info(f"Output directory: {output_dir}")

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)

    if cfg.batch_size % mesh.size != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by the number of devices ({mesh.size})"
        )
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.float32, output_dtype=jnp.float32)
    dacite_config = DaciteConfig(cast=[tuple], strict=False)

    reference_path = get_imagenet_256_val()
    logging.info(f"Loading reference images from {reference_path}")
    ref_data = np.load(reference_path)
    ref_images = ref_data["arr_0"]
    logging.info(f"Loaded {ref_images.shape[0]} reference images, shape: {ref_images.shape}")

    total_available = ref_images.shape[0]
    num_samples = cfg.num_samples if cfg.num_samples is not None else total_available
    num_samples = min(num_samples, total_available)

    # Load checkpoint
    ckpt_path = cfg.vqdino_path.absolute()
    mngr = ocp.CheckpointManager(
        ckpt_path,
        item_names=["model", "optim", "loader", "config"],
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    step = mngr.best_step()
    assert step is not None, "Failed to load best step from checkpoint"
    logging.info(f"Found checkpoint at step {step}. Restoring...")

    cfg_ckpt = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"]
    vqdino_cfg = from_dict(VQDinoConfig, cfg_ckpt, config=dacite_config)

    # Create models
    dino = DinoWithRegisters(vqdino_cfg.dino_name, resolution=224, dtype=mp.param_dtype)
    vqdino = VQDinoAutoencoder(vqdino_cfg, mp=mp, rngs=nnx.Rngs(cfg.seed))
    restore_model_state(mngr, step, vqdino, mesh)

    dino.eval()
    vqdino.encoder.eval()
    vqdino.decoder.eval()

    image_h, image_w = ref_images.shape[1], ref_images.shape[2]
    num_output_patches = vqdino_cfg.num_output_patches

    rae_decoder = make_rae_decoder(
        num_patches=num_output_patches,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    mean = jnp.array(IMAGENET_DEFAULT_MEAN, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(IMAGENET_DEFAULT_STD, dtype=jnp.float32)[None, None, None, :]

    shard = NamedSharding(mesh, P("data"))

    reconstruct_fn = partial(
        reconstruct_batch,
        dino=dino,
        vqdino=vqdino,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        image_h=image_h,
        image_w=image_w,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    total_collected = 0

    logging.info(f"Reconstructing {num_samples} images")
    logging.info(f"Batch size: {cfg.batch_size}")

    pbar = tqdm(total=num_samples, desc="Reconstructing")

    for batch_start in range(0, num_samples, cfg.batch_size):
        batch_end = min(batch_start + cfg.batch_size, num_samples)
        images = ref_images[batch_start:batch_end]
        batch_len = images.shape[0]

        if batch_len % mesh.size != 0:
            pad_size = mesh.size - (batch_len % mesh.size)
            images = np.concatenate([images, np.repeat(images[-1:], pad_size, axis=0)], axis=0)
        else:
            pad_size = 0

        images = jax.device_put(jnp.array(images), shard)

        recon = reconstruct_fn(images)

        recon_np = np.asarray(jax.device_get(recon))
        if pad_size > 0:
            recon_np = recon_np[:-pad_size]

        recon_uint8 = fp_to_uint8(recon_np)

        if cfg.save_npz:
            all_samples.append(recon_uint8)

        if cfg.save_png:
            for i, img in enumerate(recon_uint8):
                img_idx = total_collected + i
                img_path = output_dir / "images" / f"sample_{img_idx:06d}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(img_path)

        total_collected += batch_len - pad_size if pad_size > 0 else batch_len
        pbar.update(batch_len - pad_size if pad_size > 0 else batch_len)

    pbar.close()

    if cfg.save_npz:
        all_samples = np.concatenate(all_samples, axis=0)
        npz_path = output_dir / "samples.npz"
        np.savez(npz_path, arr_0=all_samples)
        logging.info(f"Saved {all_samples.shape[0]} samples to {npz_path}")
        logging.info(f"Sample shape: {all_samples.shape}, dtype: {all_samples.dtype}")

    if cfg.save_png:
        logging.info(f"Saved individual PNGs to {output_dir / 'images'}")

    config_path = output_dir / "reconstruction_config.txt"
    with open(config_path, "w") as f:
        f.write(f"reference_path: {reference_path}\n")
        f.write(f"vqdino_path: {cfg.vqdino_path}\n")
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"total_reconstructed: {total_collected}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"image_size: {image_h}x{image_w}\n")
        f.write(f"num_latents: {vqdino_cfg.num_latents}\n")
        f.write(f"levels: {vqdino_cfg.levels}\n")
        f.write(f"codebook_size: {vqdino_cfg.codebook_size}\n")
        f.write(f"num_output_patches: {num_output_patches}\n")

    logging.info(f"Config saved to {config_path}")

    if cfg.save_npz:
        logging.info("")
        logging.info("To compute rFID:")
        logging.info(f"  CUDA: uv run --no-project --python=3.12 --with='tensorflow[and-cuda],scipy,requests,tqdm' flatdino/metrics/adm.py {npz_path} --ref-batch {reference_path}")
        logging.info(f"  TPU:  uv run --no-project --python=3.12 --with='tensorflow,scipy,requests,tqdm' flatdino/metrics/adm.py {npz_path} --ref-batch {reference_path}")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    cfg = tyro.cli(Config)
    main(cfg)
