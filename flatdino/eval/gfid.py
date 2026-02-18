from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import jmp
import numpy as np
from PIL import Image
import tyro
from dacite import Config as DaciteConfig, from_dict
from tqdm import tqdm
from absl import logging

from flatdino.models.vit import ViTEncoder
from flatdino.data import DataConfig
import flatdino.metrics.gfid as gfid_eval
from flatdino.eval import restore_encoder, save_eval_results
from flatdino.decoder.sampling import (
    _infer_time_dist_shift,
    _restore_generator,
    _sample_latents,
)
from flatdino.pretrained.rae_decoder import make_rae_decoder


@dataclass
class Config:
    generator_path: Path
    """Directory containing checkpoints produced by train_generator.py or train_rae_dit.py."""

    flatdino_path: Path
    """Directory containing the FlatDINO autoencoder checkpoint (required for FlatDINO latents)."""

    steps: int = 50
    """Number of flow-matching integration steps."""

    per_class: int = 50
    """Number of samples to draw per class (50 -> 50_000 for ImageNet)."""

    batch_size: int = 200
    """Batch size used for sampling/generation and FID feature extraction."""

    seed: int = 0
    use_ema: bool = True
    cfg_scale: float | None = None
    cfg_interval_min: float = 0.0
    """Minimum timestep for applying CFG (DDT default: 0.3)."""
    cfg_interval_max: float = 1.0
    """Maximum timestep for applying CFG."""
    verbose: bool = False

    save_samples: bool = False
    """Save generated samples to disk organized by class."""

    samples_per_class: int = 10
    """Number of samples to save per class (only used when save_samples=True)."""

    save_grid: bool = False
    """Save a grid of generated samples (one row per class)."""

    grid_classes: int = 10
    """Number of classes to include in the grid."""

    grid_samples_per_class: int = 10
    """Number of samples per class in the grid (columns)."""


def prepare_data_config(generator_cfg: dict, restored_data: DataConfig | None) -> DataConfig:
    if restored_data is not None:
        return restored_data
    data_cfg_blob = generator_cfg.get("data", None)
    if data_cfg_blob is None:
        return DataConfig()
    dacite_cfg = DaciteConfig(cast=[tuple])
    return from_dict(DataConfig, data_cfg_blob, config=dacite_cfg)  # type: ignore[arg-type]


def _maybe_asdict(cfg) -> dict:
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    return cfg


@nnx.jit(static_argnames=("patch_tokens", "image_h", "image_w"))
def decode_latents(
    flatdino_decoder, rae_decoder, decoder_input, std, mean, patch_tokens, image_h, image_w
):
    decoder_tokens = flatdino_decoder(decoder_input, deterministic=True)[:, :patch_tokens]
    decoder_out = rae_decoder(decoder_tokens)
    recon = rae_decoder.unpatchify(
        decoder_out.logits,
        original_image_size=(image_h, image_w),
    )
    recon = jnp.transpose(recon, (0, 2, 3, 1))
    return jnp.clip(recon * std + mean, 0.0, 1.0)


def generate_batch(
    labels: jax.Array,
    sample_key: jax.Array,
    generator: nnx.Module,
    flatdino_decoder: ViTEncoder,
    rae_decoder,
    mean: jax.Array,
    std: jax.Array,
    image_h: int,
    image_w: int,
    steps: int,
    latent_shape,
    cfg_scale: float,
    cfg_interval_min: float,
    cfg_interval_max: float,
    time_dist_shift: float,
    pred_type: str,
) -> jax.Array:
    latents = _sample_latents(
        generator,
        labels=labels,
        steps=steps,
        key=sample_key,
        latent_shape=latent_shape,
        cfg_scale=cfg_scale,
        cfg_interval_min=cfg_interval_min,
        cfg_interval_max=cfg_interval_max,
        time_dist_shift=time_dist_shift,
        pred_type=pred_type,
    )
    if isinstance(latents, dict):
        latents = latents["x"]

    # Denormalize latents if normalization was used during training
    latents = generator.denormalize(latents)

    patch_tokens = flatdino_decoder.num_reg
    return decode_latents(
        flatdino_decoder, rae_decoder, latents, std, mean, patch_tokens, image_h, image_w
    )


def compute_gfid(
    generator,
    generator_cfg,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    steps: int,
    per_class: int,
    batch_size: int,
    seed: int,
    cfg_scale: float | None = None,
    cfg_interval_min: float = 0.0,
    cfg_interval_max: float = 1.0,
    verbose: bool = False,
    restored,
    num_grid_images: int = 0,
    time_dist_shift: float | None = None,
) -> tuple[float, np.ndarray | None]:
    """Compute gFID for a trained generator.

    Returns:
        fid_value: scalar gFID.
        sample_images: optional (N, H, W, C) array with denormalized samples if
            num_grid_images > 0.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if per_class <= 0:
        raise ValueError("per_class must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if restored is None:
        raise ValueError("FlatDINO checkpoint must be provided for gFID computation.")

    generator_cfg = _maybe_asdict(generator_cfg)

    if batch_size % mesh.size != 0:
        raise ValueError("batch_size must be divisible by the number of devices in the mesh.")

    latent_tokens: int | None = None
    patch_tokens: int | None = None
    image_h: int | None = None
    image_w: int | None = None
    mean_vals = std_vals = None

    assert restored.encoder is not None, "FlatDINO checkpoint does not include encoder weights."
    assert restored.decoder is not None, "FlatDINO checkpoint does not include decoder weights."
    if restored.data_cfg is None or restored.aug_cfg is None:
        raise RuntimeError("FlatDINO checkpoint is missing data or augmentation configuration.")

    latent_tokens = restored.encoder.num_reg
    if latent_tokens is None or latent_tokens <= 0:
        raise ValueError("Encoder configuration does not define a positive number of registers.")

    patch_tokens = restored.decoder.num_reg
    if patch_tokens is None or patch_tokens <= 0:
        raise ValueError("Decoder configuration does not define a positive number of patch tokens.")

    image_h, image_w = restored.aug_cfg.image_size
    mean_vals = restored.data_cfg.normalization_mean
    std_vals = restored.data_cfg.normalization_std

    model_type = generator_cfg.get("model_type", "dit")
    latent_dim = generator.cfg.in_channels
    dit_dh_cfg = generator_cfg.get("dit_dh", {}) if isinstance(generator_cfg, dict) else {}
    patch_size = dit_dh_cfg.get("patch_size", None) if isinstance(dit_dh_cfg, dict) else None
    is_flat_dit_dh = model_type == "dit_dh" and patch_size is None
    if model_type == "dit_dh" and not is_flat_dit_dh:
        grid = int(jnp.sqrt(latent_tokens))
        if grid * grid != latent_tokens:
            raise ValueError(f"Latent tokens ({latent_tokens}) are not a perfect square for DiTDH.")
        latent_shape = (grid, grid, latent_dim)
    else:
        latent_shape = (latent_tokens, latent_dim)

    if latent_tokens is None or latent_tokens <= 0:
        raise ValueError("Unable to infer a positive number of latent tokens.")
    if patch_tokens is None or patch_tokens <= 0:
        raise ValueError("Unable to infer a positive number of patch tokens.")
    if image_h is None or image_w is None:
        raise ValueError("Unable to infer image size for reconstruction.")
    if image_h != image_w:
        raise ValueError("Only square image sizes are supported.")
    if mean_vals is None or std_vals is None:
        raise ValueError("Normalization statistics are missing; cannot denormalize outputs.")

    if generator.cfg.num_classes != 1000:
        raise ValueError(
            f"Expected a 1000-class generator for ImageNet, got {generator.cfg.num_classes}."
        )

    rae_decoder = make_rae_decoder(
        num_patches=patch_tokens,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=seed,
    )

    mean = jnp.array(mean_vals, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(std_vals, dtype=jnp.float32)[None, None, None, :]

    if time_dist_shift is None:
        time_dist_shift = _infer_time_dist_shift(
            generator_cfg,
            latent_tokens=latent_tokens,
            latent_dim=latent_dim,
        )

    shard = NamedSharding(mesh, P("data"))

    data_cfg = prepare_data_config(generator_cfg, restored.data_cfg if restored else None)

    generate_fn = partial(
        generate_batch,
        generator=generator,
        flatdino_decoder=restored.decoder,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        image_h=image_h,
        image_w=image_w,
        steps=steps,
        latent_shape=latent_shape,
        cfg_scale=cfg_scale,
        cfg_interval_min=cfg_interval_min,
        cfg_interval_max=cfg_interval_max,
        time_dist_shift=time_dist_shift,
        pred_type=generator_cfg["pred_type"] if "pred_type" in generator_cfg else "v",
    )

    fid_value, grid = gfid_eval.compute_gfid(
        generate_fn,
        num_classes=generator.cfg.num_classes,
        per_class=per_class,
        batch_size=batch_size,
        mesh=mesh,
        seed=seed,
        data_cfg=data_cfg,
        label_sharding=shard,
        num_grid_images=num_grid_images,
        verbose=verbose,
    )

    return fid_value, grid


def save_samples(
    generator,
    generator_cfg,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    steps: int,
    samples_per_class: int,
    batch_size: int,
    seed: int,
    cfg_scale: float | None,
    cfg_interval_min: float = 0.0,
    cfg_interval_max: float = 1.0,
    restored,
    output_dir: Path,
    time_dist_shift: float | None = None,
) -> None:
    """Generate and save sample images organized by class.

    Saves images to: output_dir/cfg{scale}/class_{id:04d}/sample_{i:04d}.png
    """
    if restored is None:
        raise ValueError("FlatDINO checkpoint must be provided.")

    generator_cfg = _maybe_asdict(generator_cfg)

    assert restored.encoder is not None, "FlatDINO checkpoint does not include encoder weights."
    assert restored.decoder is not None, "FlatDINO checkpoint does not include decoder weights."
    if restored.data_cfg is None or restored.aug_cfg is None:
        raise RuntimeError("FlatDINO checkpoint is missing data or augmentation configuration.")

    latent_tokens = restored.encoder.num_reg
    patch_tokens = restored.decoder.num_reg
    image_h, image_w = restored.aug_cfg.image_size
    mean_vals = restored.data_cfg.normalization_mean
    std_vals = restored.data_cfg.normalization_std

    model_type = generator_cfg.get("model_type", "dit")
    latent_dim = generator.cfg.in_channels
    dit_dh_cfg = generator_cfg.get("dit_dh", {}) if isinstance(generator_cfg, dict) else {}
    patch_size = dit_dh_cfg.get("patch_size", None) if isinstance(dit_dh_cfg, dict) else None
    is_flat_dit_dh = model_type == "dit_dh" and patch_size is None
    if model_type == "dit_dh" and not is_flat_dit_dh:
        grid = int(jnp.sqrt(latent_tokens))
        latent_shape = (grid, grid, latent_dim)
    else:
        latent_shape = (latent_tokens, latent_dim)

    rae_decoder = make_rae_decoder(
        num_patches=patch_tokens,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=seed,
    )

    mean = jnp.array(mean_vals, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(std_vals, dtype=jnp.float32)[None, None, None, :]

    if time_dist_shift is None:
        time_dist_shift = _infer_time_dist_shift(
            generator_cfg,
            latent_tokens=latent_tokens,
            latent_dim=latent_dim,
        )

    shard = NamedSharding(mesh, P("data"))

    generate_fn = partial(
        generate_batch,
        generator=generator,
        flatdino_decoder=restored.decoder,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        image_h=image_h,
        image_w=image_w,
        steps=steps,
        latent_shape=latent_shape,
        cfg_scale=cfg_scale,
        cfg_interval_min=cfg_interval_min,
        cfg_interval_max=cfg_interval_max,
        time_dist_shift=time_dist_shift,
        pred_type=generator_cfg["pred_type"] if "pred_type" in generator_cfg else "v",
    )

    # Create output directory structure
    cfg_str = f"cfg{cfg_scale}" if cfg_scale is not None else "cfg_none"
    samples_dir = output_dir / "samples" / cfg_str

    num_classes = generator.cfg.num_classes
    total_samples = num_classes * samples_per_class

    # Generate labels: [0,0,...,1,1,...,2,2,...] with samples_per_class of each
    labels = jnp.arange(num_classes, dtype=jnp.int32)
    labels = jnp.repeat(labels, samples_per_class)

    key = jax.random.PRNGKey(seed)

    # Track how many samples saved per class
    class_counts = {c: 0 for c in range(num_classes)}

    total_batches = (total_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc="Generating samples"):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_samples)
        valid = end - start

        label_batch = jax.lax.dynamic_slice(labels, (start,), (valid,))
        key, sample_key = jax.random.split(key)

        if valid < batch_size:
            pad = batch_size - valid
            label_batch = jnp.concatenate(
                [label_batch, jnp.repeat(label_batch[-1:], pad, axis=0)], axis=0
            )

        label_batch = jax.device_put(label_batch, shard)
        images = generate_fn(label_batch, sample_key)

        # Get labels and images as numpy
        images_np = np.asarray(jax.device_get(images))[:valid]
        labels_np = np.asarray(jax.device_get(label_batch))[:valid]

        # Save each image to its class folder
        for img, lbl in zip(images_np, labels_np):
            class_dir = samples_dir / f"class_{lbl:04d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            img_idx = class_counts[int(lbl)]
            img_path = class_dir / f"sample_{img_idx:04d}.png"

            # Convert to uint8 and save
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(img_path)

            class_counts[int(lbl)] += 1

    logging.info(f"Saved {total_samples} samples to {samples_dir}")


def save_grid(
    generator,
    generator_cfg,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    steps: int,
    num_classes: int,
    samples_per_class: int,
    batch_size: int,
    seed: int,
    cfg_scale: float | None,
    cfg_interval_min: float = 0.0,
    cfg_interval_max: float = 1.0,
    restored,
    output_dir: Path,
    time_dist_shift: float | None = None,
) -> None:
    """Generate and save a grid of sample images (one row per class).

    Saves grid to: output_dir/samples/cfg{scale}/grid.png
    """
    if restored is None:
        raise ValueError("FlatDINO checkpoint must be provided.")

    generator_cfg = _maybe_asdict(generator_cfg)

    assert restored.encoder is not None
    assert restored.decoder is not None
    if restored.data_cfg is None or restored.aug_cfg is None:
        raise RuntimeError("FlatDINO checkpoint is missing data or augmentation configuration.")

    latent_tokens = restored.encoder.num_reg
    patch_tokens = restored.decoder.num_reg
    image_h, image_w = restored.aug_cfg.image_size
    mean_vals = restored.data_cfg.normalization_mean
    std_vals = restored.data_cfg.normalization_std

    model_type = generator_cfg.get("model_type", "dit")
    latent_dim = generator.cfg.in_channels
    dit_dh_cfg = generator_cfg.get("dit_dh", {}) if isinstance(generator_cfg, dict) else {}
    patch_size = dit_dh_cfg.get("patch_size", None) if isinstance(dit_dh_cfg, dict) else None
    is_flat_dit_dh = model_type == "dit_dh" and patch_size is None
    if model_type == "dit_dh" and not is_flat_dit_dh:
        grid = int(jnp.sqrt(latent_tokens))
        latent_shape = (grid, grid, latent_dim)
    else:
        latent_shape = (latent_tokens, latent_dim)

    rae_decoder = make_rae_decoder(
        num_patches=patch_tokens,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=seed,
    )

    mean = jnp.array(mean_vals, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(std_vals, dtype=jnp.float32)[None, None, None, :]

    if time_dist_shift is None:
        time_dist_shift = _infer_time_dist_shift(
            generator_cfg,
            latent_tokens=latent_tokens,
            latent_dim=latent_dim,
        )

    shard = NamedSharding(mesh, P("data"))

    generate_fn = partial(
        generate_batch,
        generator=generator,
        flatdino_decoder=restored.decoder,
        rae_decoder=rae_decoder,
        mean=mean,
        std=std,
        image_h=image_h,
        image_w=image_w,
        steps=steps,
        latent_shape=latent_shape,
        cfg_scale=cfg_scale,
        cfg_interval_min=cfg_interval_min,
        cfg_interval_max=cfg_interval_max,
        time_dist_shift=time_dist_shift,
        pred_type=generator_cfg["pred_type"] if "pred_type" in generator_cfg else "v",
    )

    # Generate labels: for each class, generate samples_per_class samples
    # Use evenly spaced classes across the full range
    total_classes = generator.cfg.num_classes
    class_indices = np.linspace(0, total_classes - 1, num_classes, dtype=np.int32)
    labels = jnp.repeat(jnp.array(class_indices, dtype=jnp.int32), samples_per_class)
    total_samples = num_classes * samples_per_class

    key = jax.random.PRNGKey(seed)
    all_images = []

    total_batches = (total_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc="Generating grid"):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_samples)
        valid = end - start

        label_batch = jax.lax.dynamic_slice(labels, (start,), (valid,))
        key, sample_key = jax.random.split(key)

        if valid < batch_size:
            pad = batch_size - valid
            label_batch = jnp.concatenate(
                [label_batch, jnp.repeat(label_batch[-1:], pad, axis=0)], axis=0
            )

        label_batch = jax.device_put(label_batch, shard)
        images = generate_fn(label_batch, sample_key)
        images_np = np.asarray(jax.device_get(images))[:valid]
        all_images.append(images_np)

    all_images = np.concatenate(all_images, axis=0)

    # Reshape to (num_classes, samples_per_class, H, W, C) and create grid
    all_images = all_images.reshape(num_classes, samples_per_class, *all_images.shape[1:])
    # Create grid: (num_classes * H, samples_per_class * W, C)
    grid_rows = []
    for row in all_images:
        grid_rows.append(np.concatenate(row, axis=1))  # concat along width
    grid_image = np.concatenate(grid_rows, axis=0)  # concat along height

    # Save grid
    cfg_str = f"cfg{cfg_scale}" if cfg_scale is not None else "cfg_none"
    samples_dir = output_dir / "samples" / cfg_str
    samples_dir.mkdir(parents=True, exist_ok=True)
    grid_path = samples_dir / "grid.png"

    grid_uint8 = (np.clip(grid_image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(grid_uint8).save(grid_path)
    logging.info(f"Saved grid ({num_classes} classes x {samples_per_class} samples) to {grid_path}")


def main(cfg: Config):
    if cfg.steps <= 0:
        raise ValueError("steps must be positive.")
    if cfg.per_class <= 0:
        raise ValueError("per_class must be positive.")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.float32, output_dtype=jnp.float32)

    restored = restore_encoder(cfg.flatdino_path, mesh=mesh, mp=mp, encoder=True, decoder=True)
    if restored.encoder is None:
        raise RuntimeError("FlatDINO checkpoint does not include encoder weights.")
    latent_tokens: int | None = restored.encoder.num_reg

    generator, generator_cfg = _restore_generator(
        cfg.generator_path,
        mesh=mesh,
        mp=mp,
        latent_tokens=latent_tokens,
        use_ema=cfg.use_ema,
    )

    generator.eval()
    if restored is not None and restored.decoder is not None:
        restored.decoder.eval()

    fid_value, _ = compute_gfid(
        generator,
        generator_cfg,
        mesh=mesh,
        mp=mp,
        steps=cfg.steps,
        per_class=cfg.per_class,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        cfg_scale=cfg.cfg_scale,
        cfg_interval_min=cfg.cfg_interval_min,
        cfg_interval_max=cfg.cfg_interval_max,
        verbose=cfg.verbose,
        restored=restored,
    )

    logging.info(f"Generated images: {cfg.per_class * generator.cfg.num_classes}")
    logging.info(f"Generation FID (Inception V3): {fid_value:.4f}")

    # Build key name: include interval if not default
    key_parts = ["gfid"]
    if cfg.cfg_scale is not None:
        key_parts.append(f"cfg{cfg.cfg_scale}")
        if cfg.cfg_interval_min != 0.0 or cfg.cfg_interval_max != 1.0:
            key_parts.append(f"int{cfg.cfg_interval_min}-{cfg.cfg_interval_max}")
    key = "_".join(key_parts)

    results = {
        "gfid": fid_value,
        "cfg_scale": cfg.cfg_scale,
        "cfg_interval_min": cfg.cfg_interval_min,
        "cfg_interval_max": cfg.cfg_interval_max,
    }
    save_eval_results(cfg.generator_path, key, results)
    logging.info(f"Results saved to {cfg.generator_path / 'eval_results.json'}")

    if cfg.save_samples:
        save_samples(
            generator,
            generator_cfg,
            mesh=mesh,
            mp=mp,
            steps=cfg.steps,
            samples_per_class=cfg.samples_per_class,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            cfg_scale=cfg.cfg_scale,
            cfg_interval_min=cfg.cfg_interval_min,
            cfg_interval_max=cfg.cfg_interval_max,
            restored=restored,
            output_dir=cfg.generator_path,
        )

    if cfg.save_grid:
        save_grid(
            generator,
            generator_cfg,
            mesh=mesh,
            mp=mp,
            steps=cfg.steps,
            num_classes=cfg.grid_classes,
            samples_per_class=cfg.grid_samples_per_class,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            cfg_scale=cfg.cfg_scale,
            cfg_interval_min=cfg.cfg_interval_min,
            cfg_interval_max=cfg.cfg_interval_max,
            restored=restored,
            output_dir=cfg.generator_path,
        )


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
