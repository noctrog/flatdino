from typing import Callable

import flax.nnx as nnx
import jax
import jax.image
import jax.numpy as jnp
import numpy as np
from dacite import Config as DaciteConfig, from_dict
from jax.sharding import NamedSharding, PartitionSpec as P

from flatdino.data import DataConfig
import flatdino.metrics.fid as fid_eval
from flatdino.metrics.fid import FIDRunningStats
from flatdino.pretrained.inception import InceptionV3


def _labels_for_all_classes(num_classes: int, per_class: int) -> jax.Array:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if per_class <= 0:
        raise ValueError("per_class must be positive.")
    labels = jnp.arange(num_classes, dtype=jnp.int32)
    labels = jnp.repeat(labels, per_class, total_repeat_length=num_classes * per_class)
    return labels


def _prepare_data_config(data_cfg: DataConfig | dict | None) -> DataConfig:
    if data_cfg is None:
        return DataConfig()
    if isinstance(data_cfg, DataConfig):
        return data_cfg
    dacite_cfg = DaciteConfig(cast=[tuple])
    return from_dict(DataConfig, data_cfg, config=dacite_cfg)  # type: ignore[arg-type]


def _default_feature_fns() -> tuple[
    Callable[[jax.Array], jax.Array], Callable[[dict[str, jax.Array]], jax.Array]
]:
    if not hasattr(fid_eval, "InceptionV3"):
        fid_eval.InceptionV3 = InceptionV3  # type: ignore[attr-defined]

    inception_forward = fid_eval.create_inception_forward()

    @nnx.jit
    def fake_features(pixels: jax.Array) -> jax.Array:
        # Use bilinear with antialias for consistency with real image preprocessing
        # (real images use cv2.INTER_AREA which is similar to antialiased downscaling)
        resized = jax.image.resize(
            pixels, (pixels.shape[0], 299, 299, 3), method="bicubic", antialias=False
        )
        normalized = resized * 2.0 - 1.0
        return inception_forward(normalized)

    @nnx.jit
    def real_features(batch: dict[str, jax.Array]) -> jax.Array:
        return inception_forward(batch["image"])

    return fake_features, real_features


def compute_gfid(
    generate_fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    num_classes: int,
    per_class: int,
    batch_size: int,
    mesh: jax.sharding.Mesh,
    seed: int = 0,
    data_cfg: DataConfig | dict | None = None,
    real_stats: dict[str, np.ndarray] | None = None,
    feature_fn: Callable[[jax.Array], jax.Array] | None = None,
    real_feature_fn: Callable[[dict[str, jax.Array]], jax.Array] | None = None,
    label_sharding: NamedSharding | None = None,
    num_grid_images: int = 0,
    verbose: bool = False,
) -> tuple[float, np.ndarray | None]:
    """Compute gFID using a user-provided image generator.

    Args:
        generate_fn: callable(labels, rng) -> images in [0, 1], shape (B, H, W, C).
        num_classes: number of classes in the generator.
        per_class: samples per class.
        batch_size: per-step batch size; must be divisible by mesh.size.
        mesh: device mesh for sharding and dataset prefetching.
        seed: PRNG seed for generation.
        data_cfg: DataConfig or dict to build real-data stats when real_stats is None.
        real_stats: optional precomputed {"mu", "cov"} for the real data.
        feature_fn: optional images -> features function for generated samples.
        real_feature_fn: optional batch -> features function for real samples.
        label_sharding: optional sharding for the label tensor (defaults to data axis).
        num_grid_images: optionally return up to this many generated images.
        verbose: whether to log dataset FID progress.
    """
    assert batch_size % mesh.size == 0

    total_samples = per_class * num_classes
    label_sharding = label_sharding or NamedSharding(mesh, P("data"))

    default_feature_fn, default_real_feature_fn = _default_feature_fns()
    feature_fn = feature_fn or default_feature_fn
    real_feature_fn = real_feature_fn or default_real_feature_fn

    if real_stats is None:
        resolved_data_cfg = _prepare_data_config(data_cfg)
        real_stats = fid_eval.fid_dataset(
            resolved_data_cfg,
            batch_size,
            real_feature_fn,
            num_eval_images=total_samples,
            verbose=verbose,
            mesh=mesh,
        )

    labels = _labels_for_all_classes(num_classes, per_class)
    stats_fake = FIDRunningStats()

    key = jax.random.PRNGKey(seed)
    sample_images: list[np.ndarray] = []
    collected = 0

    total_batches = (total_samples + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
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

        label_batch = jax.device_put(label_batch, label_sharding)
        images = generate_fn(label_batch, sample_key)

        feats_fake = feature_fn(images)
        feats_fake_np = np.asarray(jax.device_get(feats_fake), dtype=np.float64)
        stats_fake.update(feats_fake_np[:valid])

        if num_grid_images > 0 and collected < num_grid_images:
            images_np = np.asarray(jax.device_get(images))
            take = min(images_np.shape[0], num_grid_images - collected, valid)
            sample_images.append(images_np[:take])
            collected += take

    fake_stats = stats_fake.finalize()
    fid_value = fid_eval.frechet_distance(real_stats, fake_stats)

    grid = None
    if num_grid_images > 0 and sample_images:
        grid = np.concatenate(sample_images, axis=0)[:num_grid_images]

    return fid_value, grid
