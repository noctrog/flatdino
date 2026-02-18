from typing import Iterable, Callable

import jax
import jax.numpy as jnp
import numpy as np
import grain.python as grain
from scipy.linalg import sqrtm
from tqdm import tqdm
import albumentations as A
import cv2
from absl import logging

from flatdino.pretrained.inception import InceptionV3
from flatdino.data import DataConfig, create_dataloaders, CACHE_FOLDER
from flatdino.data.utils import hash_dataclass
from flatdino.distributed import prefetch_to_mesh

FID_STAT_DATASET_CACHE = CACHE_FOLDER / "fid"


class FIDRunningStats:
    """Accumulates mean and covariance without storing all features."""

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None
        self._sum_outer: np.ndarray | None = None
        self.count: int = 0

    def update(self, features: np.ndarray) -> None:
        if features.size == 0:
            return
        if features.ndim != 2:
            raise ValueError("Expected 2D array for features.")

        if self._sum is None:
            dim = features.shape[1]
            self._sum = np.zeros(dim, dtype=np.float64)
            self._sum_outer = np.zeros((dim, dim), dtype=np.float64)

        assert self._sum is not None
        assert self._sum_outer is not None

        self._sum += np.sum(features, axis=0, dtype=np.float64)
        self._sum_outer += features.T @ features
        self.count += features.shape[0]

    def finalize(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("No samples processed; cannot compute statistics.")
        if self._sum is None or self._sum_outer is None:
            raise RuntimeError("Running statistics are uninitialized.")

        mu = self._sum / self.count
        if self.count < 2:
            raise RuntimeError("At least two samples required to compute covariance.")

        cov = (self._sum_outer - np.outer(mu, mu) * self.count) / (self.count - 1)
        cov = (cov + cov.T) * 0.5
        return {"mu": mu, "cov": cov}


class InceptionAugmentation(grain.MapTransform):
    def __init__(self):
        super().__init__()
        self.transform = A.Compose(
            [
                A.Resize(299, 299, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transform(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}


# TODO: remove cache name
def fid_for_iterable(
    image_iter: Iterable[jnp.ndarray],
    feat_fn: Callable[[jnp.ndarray], jnp.ndarray],
    num_eval_images: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Calculate the FID statistics for a batched iterable of images.

    Args:
      image_iter (Iterable): data loader.
      feat_fn (Callable): the feature extractor function
      num_eval_images (int, optional): the total number of images to evaluate
      verbose (bool)
    """

    def _valid_size(batch) -> int:
        if isinstance(batch, dict):
            if "_valid_size" in batch:
                val = batch["_valid_size"]
                try:
                    return int(jax.device_get(val))
                except Exception:
                    return int(val)
            img = batch.get("image")
            if img is not None and hasattr(img, "shape") and len(img.shape) > 0:
                return int(img.shape[0])
        leaves = jax.tree.leaves(batch)
        if leaves and hasattr(leaves[0], "shape") and leaves[0].shape:
            return int(leaves[0].shape[0])
        raise ValueError("Cannot infer batch size for FID computation.")

    # TODO: remove magic numbers
    running_mu = np.zeros(2048, dtype=np.float64)
    running_cov = np.zeros((2048, 2048), dtype=np.float64)

    total_images = 0
    for i, batch in enumerate(tqdm(image_iter, desc="Calculating statistics", disable=not verbose)):
        # TODO: check the data normalization for the inception input
        features = feat_fn(batch)
        valid = _valid_size(batch)

        if num_eval_images is not None and total_images >= num_eval_images:
            break

        if num_eval_images is not None:
            valid = min(valid, num_eval_images - total_images)
        features = features[:valid]
        total_images += features.shape[0]

        features = np.asarray(jax.device_get(features), dtype=np.float64)

        running_mu = running_mu + np.sum(features, axis=0)
        running_cov = running_cov + np.matmul(features.T, features, dtype=np.float64)

    if verbose:
        print(f"FID total number of images: {total_images}")

    mu = running_mu / total_images
    cov = (running_cov - np.outer(mu, mu) * total_images) / (total_images - 1)
    stats = {"mu": mu, "cov": cov}

    return stats


def fid_dataset(
    data_cfg: DataConfig,
    batch_size: int,
    feat_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    aug: grain.MapTransform = InceptionAugmentation(),
    num_eval_images: int | None = None,
    verbose: bool = False,
    *,
    mesh: jax.sharding.Mesh,
    feat_fn_factory: Callable[[], Callable[[jnp.ndarray], jnp.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    dataset_hash = hash_dataclass(data_cfg)
    cache_file = FID_STAT_DATASET_CACHE / f"{dataset_hash}.npz"
    if cache_file.exists():
        logging.info(f"FID: found cached stats for dataset {data_cfg.dataset}")
        with np.load(cache_file.absolute()) as f:
            stats = {"mu": f["mu"], "cov": f["cov"]}
        return stats

    if feat_fn is None and feat_fn_factory is None:
        raise ValueError("Provide either feat_fn or feat_fn_factory.")

    resolved_feat_fn = feat_fn or feat_fn_factory()

    logging.info(f"FID: computing statistics for dataset {data_cfg.dataset}")
    loaders = create_dataloaders(
        data_cfg, batch_size, train_epochs=1, val_epochs=1, val_aug=aug, drop_remainder_val=False
    )
    val_iter = prefetch_to_mesh(
        iter(loaders.val_loader),
        1,
        mesh=mesh,
        pad_to=mesh.size,
    )
    stats = fid_for_iterable(
        val_iter, resolved_feat_fn, num_eval_images=num_eval_images, verbose=verbose
    )

    cache_file.parent.mkdir(exist_ok=True)
    np.savez(cache_file, mu=stats["mu"], cov=stats["cov"])
    return stats


def create_inception_forward() -> Callable[[jnp.ndarray], jnp.ndarray]:
    model = InceptionV3(pretrained=True)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))

    def forward(x: jax.Array) -> jax.Array:
        return model.apply(params, x, train=False).squeeze(axis=(1, 2))

    return forward


def frechet_distance(
    stats_real: dict[str, np.ndarray],
    stats_fake: dict[str, np.ndarray],
) -> float:
    mu_real, mu_fake = stats_real["mu"], stats_fake["mu"]
    cov_real, cov_fake = stats_real["cov"], stats_fake["cov"]

    eps = 1e-6
    eye = np.eye(cov_real.shape[0], dtype=np.float64)
    cov_real = cov_real + eps * eye
    cov_fake = cov_fake + eps * eye

    covmean = sqrtm(cov_real @ cov_fake)
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0.0, atol=1e-6):
            raise RuntimeError("Matrix square root produced significant imaginary components.")
        covmean = covmean.real

    diff = mu_real - mu_fake
    fid_value = np.dot(diff, diff) + np.trace(cov_real + cov_fake - 2.0 * covmean)
    return float(fid_value)


if __name__ == "__main__":
    import flax.nnx as nnx
    from flatdino.pretrained.inception import InceptionV3

    mesh = jax.make_mesh((jax.local_device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    cfg = DataConfig()
    aug = InceptionAugmentation()

    def forward_factory():
        inception_forward = create_inception_forward()

        @nnx.jit
        def forward(batch) -> jax.Array:
            return inception_forward(batch["image"])

        return forward

    fid_dataset(cfg, 1024, feat_fn_factory=forward_factory, verbose=True, mesh=mesh)
