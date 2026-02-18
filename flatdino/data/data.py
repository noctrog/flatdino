from dataclasses import dataclass
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"

import grain.python as grain
import tensorflow_datasets as tfds
import cv2

from flatdino.distributed import is_running_on_gcp

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_interpolation(source_resolution: int, target_resolution: int) -> int:
    """Select interpolation method based on whether we're upsampling or downsampling.

    Args:
        source_resolution: Typical source image resolution
        target_resolution: Target resolution after resize

    Returns:
        OpenCV interpolation flag (cv2.INTER_LINEAR for upsampling, cv2.INTER_AREA for downsampling)
    """
    if target_resolution > source_resolution:
        return cv2.INTER_LINEAR  # Upsampling: use bilinear
    else:
        return cv2.INTER_AREA  # Downsampling: use area averaging


# Dataset presets: (tfds_name, num_classes, train_split, val_split, source_resolution)
# source_resolution is the typical image size (used to select interpolation method)
DATASET_PRESETS: dict[str, tuple[str, int, str, str, int]] = {
    "imagenet": ("imagenet2012", 1000, "train", "validation", 256),
    "cifar10": ("cifar10", 10, "train", "test", 32),
    "cifar100": ("cifar100", 100, "train", "test", 32),
    "caltech101": ("caltech101", 102, "train", "test", 300),
    "flowers102": ("oxford_flowers102", 102, "train", "test", 512),
    "pets": ("oxford_iiit_pet", 37, "train", "test", 256),
    "food101": ("food101", 101, "train", "validation", 512),
    "dtd": ("dtd", 47, "train", "test", 640),
}


@dataclass
class DataConfig:
    num_workers: int = 8
    seed: int = 0

    normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD

    dataset: str = "imagenet2012"
    num_classes: int = 1000
    train_name: str = "train"
    val_name: str = "validation"
    source_resolution: int = 256
    """Typical source image resolution (used to select interpolation method)."""

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "DataConfig":
        """Create a DataConfig from a preset name.

        Args:
            name: Preset name (e.g., 'cifar10', 'flowers102', 'pets').
                  See DATASET_PRESETS for available presets.
            **kwargs: Override any DataConfig field.

        Returns:
            DataConfig with preset values, optionally overridden.
        """
        if name not in DATASET_PRESETS:
            available = ", ".join(sorted(DATASET_PRESETS.keys()))
            raise ValueError(f"Unknown dataset preset '{name}'. Available: {available}")

        tfds_name, num_classes, train_split, val_split, source_res = DATASET_PRESETS[name]
        return cls(
            dataset=tfds_name,
            num_classes=num_classes,
            train_name=train_split,
            val_name=val_split,
            source_resolution=source_res,
            **kwargs,
        )


@dataclass(frozen=True)
class DataLoaders:
    train_loader: grain.DataLoader
    val_loader: grain.DataLoader
    train_ds_size: int
    val_ds_size: int


def create_dataloaders(
    cfg: DataConfig,
    batch_size: int,
    train_epochs: int | None = None,
    val_epochs: int | None = None,
    train_aug: grain.MapTransform | None = None,
    val_aug: grain.MapTransform | None = None,
    drop_remainder_train: bool = True,
    drop_remainder_val: bool = False,
    val_shuffle: bool = False,
    shard_index: int | None = None,
    shard_count: int | None = None,
    gcs_bucket: str | None = None,
) -> DataLoaders:
    """Create dataset loaders.

    Args:
      cfg (DataConfig)
      batch_size (int): Per-host batch size (will be sharded across hosts if shard_count > 1)
      train_epochs (int | None): None means infinite epochs.
      val_epochs (int | None): None means infinite epochs.
      train_aug (grain.MapTransform | None)
      val_aug (grain.MapTransform | None)
      drop_remainder_train (bool): Drop incomplete final batches for training
      drop_remainder_val (bool): Drop incomplete final batches for validation
      val_shuffle (bool): Whether to shuffle validation data
      shard_index (int | None): Index of current host/process for multi-host sharding.
          If None, uses jax.process_index() when jax.process_count() > 1, else no sharding.
      shard_count (int | None): Total number of hosts/processes for multi-host sharding.
          If None, uses jax.process_count() when > 1, else no sharding.
      gcs_bucket (str | None): the Google Cloud Storage bucket name where the dataset is stored.
          It assumes the following structure: gs://<gcs_bucket>/<cfg.dataset>.
          If not specified, the default tfds data_dir will be used.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    data_dir = None
    if gcs_bucket is not None:
        if not is_running_on_gcp():
            raise RuntimeError(
                f"gcs_bucket='{gcs_bucket}' specified but not running on GCP. "
                "GCS dataset loading requires running on a GCP instance."
            )
        data_dir = f"gs://{gcs_bucket}"

    # Determine sharding options for multi-host training
    # Import jax here to avoid importing it at module level (affects JAX_PLATFORMS)
    import jax

    if shard_index is None and shard_count is None:
        # Auto-detect multi-host setup
        if jax.process_count() > 1:
            shard_index = jax.process_index()
            shard_count = jax.process_count()

    if shard_count is not None and shard_count > 1:
        shard_options = grain.ShardOptions(
            shard_index=shard_index,
            shard_count=shard_count,
            drop_remainder=True,
        )
    else:
        shard_options = grain.NoSharding()

    def create_split(
        dataset_name, split_name, aug, epochs, drop_remainder: bool, shuffle: bool
    ) -> tuple[grain.DataLoader, int]:
        dataset = tfds.data_source(dataset_name, split=split_name, data_dir=data_dir)  # ty: ignore
        operations = [grain.Batch(batch_size, drop_remainder=drop_remainder)]
        if aug is not None:
            operations.insert(0, aug)
        loader = grain.DataLoader(
            data_source=dataset,
            operations=operations,
            sampler=grain.IndexSampler(
                num_records=len(dataset),
                num_epochs=epochs,
                shard_options=shard_options,
                shuffle=shuffle,
                seed=0,
            ),
            worker_count=cfg.num_workers,
            read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=32),
        )
        return loader, len(dataset)

    t_ld, ts = create_split(
        cfg.dataset,
        cfg.train_name,
        train_aug,
        train_epochs,
        drop_remainder=drop_remainder_train,
        shuffle=True,
    )
    v_ld, vs = create_split(
        cfg.dataset,
        cfg.val_name,
        val_aug,
        val_epochs,
        drop_remainder=drop_remainder_val,
        shuffle=val_shuffle,
    )

    return DataLoaders(
        train_loader=t_ld,
        val_loader=v_ld,
        train_ds_size=ts,
        val_ds_size=vs,
    )
