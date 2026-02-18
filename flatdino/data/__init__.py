from pathlib import Path
from flatdino.data.data import (
    DataConfig as DataConfig,
    DataLoaders as DataLoaders,
    create_dataloaders as create_dataloaders,
    DATASET_PRESETS as DATASET_PRESETS,
    IMAGENET_DEFAULT_MEAN as IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD as IMAGENET_DEFAULT_STD,
    get_interpolation as get_interpolation,
)

CACHE_FOLDER: Path = Path.home() / ".cache" / "flatdino"
HASH_FOLDER = CACHE_FOLDER / "hashes"
MODELS_CACHE_ROOT = CACHE_FOLDER / "pretrained_models"
