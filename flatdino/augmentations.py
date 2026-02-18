from dataclasses import dataclass

import albumentations as A
import grain.python as grain
import numpy as np

from flatdino.data import DataConfig, get_interpolation
from flatdino.decoder.augmentations import ADMCenterCrop


@dataclass
class FlatDinoAugConfig:
    image_size: tuple[int, int] = (256, 256)
    crop_scale: tuple[float, float] = (0.3, 1.0)
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    horizontal_flip: bool = True
    """Whether to apply random horizontal flip (for generator training)."""


class FlatDinoTrainAugmentations(grain.MapTransform):
    def __init__(self, cfg: FlatDinoAugConfig, data_cfg: DataConfig):
        super().__init__()
        interp = get_interpolation(data_cfg.source_resolution, cfg.image_size[0])
        transforms = [
            A.RandomResizedCrop(cfg.image_size, cfg.crop_scale, interpolation=interp),
            A.HorizontalFlip(),
            A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std),
        ]
        self.transforms = A.Compose(transforms)

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}


class FlatDinoValAugmentations(grain.MapTransform):
    def __init__(self, cfg: FlatDinoAugConfig, data_cfg: DataConfig):
        super().__init__()
        interp = get_interpolation(data_cfg.source_resolution, cfg.image_size[0])
        self.transforms = A.Compose(
            [
                A.Resize(*cfg.image_size, interpolation=interp),
                A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std),
            ]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}


class FlatDinoGeneratorTrainAugmentations(grain.MapTransform):
    """Training augmentations for diffusion generator.

    Uses ADM-style center crop (no random cropping) to match the distribution
    of the ImageNet reference batch used for FID evaluation. This is critical
    for achieving good FID scores.

    Optionally applies horizontal flip (controlled by cfg.horizontal_flip).
    """

    def __init__(self, cfg: FlatDinoAugConfig, data_cfg: DataConfig):
        super().__init__()
        transforms = [ADMCenterCrop(cfg.image_size[0])]
        if cfg.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std))
        self.transforms = A.Compose(transforms)

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}


class FlatDinoGeneratorValAugmentations(grain.MapTransform):
    """Validation augmentations for FID evaluation (gFID/rFID).

    Uses ADM-style center crop without horizontal flip to match the distribution
    of the ImageNet reference batch used for FID evaluation exactly.
    """

    def __init__(self, cfg: FlatDinoAugConfig, data_cfg: DataConfig):
        super().__init__()
        self.transforms = A.Compose([
            ADMCenterCrop(cfg.image_size[0]),
            A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std),
        ])

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}
