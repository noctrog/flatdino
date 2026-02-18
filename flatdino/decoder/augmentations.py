"""Decoder augmentation utilities.

This module provides augmentations commonly used for RAE and diffusion model training,
including ADM-style center cropping which is the standard preprocessing for
ImageNet evaluation (FID/sFID/IS).
"""

from dataclasses import dataclass

import albumentations as A
import cv2
import grain.python as grain
import numpy as np
from PIL import Image

from flatdino.data import DataConfig, get_interpolation


# --- ADM center crop ---

def adm_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Center crop implementation from ADM (guided-diffusion).

    This is the standard preprocessing used by ADM, DiT, DDT and other
    diffusion models for ImageNet evaluation.

    Args:
        image: Input image as numpy array (H, W, C).
        size: Target size for the square crop.

    Returns:
        Center-cropped image as numpy array (size, size, C).
    """
    pil_image = Image.fromarray(image)

    # Downsample by 2 while image is >= 2x target size (faster than single large resize)
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    # Resize so shortest side equals target size
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # Center crop to target size
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - size) // 2
    crop_x = (arr.shape[1] - size) // 2
    return arr[crop_y : crop_y + size, crop_x : crop_x + size]


class ADMCenterCrop(A.ImageOnlyTransform):
    """Albumentations wrapper for ADM-style center crop.

    This transform applies the standard ADM center crop preprocessing,
    which iteratively downsamples large images before the final resize
    for better quality.

    Args:
        size: Target size for the square crop.
        always_apply: Whether to always apply the transform.
        p: Probability of applying the transform.
    """

    def __init__(self, size: int, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.size = size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return adm_center_crop(img, self.size)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("size",)


@dataclass
class ADMCenterCropConfig:
    """Configuration for ADM center crop augmentations."""

    crop_size: int = 256
    """Target size for the ADM center crop."""
    output_size: int | None = None
    """If specified, resize to this size after cropping (e.g., 224 for DINO)."""
    horizontal_flip: bool = False
    """Whether to apply random horizontal flip."""


class ADMCenterCropAugmentations(grain.MapTransform):
    """ADM-style center crop augmentation for grain dataloaders.

    This augmentation applies:
    1. ADM center crop to crop_size (default 256)
    2. Optional resize to output_size (e.g., 224 for DINO)
    3. Optional horizontal flip
    4. ImageNet normalization

    Args:
        cfg: Configuration for the augmentation.
        data_cfg: Data configuration with normalization parameters.
    """

    def __init__(self, cfg: ADMCenterCropConfig, data_cfg: DataConfig):
        super().__init__()
        transforms = [ADMCenterCrop(cfg.crop_size)]

        if cfg.output_size is not None and cfg.output_size != cfg.crop_size:
            transforms.append(A.Resize(cfg.output_size, cfg.output_size, interpolation=cv2.INTER_LINEAR))

        if cfg.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))

        transforms.append(A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std))
        self.transforms = A.Compose(transforms)

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}


# --- RAE decoder augmentations ---

@dataclass
class RAEDecoderAugConfig:
    resize: tuple[int, int] = (384, 384)
    crop_size: tuple[int, int] = (256, 256)


class RAEDecoderTrainAugmentations(grain.MapTransform):
    def __init__(self, cfg: RAEDecoderAugConfig, data_cfg: DataConfig):
        super().__init__()
        transforms = [
            A.Resize(*cfg.resize, interpolation=cv2.INTER_AREA),
            A.RandomCrop(*cfg.crop_size, pad_if_needed=True, pad_position="random"),
            # A.HorizontalFlip(),
            A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std),
        ]
        self.transforms = A.Compose(transforms)

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return {"image": element["image"], "label": element["label"]}


class RAEDecoderValAugmentations(grain.MapTransform):
    def __init__(self, cfg: RAEDecoderAugConfig, data_cfg: DataConfig):
        super().__init__()
        self.crop_size = cfg.crop_size[0]
        self.normalize = A.Normalize(data_cfg.normalization_mean, data_cfg.normalization_std)

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Use ADM center crop (handles large images with iterative BOX downsampling)
        image = adm_center_crop(element["image"], self.crop_size)
        # Normalize
        image = self.normalize(image=image)["image"]
        return {"image": image, "label": element["label"]}


# --- DINO validation augmentations ---

class DINOValAugmentations(grain.MapTransform):
    def __init__(self, cfg: DataConfig, resolution: int = 224):
        interp = get_interpolation(cfg.source_resolution, resolution)
        self.transforms = A.Compose(
            [
                A.Resize(resolution, resolution, interpolation=interp),
                A.Normalize(cfg.normalization_mean, cfg.normalization_std),
            ]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return {key: element[key] for key in ["image", "label"]}


__all__ = [
    "adm_center_crop",
    "ADMCenterCrop",
    "ADMCenterCropConfig",
    "ADMCenterCropAugmentations",
    "RAEDecoderAugConfig",
    "RAEDecoderTrainAugmentations",
    "RAEDecoderValAugmentations",
    "DINOValAugmentations",
]
