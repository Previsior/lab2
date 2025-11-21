import random
from typing import Callable, Tuple

from torchvision import transforms


# Standard CIFAR-10 statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_normalization(stats: Tuple[Tuple[float, ...], Tuple[float, ...]] = (CIFAR10_MEAN, CIFAR10_STD)) -> transforms.Normalize:
    """Return a normalization transform for given dataset statistics."""
    mean, std = stats
    return transforms.Normalize(mean=mean, std=std)


class SimCLRAugmentation:
    """
    Apply the same base augmentation twice to create a positive pair.
    """

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def _base_color_jitter() -> transforms.ColorJitter:
    return transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)


def get_simclr_transform(
    variant: str = "full",
    image_size: int = 32,
    stats: Tuple[Tuple[float, ...], Tuple[float, ...]] = (CIFAR10_MEAN, CIFAR10_STD),
) -> Callable:
    """
    Build the SimCLR augmentation pipeline.

    Variants:
        only_crop: RandomResizedCrop + RandomHorizontalFlip
        crop_color: only_crop + ColorJitter
        crop_color_gray: crop_color + RandomGrayscale
        full: crop_color_gray + GaussianBlur
    """
    normalize = get_normalization(stats)
    variant = variant.lower()

    aug_blocks = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if variant in {"crop_color", "crop_color_gray", "full"}:
        aug_blocks.append(_base_color_jitter())

    if variant in {"crop_color_gray", "full"}:
        aug_blocks.append(transforms.RandomGrayscale(p=0.2))

    if variant == "full":
        aug_blocks.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5))

    aug_blocks.extend(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    return SimCLRAugmentation(transforms.Compose(aug_blocks))


def get_classification_transform(
    train: bool = True,
    image_size: int = 32,
    stats: Tuple[Tuple[float, ...], Tuple[float, ...]] = (CIFAR10_MEAN, CIFAR10_STD),
) -> transforms.Compose:
    """Standard classification augmentations for CIFAR-like data."""
    normalize = get_normalization(stats)
    if train:
        ops = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        ops = [
            transforms.ToTensor(),
            normalize,
        ]
    return transforms.Compose(ops)
