from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from .augmentations import get_classification_transform, get_simclr_transform


def _fraction_subset(dataset, fraction: float, seed: int) -> Subset:
    if fraction >= 1.0:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    subset_length = max(1, int(len(indices) * fraction))
    return Subset(dataset, indices[:subset_length])


def get_cifar10_simclr_dataloader(
    data_dir: str,
    batch_size: int,
    augment: str = "full",
    num_workers: int = 4,
) -> DataLoader:
    """
    Build the CIFAR-10 dataloader for SimCLR pretraining.
    """
    transform = get_simclr_transform(augment, image_size=32)
    dataset = CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def get_cifar10_classification_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    label_fraction: float = 1.0,
    seed: int = 42,
    train_transform=None,
    test_transform=None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 loaders for supervised, linear eval, or fine-tune.
    """
    train_transform = train_transform or get_classification_transform(train=True, image_size=32)
    test_transform = test_transform or get_classification_transform(train=False, image_size=32)

    train_dataset = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    train_dataset = _fraction_subset(train_dataset, label_fraction, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader

