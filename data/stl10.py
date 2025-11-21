"""
STL-10 utilities. The main codebase is written to be easily extended;
these helpers mirror the CIFAR-10 ones but default to the unlabeled split
for unsupervised pretraining.
"""
from typing import Tuple

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10

from .augmentations import get_classification_transform, get_simclr_transform


def get_stl10_simclr_dataloader(
    data_dir: str,
    batch_size: int,
    augment: str = "full",
    num_workers: int = 4,
) -> DataLoader:
    transform = get_simclr_transform(augment, image_size=96)
    dataset = STL10(root=data_dir, split="unlabeled", transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_stl10_classification_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    label_fraction: float = 1.0,
    seed: int = 42,
    train_transform=None,
    test_transform=None,
) -> Tuple[DataLoader, DataLoader]:
    train_transform = train_transform or get_classification_transform(train=True, image_size=96)
    test_transform = test_transform or get_classification_transform(train=False, image_size=96)
    train_dataset = STL10(root=data_dir, split="train", transform=train_transform, download=True)
    test_dataset = STL10(root=data_dir, split="test", transform=test_transform, download=True)

    if label_fraction < 1.0:
        from .cifar10 import _fraction_subset

        train_dataset = _fraction_subset(train_dataset, label_fraction, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
