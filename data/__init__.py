from typing import Tuple

from torch.utils.data import DataLoader

from .cifar10 import get_cifar10_classification_dataloaders, get_cifar10_simclr_dataloader
from .stl10 import get_stl10_classification_dataloaders, get_stl10_simclr_dataloader


def get_simclr_dataloader(
    dataset: str,
    data_dir: str,
    batch_size: int,
    augment: str = "full",
    num_workers: int = 4,
) -> DataLoader:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return get_cifar10_simclr_dataloader(data_dir, batch_size, augment=augment, num_workers=num_workers)
    if dataset == "stl10":
        return get_stl10_simclr_dataloader(data_dir, batch_size, augment=augment, num_workers=num_workers)
    raise ValueError(f"Unsupported dataset for SimCLR: {dataset}")


def get_classification_dataloaders(
    dataset: str,
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
    dataset = dataset.lower()
    if dataset == "cifar10":
        return get_cifar10_classification_dataloaders(
            data_dir,
            batch_size,
            num_workers=num_workers,
            label_fraction=label_fraction,
            seed=seed,
            train_transform=train_transform,
            test_transform=test_transform,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    if dataset == "stl10":
        return get_stl10_classification_dataloaders(
            data_dir,
            batch_size,
            num_workers=num_workers,
            label_fraction=label_fraction,
            seed=seed,
            train_transform=train_transform,
            test_transform=test_transform,
        )
    raise ValueError(f"Unsupported dataset for classification: {dataset}")

