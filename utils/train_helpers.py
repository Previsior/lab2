import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_simclr_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for (x_i, x_j), _ in tqdm(loader, desc="SimCLR", leave=False):
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        z_i, _ = model(x_i, return_features=True)
        z_j, _ = model(x_j, return_features=True)

        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_autoencoder_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for images, _ in tqdm(loader, desc="AE", leave=False):
        images = images.to(device)
        recon, _ = model(images)
        loss = F.mse_loss(recon, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_classifier_epoch(
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    freeze_encoder: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[float, float]:
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if freeze_encoder:
            with torch.no_grad():
                feats = feature_extractor(images)
            feats = feats.detach()
        else:
            feats = feature_extractor(images)

        logits = classifier(feats)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_classifier(
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        feats = feature_extractor(images)
        logits = classifier(feats)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def build_feature_extractor(
    encoder: nn.Module,
    projection_head: Optional[nn.Module] = None,
    use_projection: bool = False,
    device: Optional[torch.device] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wrap encoder (and optional projection head) into a callable that returns features.
    """
    def _forward(images: torch.Tensor) -> torch.Tensor:
        if device is not None:
            images_local = images.to(device)
        else:
            images_local = images
        feats = encoder(images_local)
        if use_projection and projection_head is not None:
            feats = projection_head(feats)
        return feats

    return _forward

