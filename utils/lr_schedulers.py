import math
from typing import Optional

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup for a fixed number of steps followed by cosine decay.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = (step + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            (base_lr - base_lr * self.min_lr_factor) * cosine_decay + base_lr * self.min_lr_factor
            for base_lr in self.base_lrs
        ]


def build_warmup_cosine_scheduler(
    optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 10,
    min_lr_factor: float = 0.01,
) -> WarmupCosineScheduler:
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_factor=min_lr_factor,
    )

