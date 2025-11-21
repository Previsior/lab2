import torch.nn as nn


class LinearClassifier(nn.Module):
    """Single linear layer classifier used for linear eval or fine-tuning."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

