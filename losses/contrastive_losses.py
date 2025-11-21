import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineMarginLoss(nn.Module):
    """
    Penalize negatives whose similarity exceeds a margin while pulling positives together.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        pos_sim = F.cosine_similarity(z_i, z_j, dim=1)
        pos_loss = 1.0 - pos_sim

        sim_matrix = torch.matmul(z, z.T)
        diag_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -2.0)

        positive_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        positive_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device, dtype=torch.bool)
        positive_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device, dtype=torch.bool)

        negative_mask = (~positive_mask) & (~diag_mask)
        negative_sim = sim_matrix[negative_mask]
        neg_loss = torch.relu(negative_sim - self.margin)

        return pos_loss.mean() + (neg_loss.mean() if neg_loss.numel() > 0 else 0.0)


class PositiveOnlySimilarityLoss(nn.Module):
    """
    Encourage only the positive pairs to be similar (no explicit negatives).
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        sim = F.cosine_similarity(z_i, z_j, dim=1)
        return 1.0 - sim.mean()

