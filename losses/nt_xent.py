import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """
    Normalized temperature-scaled cross entropy loss used in SimCLR.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        # Remove similarities of samples with themselves
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive pairs: i with j (offset by batch_size)
        positives = torch.cat(
            [torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)], dim=0
        )

        numerator = torch.exp(positives)
        denominator = torch.exp(sim_matrix).sum(dim=1)

        loss = -torch.log(numerator / denominator)
        return loss.mean()

