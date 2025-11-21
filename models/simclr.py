import torch.nn as nn
import torch.nn.functional as F


class SimCLRModel(nn.Module):
    """
    Combine an encoder and projection head with configurable projection usage.
    proj_mode:
        - no_head: use encoder features directly for the contrastive loss.
        - head_on_hfeat: compute loss on projection g(h), downstream uses h.
        - head_on_zfeat: compute loss on projection g(h), downstream uses z for downstream.
    """

    def __init__(self, encoder: nn.Module, projection_head: nn.Module, proj_mode: str = "head_on_hfeat"):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        assert proj_mode in {"no_head", "head_on_hfeat", "head_on_zfeat"}
        self.proj_mode = proj_mode

    def forward(self, x, return_features: bool = False):
        h = self.encoder(x)
        if self.proj_mode == "no_head":
            z = F.normalize(h, dim=1)
            downstream_feat = h
        else:
            z_raw = self.projection_head(h)
            z = F.normalize(z_raw, dim=1)
            downstream_feat = z_raw if self.proj_mode == "head_on_zfeat" else h

        if return_features:
            return z, downstream_feat
        return z

