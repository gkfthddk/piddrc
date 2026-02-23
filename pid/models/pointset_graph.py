"""
DGCNN-style GNN baseline for point-set inputs (dual-readout calorimeter hits).

- Local kNN graph + EdgeConv message passing (static or dynamic)
- Mask-aware pipeline, AMP-friendly
- Drop-in compatible with PointSetAggregator / ModelOutputs used in piddrc

Batch expectation (minimal):
    batch["points"] : (B, N, C_in)  per-point features
    batch["pos"]    : (B, N, 3)     xyz coordinates (required for kNN)
    batch["mask"]   : (B, N)        bool mask (True for valid points)

Usage example:
    model = PointGraphNet(
        in_channels=5,
        summary_dim=8,
        num_classes=3,
        embed_dim=128,
        k=16,
        depth=4,
        dynamic_knn=False,
        head_hidden=(256,128),
        dropout=0.1,
        use_summary=True,
        use_uncertainty=True,
    )
    outputs: ModelOutputs = model(batch)
"""
from __future__ import annotations

from typing import Sequence, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .base import ModelOutputs, PointSetAggregator


# ---------------------------------
# Utility: safe norms and kNN
# ---------------------------------

def _pairwise_dist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between x and y.
    x: (B, Nx, C), y: (B, Ny, C) -> (B, Nx, Ny)
    """
    x2 = (x * x).sum(-1, keepdim=True)           # (B, Nx, 1)
    y2 = (y * y).sum(-1).unsqueeze(1)            # (B, 1, Ny)
    xy = x @ y.transpose(-1, -2)                 # (B, Nx, Ny)
    dist2 = (x2 + y2 - 2 * xy).clamp_min(0.0)
    return dist2


def knn_indices(pos: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return indices of k nearest neighbors for each point, including self.
    pos:  (B, N, 3)
    mask: (B, N) bool, True for valid points
    -> (B, N, k) long
    """
    B, N, _ = pos.shape
    dist2 = _pairwise_dist2(pos, pos)  # (B,N,N)

    if mask is not None:
        # Invalidate neighbors that are padding
        inv = ~mask
        dist2 = dist2.masked_fill(inv.unsqueeze(1).expand(B, N, N), float('inf'))

    # ensure self is the closest
    eye = torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0)
    dist2 = dist2.masked_fill(eye, 0.0)

    k = min(k, N)
    _, idx = torch.topk(dist2, k=k, dim=-1, largest=False, sorted=False)  # (B,N,k)
    return idx


def index_points(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather per-point features by (B,N,k) indices -> (B,N,k,C)."""
    B, N, C = feats.shape
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)
    gathered = torch.gather(feats.unsqueeze(1).expand(B, N, -1, C), 2, idx_exp)
    return gathered


# ---------------------------------
# EdgeConv (DGCNN)
# ---------------------------------
class EdgeConv(nn.Module):
    """EdgeConv: h_i' = max_{j in N(i)} phi( [h_i, h_j - h_i, rel_pos] )
    - rel_pos = p_j - p_i (optional if pos is given)
    - phi: small MLP
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = max(out_dim, in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # h: (B,N,C_in), pos: (B,N,3), idx: (B,N,k)
        B, N, C = h.shape
        k = idx.size(-1)
        h_i = h.unsqueeze(2).expand(B, N, k, C)             # (B,N,k,C)
        h_j = index_points(h, idx)                          # (B,N,k,C)
        rel = index_points(pos, idx) - pos.unsqueeze(2)     # (B,N,k,3)
        msg = torch.cat([h_i, h_j - h_i, rel], dim=-1)      # (B,N,k,2C+3)
        out = self.mlp(msg)                                 # (B,N,k,out)
        out, _ = out.max(dim=2)                             # (B,N,out)
        return self.norm(out)


class GraphBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 16, dropout: float = 0.0):
        super().__init__()
        self.k = int(k)
        self.conv = EdgeConv(in_dim, out_dim, hidden=None, dropout=dropout)
        self.res = (in_dim == out_dim)
        self.ff = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, h: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor], *, idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if idx is None:
            # Dynamic graph: compute kNN on the fly
            idx = knn_indices(pos, self.k, mask=mask)

        # Static graph: kNN indices are pre-computed and passed in.
        # The 'idx' tensor is reused across all graph blocks.
        x = self.conv(h, pos, idx)
        x = x + (h if self.res else 0.0)
        x = x + self.norm(self.ff(x))
        return x


# ---------------------------------
# Encoder
# ---------------------------------
class PointGraphEncoder(nn.Module):
    """DGCNN-like encoder mapping (B,N,in) -> (B,N,embed)."""
    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 128,
        depth: int = 4,
        k_neighbors: int = 16,
        dropout: float = 0.0,
        dynamic_knn: bool = False,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(in_channels, embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(GraphBlock(embed_dim, embed_dim, k=k_neighbors, dropout=dropout))
        self.k_neighbors = int(k_neighbors)
        self.dynamic_knn = bool(dynamic_knn)

    @torch.no_grad()
    def _precompute_idx(self, pos: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return knn_indices(pos, self.k_neighbors, mask=mask)

    def forward(self, points: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # points: (B,N,C_in), pos: (B,N,3)
        x = self.input(points)
        if self.dynamic_knn:
            for block in self.blocks:
                x = block(x, pos, mask, idx=None)
        else:
            idx0 = self._precompute_idx(pos, mask)
            for block in self.blocks:
                x = block(x, pos, mask, idx=idx0)
        return x  # (B,N,embed_dim)


# ---------------------------------
# Full model: aggregator compatible
# ---------------------------------
class PointSetGraphNet(nn.Module):
    """GNN-style baseline with masked pooling & multi-task head via PointSetAggregator."""
    def __init__(
        self,
        in_channels: int,
        *,
        summary_dim: int,
        num_classes: int,
        embed_dim: int = 128,
        depth: int = 4,
        k_neighbors: int = 16,
        dropout: float = 0.1,
        dynamic_knn: bool = False,
        head_hidden: Sequence[int] = (256, 128),
        use_summary: bool = True,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = PointGraphEncoder(
            in_channels,
            embed_dim=embed_dim,
            depth=depth,
            k_neighbors=k_neighbors,
            dropout=dropout,
            dynamic_knn=dynamic_knn,
        )
        self.aggregator = PointSetAggregator(
            embed_dim,
            summary_dim=summary_dim,
            head_hidden=head_hidden,
            dropout=dropout,
            num_classes=num_classes,
            use_summary=use_summary,
            use_uncertainty=use_uncertainty,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        points = batch["points"]  # (B,N,C_in)
        pos = batch["pos"]        # (B,N,3)
        mask = batch["mask"]      # (B,N)
        per_point = self.backbone(points, pos=pos, mask=mask)
        return self.aggregator(per_point, batch)


__all__ = [
    "PointSetGraphNet",
]
