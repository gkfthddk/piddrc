"""PointMLP++-like point-set network compatible with PointSetAggregator/ModelOutputs.

- Pure MLP residual stack with AdaptiveFeatureNorm (AFN)
- Optional Geometric Affine Modulation (GAM) using per-point coordinates
- Mask-aware pipeline; integrates with your existing PointSetAggregator

Batch expectation (minimal):
    batch["points"] : (B, N, C_in) per-point features
    batch["mask"]   : (B, N)      bool mask (True for valid points)
Optional (for GAM):
    batch["pos"] or batch["xyz"] or batch["position"] : (B, N, 3)

Usage:
    model = PointMLPpp(
        in_channels=5,
        summary_dim=8,
        num_classes=3,
        embed_dim=128,
        depth=8,
        expansion=4,
        dropout=0.1,
        use_gam=True,
        use_summary=True,
        use_uncertainty=True,
    )
    out: ModelOutputs = model(batch)
"""
from __future__ import annotations

from typing import Sequence, Optional

import torch
from torch import nn

from .base import ModelOutputs, PointSetAggregator


# ---------------------------
# Normalization & Utilities
# ---------------------------
class AdaptiveFeatureNorm(nn.Module):
    """LayerNorm-like normalization across channels for each point.
    Uses small epsilon for AMP stability.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


class GeometricAffine(nn.Module):
    """Per-point affine modulation (1+gamma)*x + beta from coordinates.
    If coords are missing, acts as identity.
    """
    def __init__(self, feat_dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim * 2)
        )

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        if pos is None:
            return x
        gb = self.net(pos)                # (B, N, 2C)
        gamma, beta = gb.chunk(2, dim=-1) # (B,N,C), (B,N,C)
        return x * (1.0 + gamma) + beta


# ---------------------------
# PointMLP++ Blocks
# ---------------------------
class PointMLPppBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.pre = AdaptiveFeatureNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.pre(x))


# ---------------------------
# Encoder
# ---------------------------
class PointMLPppEncoder(nn.Module):
    """PointMLP++-like encoder that maps (B,N,in) -> (B,N,embed)."""
    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 128,
        depth: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
        use_gam: bool = True,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(in_channels, embed_dim)
        self.gam = GeometricAffine(embed_dim, hidden=64, dropout=dropout) if use_gam else None
        self.blocks = nn.ModuleList([
            PointMLPppBlock(embed_dim, expansion=expansion, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, points: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # points: (B,N,C_in), pos: (B,N,3) or None
        x = self.input(points)
        if self.gam is not None:
            x = self.gam(x, pos)
        for blk in self.blocks:
            x = blk(x)
        return x  # (B,N,embed_dim)


# ---------------------------
# Full Model (aggregator-compatible)
# ---------------------------
class PointSetMLPpp(nn.Module):
    """PointMLP++-like point-set model with masked pooling & multi-task head via PointSetAggregator."""
    def __init__(
        self,
        in_channels: int,
        *,
        summary_dim: int,
        num_classes: int,
        embed_dim: int = 128,
        depth: int = 8,
        expansion: int = 4,
        dropout: float = 0.1,
        use_gam: bool = True,
        head_hidden: Sequence[int] = (256, 128),
        use_summary: bool = True,
        use_uncertainty: bool = True,
        use_direction: bool = False,
        direction_dim: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = PointMLPppEncoder(
            in_channels,
            embed_dim=embed_dim,
            depth=depth,
            expansion=expansion,
            dropout=dropout,
            use_gam=use_gam,
        )
        self.aggregator = PointSetAggregator(
            embed_dim,
            summary_dim=summary_dim,
            head_hidden=head_hidden,
            dropout=dropout,
            num_classes=num_classes,
            use_summary=use_summary,
            use_uncertainty=use_uncertainty,
            use_direction=use_direction,
            direction_dim=direction_dim,
        )

    @staticmethod
    def _get_pos_from_batch(batch: dict) -> Optional[torch.Tensor]:
        for k in ("pos", "xyz", "position"):
            if k in batch:
                return batch[k]
        return None

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        points = batch["points"]             # (B,N,C_in)
        mask = batch["mask"]                 # (B,N)
        pos = self._get_pos_from_batch(batch) # (B,N,3) or None
        per_point = self.backbone(points, pos=pos)
        return self.aggregator(per_point, batch)


__all__ = [
    "PointSetMLPpp",
]
