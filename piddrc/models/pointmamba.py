"""Sequence model inspired by Mamba for point-cloud processing."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import ModelOutputs, MultiTaskHead


class ResidualBlock(nn.Module):
    """Gated residual block approximating Mamba-style mixing."""

    def __init__(self, dim: int, *, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.linear_f = nn.Linear(dim, hidden)
        self.linear_g = nn.Linear(dim, hidden)
        self.proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        f = torch.tanh(self.linear_f(x))
        g = torch.sigmoid(self.linear_g(conv))
        mixed = f * g
        out = self.proj(mixed)
        out = self.dropout(out)
        out = out * mask.unsqueeze(-1).float()
        return residual + out


class PointMamba(nn.Module):
    """Point-based architecture using gated mixing blocks."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int = 128,
        depth: int = 4,
        summary_dim: int,
        num_classes: int,
        head_hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        use_summary: bool = True,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        feature_dim = hidden_dim * 2
        if use_summary:
            self.summary_proj = nn.Sequential(nn.LayerNorm(summary_dim), nn.Linear(summary_dim, hidden_dim), nn.GELU())
            feature_dim += hidden_dim
        else:
            self.summary_proj = None
        self.head = MultiTaskHead(
            feature_dim,
            hidden_dims=head_hidden,
            dropout=dropout,
            num_classes=num_classes,
            use_uncertainty=use_uncertainty,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        points = batch["points"]
        mask = batch["mask"]
        x = self.input_proj(points)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        valid = mask.unsqueeze(-1)
        masked = x.masked_fill(~valid.bool(), float("-inf"))
        global_max = torch.max(masked, dim=1).values
        global_max[~torch.isfinite(global_max)] = 0.0
        mean = (x * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
        features = torch.cat([global_max, mean], dim=-1)
        if self.summary_proj is not None:
            summary = self.summary_proj(batch["summary"])
            features = torch.cat([features, summary], dim=-1)
        return self.head(features)


__all__ = ["PointMamba"]
