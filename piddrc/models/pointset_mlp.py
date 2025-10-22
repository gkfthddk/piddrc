"""Simple masked point-set MLP for dual-readout calorimeter hits."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import ModelOutputs, PointSetAggregator


class PointSetMLPEncoder(nn.Module):
    """Lightweight point-set feature extractor built from MLP blocks."""

    def __init__(self, in_channels: int, hidden_channels: Sequence[int] = (64, 128, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for hidden in hidden_channels:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden))
            prev = hidden
        self.mlp = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.mlp(points)


class PointSetMLP(nn.Module):
    """Point-set encoder with masked pooling and multi-task prediction head."""

    def __init__(
        self,
        in_channels: int,
        *,
        summary_dim: int,
        num_classes: int,
        backbone_channels: Sequence[int] = (64, 128, 256),
        head_hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        use_summary: bool = True,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = PointSetMLPEncoder(in_channels, backbone_channels)
        self.aggregator = PointSetAggregator(
            backbone_channels[-1],
            summary_dim=summary_dim,
            head_hidden=head_hidden,
            dropout=dropout,
            num_classes=num_classes,
            use_summary=use_summary,
            use_uncertainty=use_uncertainty,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        points = batch["points"]
        mask = batch["mask"]
        per_point = self.backbone(points)
        return self.aggregator(per_point, batch)


__all__ = ["PointSetMLP"]
