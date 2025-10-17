"""PointNet-style architecture for dual-readout calorimeter hits."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import MaskedMaxMeanPool, ModelOutputs, MultiTaskHead, SummaryProjector


class PointNetBackbone(nn.Module):
    """Lightweight PointNet-style feature extractor."""

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


class PointNetModel(nn.Module):
    """PointNet baseline with multi-task prediction head."""

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
        self.backbone = PointNetBackbone(in_channels, backbone_channels)
        self.pool = MaskedMaxMeanPool(backbone_channels[-1])
        self.summary_proj = SummaryProjector(summary_dim, backbone_channels[-1], enabled=use_summary)
        feature_dim = self.pool.output_dim + self.summary_proj.output_dim
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
        per_point = self.backbone(points)
        features = self.pool(per_point, mask)
        summary = self.summary_proj(batch.get("summary"))
        if summary is not None:
            features = torch.cat([features, summary], dim=-1)
        return self.head(features)


__all__ = ["PointNetModel"]
