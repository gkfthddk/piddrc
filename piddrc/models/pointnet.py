"""PointNet-style architecture for dual-readout calorimeter hits."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import ModelOutputs, MultiTaskHead


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

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.mlp(points)
        valid = mask.unsqueeze(-1)
        masked = features.masked_fill(~valid, float("-inf"))
        global_max = torch.max(masked, dim=1).values
        global_max[~torch.isfinite(global_max)] = 0.0
        mean = (features * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
        return torch.cat([global_max, mean], dim=-1)


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
        feature_dim = 2 * backbone_channels[-1]
        if use_summary:
            self.summary_proj = nn.Sequential(nn.LayerNorm(summary_dim), nn.Linear(summary_dim, backbone_channels[-1]), nn.GELU())
            feature_dim += backbone_channels[-1]
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
        global_features = self.backbone(points, mask)
        features = global_features
        if self.summary_proj is not None:
            summary = self.summary_proj(batch["summary"])
            features = torch.cat([global_features, summary], dim=-1)
        return self.head(features)


__all__ = ["PointNetModel"]
