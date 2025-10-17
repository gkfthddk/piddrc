"""Masked point-set encoder built on Transformer self-attention blocks."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import MaskedMaxMeanPool, ModelOutputs, MultiTaskHead, SummaryProjector


class PointSetTransformer(nn.Module):
    """Point-set architecture leveraging Transformer encoders."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int = 128,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        summary_dim: int,
        num_classes: int,
        head_hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_summary: bool = True,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_layer.self_attn.dropout = attn_dropout
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = MaskedMaxMeanPool(hidden_dim)
        self.summary_proj = SummaryProjector(summary_dim, hidden_dim, enabled=use_summary)

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

        x = self.input_proj(points)
        padding_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        x = x * mask.unsqueeze(-1).float()

        features = self.pool(x, mask)
        summary = self.summary_proj(batch.get("summary"))
        if summary is not None:
            features = torch.cat([features, summary], dim=-1)
        return self.head(features)


__all__ = ["PointSetTransformer"]

