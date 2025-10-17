"""Simple MLP baseline operating on engineered summary features."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .base import ModelOutputs, MultiTaskHead


class SummaryMLP(nn.Module):
    """Two-stage MLP that consumes event-level summary statistics."""

    def __init__(
        self,
        summary_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.1,
        num_classes: int,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(summary_dim)]
        in_dim = summary_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden
        self.encoder = nn.Sequential(*layers)
        self.head = MultiTaskHead(in_dim, hidden_dims=(128,), dropout=dropout, num_classes=num_classes, use_uncertainty=use_uncertainty)

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        summary = batch["summary"]
        features = self.encoder(summary)
        return self.head(features)


__all__ = ["SummaryMLP"]
