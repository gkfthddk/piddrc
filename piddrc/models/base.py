"""Model building blocks shared by multiple architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn


@dataclass
class ModelOutputs:
    """Bundle returned by all models in this package."""

    logits: torch.Tensor
    energy: torch.Tensor
    log_sigma: Optional[torch.Tensor]
    extras: Dict[str, torch.Tensor]


class MultiTaskHead(nn.Module):
    """Shared head that predicts class logits, energy and log-variance."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        num_classes: int,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)
        self.regressor = nn.Linear(prev, 1)
        self.log_sigma_head = nn.Linear(prev, 1) if use_uncertainty else None

    def forward(self, features: torch.Tensor) -> ModelOutputs:
        shared = self.backbone(features)
        logits = self.classifier(shared)
        energy = self.regressor(shared).squeeze(-1)
        log_sigma = self.log_sigma_head(shared).squeeze(-1) if self.log_sigma_head is not None else None
        return ModelOutputs(logits=logits, energy=energy, log_sigma=log_sigma, extras={"shared": shared})


class MaskedMaxMeanPool(nn.Module):
    """Combine masked global max and mean pooling for variable-length sets."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    @property
    def output_dim(self) -> int:
        return self.feature_dim * 2

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1)
        masked = features.masked_fill(~valid.bool(), float("-inf"))
        global_max = masked.max(dim=1).values
        global_max[~torch.isfinite(global_max)] = 0.0
        denom = valid.float().sum(dim=1).clamp_min(1.0)
        mean = (features * valid.float()).sum(dim=1) / denom
        return torch.cat([global_max, mean], dim=-1)


class SummaryProjector(nn.Module):
    """Optional projection layer that fuses summary features with set embeddings."""

    def __init__(self, summary_dim: int, target_dim: int, enabled: bool = True) -> None:
        super().__init__()
        if enabled:
            self.proj = nn.Sequential(
                nn.LayerNorm(summary_dim),
                nn.Linear(summary_dim, target_dim),
                nn.GELU(),
            )
            self._output_dim = target_dim
        else:
            self.proj = None
            self._output_dim = 0

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, summary: torch.Tensor | None) -> torch.Tensor | None:
        if self.proj is None or summary is None:
            return None
        return self.proj(summary)


__all__ = [
    "ModelOutputs",
    "MultiTaskHead",
    "MaskedMaxMeanPool",
    "SummaryProjector",
]
