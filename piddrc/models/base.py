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


__all__ = ["ModelOutputs", "MultiTaskHead"]
