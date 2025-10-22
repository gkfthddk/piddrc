"""Point-set sequence model powered by the official Mamba selective-state layer."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence

import torch
from torch import nn

from .base import ModelOutputs, PointSetAggregator

_MAMBA_SPEC = importlib.util.find_spec("mamba_ssm")
if _MAMBA_SPEC is not None:
    _MAMBA_MODULE = importlib.import_module("mamba_ssm")
    Mamba = getattr(_MAMBA_MODULE, "Mamba")
else:
    Mamba = None


class MambaBlock(nn.Module):
    """Wrapper around :class:`mamba_ssm.Mamba` with masking support."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "PointSetMamba requires the optional dependency 'mamba-ssm'. "
                "Install it with `pip install mamba-ssm`."
            )
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x * mask.unsqueeze(-1).float()
        return residual + x


class PointSetMamba(nn.Module):
    """Point-set architecture built on stacked Mamba blocks."""

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
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    hidden_dim,
                    d_state=hidden_dim // 2,
                    d_conv=4,
                    expand=2,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.aggregator = PointSetAggregator(
            hidden_dim,
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
        x = self.input_proj(points)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.aggregator(x, batch)


__all__ = ["PointSetMamba"]
