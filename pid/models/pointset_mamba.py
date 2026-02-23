"""Point-set sequence model powered by ``mamba-ssm`` selective-state layers."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Literal, Sequence

import torch
from torch import nn

from .base import ModelOutputs, PointSetAggregator

_MAMBA_SPEC = importlib.util.find_spec("mamba_ssm")
if _MAMBA_SPEC is not None:
    _MAMBA_MODULE = importlib.import_module("mamba_ssm")
    Mamba = getattr(_MAMBA_MODULE, "Mamba", None)
    try:
        Mamba2 = getattr(_MAMBA_MODULE, "Mamba2")
    except AttributeError:
        try:
            _MAMBA2_MODULE = importlib.import_module("mamba_ssm.modules.mamba2")
        except ModuleNotFoundError:
            Mamba2 = None
        else:
            Mamba2 = getattr(_MAMBA2_MODULE, "Mamba2", None)
else:
    Mamba = None
    Mamba2 = None


_SUPPORTED_BACKENDS: tuple[str, ...] = ("mamba", "mamba2")
_AVAILABLE_MAMBA_BACKENDS = {
    name: layer
    for name, layer in (("mamba", Mamba), ("mamba2", Mamba2))
    if layer is not None
}


def _resolve_backend(backend: Literal["mamba", "mamba2"]) -> type[nn.Module]:
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown Mamba backend '{backend}'. Supported backends: {_SUPPORTED_BACKENDS!r}.")
    if not _AVAILABLE_MAMBA_BACKENDS:
        raise ImportError(
            "PointSetMamba requires the optional dependency 'mamba-ssm'. "
            "Install it with `pip install mamba-ssm`."
        )
    layer_cls = _AVAILABLE_MAMBA_BACKENDS.get(backend)
    if layer_cls is None:
        available = ", ".join(sorted(_AVAILABLE_MAMBA_BACKENDS)) or "none"
        raise ImportError(
            f"Requested backend '{backend}' is unavailable. "
            f"Detected backends: {available}. Upgrade or reinstall `mamba-ssm` to access additional backends."
        )
    return layer_cls


class MambaBlock(nn.Module):
    """Wrapper around selective-state layers with masking support."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        backend: Literal["mamba", "mamba2"] = "mamba",
        **backend_kwargs,
    ) -> None:
        super().__init__()
        layer_cls = _resolve_backend(backend)
        self.norm = nn.LayerNorm(dim)
        self.mamba = layer_cls(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, **backend_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(dtype=x.dtype)
        residual = x * mask_float
        x = self.norm(x)
        x = x * mask_float
        x = self.mamba(x)
        x = self.dropout(x)
        x = x * mask_float
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
        use_direction: bool = False,
        direction_dim: int = 2,
        backend: Literal["mamba", "mamba2"] = "mamba",
        backend_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if backend_kwargs is None:
            backend_kwargs = {}
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    hidden_dim,
                    dropout=dropout,
                    backend=backend,
                    **backend_kwargs,
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
            use_direction=use_direction,
            direction_dim=direction_dim,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        points = batch["points"]
        mask = batch["mask"]
        mask_float = mask.unsqueeze(-1).to(dtype=points.dtype)
        x = self.input_proj(points) * mask_float
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.aggregator(x, batch)


__all__ = ["PointSetMamba"]
