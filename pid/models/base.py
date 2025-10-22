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
    """Combine masked global max, mean and multi-head attention pooling."""

    def __init__(
        self,
        feature_dim: int,
        *,
        use_attention: bool = True,
        attention_heads: int = 4,
        attention_queries: int = 1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        if use_attention:
            if feature_dim % attention_heads != 0:
                raise ValueError("feature_dim must be divisible by attention_heads")
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
            self.num_attention_queries = attention_queries
            query = torch.randn(attention_queries, feature_dim)
            nn.init.xavier_uniform_(query)
            self.register_parameter("attention_query", nn.Parameter(query))
            self.attn_norm = nn.LayerNorm(feature_dim)
        else:
            self.attention = None
            self.num_attention_queries = 0
            self.register_parameter("attention_query", None)
            self.attn_norm = None

    @property
    def output_dim(self) -> int:
        base = self.feature_dim * 2
        if self.attention is not None:
            base += self.feature_dim * self.num_attention_queries
        return base

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.bool().unsqueeze(-1)
        masked = features.masked_fill(~valid, float("-inf"))
        global_max = masked.max(dim=1).values
        global_max[~torch.isfinite(global_max)] = 0.0

        valid_float = valid.float()
        denom = valid_float.sum(dim=1).clamp_min(1.0)
        mean = (features * valid_float).sum(dim=1) / denom

        pooled = [global_max, mean]

        if self.attention is not None:
            batch, _, _ = features.shape
            attn_output = torch.zeros(
                batch,
                self.num_attention_queries,
                self.feature_dim,
                device=features.device,
                dtype=features.dtype,
            )
            valid_batches = mask.any(dim=1)
            if valid_batches.any():
                queries = self.attention_query.unsqueeze(0).expand(batch, -1, -1)
                attn_mask = ~mask.bool()
                attn_input = self.attn_norm(features)
                attn_valid, _ = self.attention(
                    queries[valid_batches],
                    attn_input[valid_batches],
                    attn_input[valid_batches],
                    key_padding_mask=attn_mask[valid_batches],
                )
                attn_output[valid_batches] = torch.nan_to_num(attn_valid).to(attn_output.dtype)
            pooled.append(attn_output.reshape(batch, -1))

        return torch.cat(pooled, dim=-1)


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


class PointSetAggregator(nn.Module):
    """Helper module that pools per-point embeddings and applies the multi-task head."""

    def __init__(
        self,
        feature_dim: int,
        *,
        summary_dim: int,
        head_hidden: Sequence[int],
        dropout: float,
        num_classes: int,
        use_summary: bool,
        use_uncertainty: bool,
    ) -> None:
        super().__init__()
        self.pool = MaskedMaxMeanPool(feature_dim)
        self.summary_proj = SummaryProjector(summary_dim, feature_dim, enabled=use_summary)
        combined_dim = self.pool.output_dim + self.summary_proj.output_dim
        self.head = MultiTaskHead(
            combined_dim,
            hidden_dims=head_hidden,
            dropout=dropout,
            num_classes=num_classes,
            use_uncertainty=use_uncertainty,
        )

    def forward(self, per_point: torch.Tensor, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        mask = batch["mask"]
        features = self.pool(per_point, mask)
        summary = self.summary_proj(batch.get("summary"))
        if summary is not None:
            features = torch.cat([features, summary], dim=-1)
        return self.head(features)


__all__ = [
    "ModelOutputs",
    "MultiTaskHead",
    "MaskedMaxMeanPool",
    "SummaryProjector",
    "PointSetAggregator",
]
