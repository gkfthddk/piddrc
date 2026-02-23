"""
Point Transformer v3 (PTv3) style local-attention encoder for point-set inputs.
- Local neighborhood attention with relative positional encoding (RPE)
- O(N * k) complexity (k: neighbors per point)
- AMP-friendly (FP16/BF16): uses numerically-stable ops, no global SDPA

Drop-in for piddrc:
    File: pid/models/pointset_transformer_v3.py
    Class: PointSetTransformerV3

Inputs
------
- x:   (B, N, C) point features
- pos: (B, N, 3)  xyz coordinates
- mask: (B, N) boolean mask (True for valid point), optional

Outputs
-------
- y: (B, N, C) encoded features (same channel dim as input `embed_dim`)

Notes
-----
- Neighborhood via kNN in torch (no external deps).
- Stable attention weight computation with small eps and clamping.
- Attention is computed only over local neighbors -> speed & memory efficiency.
- Includes a tiny feed-forward block per layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutputs, PointSetAggregator

# ---------------------------
# Utilities
# ---------------------------

def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-6) -> torch.Tensor:
    return torch.linalg.norm(x, ord=2, dim=dim, keepdim=keepdim).clamp_min(eps)


def pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distance between two point sets per batch.
    a: (B, Na, C), b: (B, Nb, C) -> (B, Na, Nb)
    """
    # |a - b|^2 = |a|^2 + |b|^2 - 2 a.b
    a2 = (a * a).sum(-1, keepdim=True)            # (B, Na, 1)
    b2 = (b * b).sum(-1).unsqueeze(1)             # (B, 1, Nb)
    ab = a @ b.transpose(-1, -2)                  # (B, Na, Nb)
    dist2 = (a2 + b2 - 2 * ab).clamp_min(0.0)
    return dist2


def knn_indices(pos: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """kNN indices for each point (including self) using squared distance.
    pos:  (B, N, 3)
    mask: (B, N) optional boolean mask (True = valid)
    return: (B, N, k) long
    """
    B, N, _ = pos.shape
    dist2 = pairwise_distance(pos, pos)  # (B, N, N)

    if mask is not None:
        # Exclude invalid positions by setting large distance
        inv = ~mask
        dist2 = dist2.masked_fill(inv.unsqueeze(1).expand(B, N, N), float('inf'))

    # Always include self as neighbor by forcing distance 0 on diagonal
    eye = torch.eye(N, device=pos.device).bool().unsqueeze(0)  # (1, N, N)
    dist2 = dist2.masked_fill(eye, 0.0)

    # topk with largest=False for smallest distances
    k = min(k, N)
    _, idx = torch.topk(dist2, k=k, dim=-1, largest=False, sorted=False)  # (B, N, k)
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points/features at idx.
    points: (B, N, C)
    idx:    (B, N, k)
    return: (B, N, k, C)
    """
    B, N, k = idx.shape
    if points.ndim == 2:
        # Handle 2D masks (B, N) -> (B, N, k, 1)
        points = points.unsqueeze(-1)

    _, _, C = points.shape
    # Create the view of points to gather from: (B, 1, N, C) -> (B, N, N, C)
    view_shape = (B, 1, N, C)
    expanded_points = points.view(view_shape).expand(-1, N, -1, -1)
    # Gather based on kNN indices
    return torch.gather(expanded_points, 2, idx.view(B, N, k, 1).expand(-1, -1, -1, C))

# ---------------------------
# PTv3 Attention Block
# ---------------------------
class PTv3Attention(nn.Module):
    """Local attention with Relative Positional Encoding (RPE).

    Computes attention over kNN neighbors for each point:
        attn = softmax( (Q * K_rel).sum + phi(rel_pos) )
        out  = sum_j attn_ij * V_j

    Implementation choices for stability:
    - All ops kept in the feature dimension (no SDPA kernels).
    - Small epsilon and clamping to avoid NaNs in AMP.
    """
    def __init__(self, embed_dim: int, k: int = 16, rpe_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, rpe_dim), nn.ReLU(inplace=True),
            nn.Linear(rpe_dim, embed_dim)
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = (embed_dim ** -0.5)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        # Normalize features to reduce softmax saturation in AMP
        x_q = self.norm_q(x)
        x_kv = self.norm_kv(x)

        q = self.q(x_q)                    # (B,N,C)
        k = self.k_proj(x_kv)              # (B,N,C)
        v = self.v(x_kv)                   # (B,N,C)

        idx = knn_indices(pos, self.k, mask=mask)   # (B,N,k)
        k_nb = index_points(k, idx)                 # (B,N,k,C)
        v_nb = index_points(v, idx)                 # (B,N,k,C)

        # Relative positional encoding
        pos_nb = index_points(pos, idx)             # (B,N,k,3)
        rel = pos.unsqueeze(2) - pos_nb             # (B,N,k,3)
        rel_enc = self.pos_mlp(rel)                 # (B,N,k,C)

        # Scaled dot with relative shift (no global SDPA)
        q_exp = q.unsqueeze(2)                       # (B,N,1,C)
        attn_logits = (q_exp * k_nb).sum(-1) * self.scale  # (B,N,k)
        attn_logits = attn_logits + (rel_enc).sum(-1) * (self.scale)

        if mask is not None:
            # neighbor validity: if center invalid, zero out; if neighbor invalid, -inf
            center_invalid = (~mask).unsqueeze(-1)                  # (B,N,1)
            attn_logits = attn_logits.masked_fill(center_invalid, 0.0)
            nb_invalid = ~index_points(mask.float(), idx).bool().squeeze(-1)  # (B,N,k)
            attn_logits = attn_logits.masked_fill(nb_invalid, float('-inf'))

        # Stabilize softmax
        attn = F.softmax(attn_logits, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * v_nb).sum(dim=2)  # (B,N,C)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PTv3Block(nn.Module):
    def __init__(self, embed_dim: int, k: int = 16, rpe_dim: int = 64, dropout: float = 0.0, ff_mult: float = 2.0, pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = PTv3Attention(embed_dim, k=k, rpe_dim=rpe_dim, dropout=dropout)
        self.ff = FeedForward(embed_dim, mult=ff_mult, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-norm: norm -> attn -> add
            h = self.norm1(x)
            h = self.attn(h, pos, mask=mask)
            x = x + h

            h2 = self.norm2(x)
            h2 = self.ff(h2)
            x = x + h2
        else:
            # Post-norm: attn -> norm -> add
            h = self.attn(x, pos, mask=mask)
            x = self.norm1(x + h)
            x = self.norm2(x + self.ff(x))
        return x


# ---------------------------
# Encoder
# ---------------------------

class PointSetTransformerV3(nn.Module):
    """Point-set architecture leveraging PTv3-style local attention."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int = 128,
        depth: int = 4,
        k_neighbors: int = 16,
        rpe_dim: int = 64,
        ff_mult: float = 2.0,
        pre_norm: bool = True,
        summary_dim: int,
        num_classes: int,
        head_hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        use_summary: bool = True,
        use_uncertainty: bool = True,
        use_direction: bool = False,
        direction_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            PTv3Block(
                embed_dim=hidden_dim,
                k=k_neighbors,
                rpe_dim=rpe_dim,
                dropout=dropout,
                ff_mult=ff_mult,
                pre_norm=pre_norm,
            ) for _ in range(depth)
        ])
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

    @staticmethod
    def _get_pos_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for key in ("pos", "xyz", "position"):
            if key in batch and batch[key] is not None:
                return batch[key]
        raise KeyError(
            "PointSetTransformerV3 requires coordinates under 'pos', 'xyz' or 'position' in the batch."
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        points = batch["points"]
        mask = batch["mask"]
        pos = self._get_pos_from_batch(batch)
        if pos.shape[-1] != 3:
            raise ValueError(
                f"Expected 3D coordinates for PTv3, but received shape {pos.shape}"
            )
        if pos.dtype != points.dtype or pos.device != points.device:
            pos = pos.to(dtype=points.dtype, device=points.device)

        x = self.input_proj(points)
        for blk in self.blocks:
            x = blk(x, pos, mask=mask)

        x = self.norm(x)
        x = x * mask.unsqueeze(-1).float()
        return self.aggregator(x, batch)

__all__ = ["PointSetTransformerV3"]
