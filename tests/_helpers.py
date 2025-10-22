"""Shared testing utilities and constants."""

from __future__ import annotations

from typing import Dict

import torch

TEST_HIT_FEATURES = ("x", "y", "z", "S", "C", "t")
TEST_CLASSES = ("electron", "pion")
SUMMARY_DIM = 8
MAX_POINTS = 16
BATCH_SIZE = 4


def create_synthetic_batch(
    *,
    batch_size: int = BATCH_SIZE,
    max_points: int = MAX_POINTS,
    feature_dim: int | None = None,
    summary_dim: int = SUMMARY_DIM,
    num_classes: int | None = None,
    seed: int = 1337,
) -> Dict[str, torch.Tensor]:
    """Create a synthetic batch matching :func:`piddrc.data.collate_events` output."""

    if feature_dim is None:
        feature_dim = len(TEST_HIT_FEATURES)
    if num_classes is None:
        num_classes = len(TEST_CLASSES)

    generator = torch.Generator().manual_seed(seed)

    points = torch.randn(batch_size, max_points, feature_dim, generator=generator)
    lengths = torch.randint(1, max_points + 1, (batch_size,), generator=generator)
    mask = torch.zeros(batch_size, max_points, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, : length.item()] = True
    points = points * mask.unsqueeze(-1)

    summary = torch.randn(batch_size, summary_dim, generator=generator)
    labels = torch.randint(num_classes, (batch_size,), generator=generator)
    energy = torch.rand(batch_size, generator=generator)
    event_id = torch.stack(
        (
            torch.zeros(batch_size, dtype=torch.long),
            torch.arange(batch_size, dtype=torch.long),
        ),
        dim=1,
    )

    return {
        "points": points,
        "mask": mask,
        "summary": summary,
        "labels": labels,
        "energy": energy,
        "event_id": event_id,
    }


__all__ = [
    "TEST_HIT_FEATURES",
    "TEST_CLASSES",
    "SUMMARY_DIM",
    "MAX_POINTS",
    "BATCH_SIZE",
    "create_synthetic_batch",
]
