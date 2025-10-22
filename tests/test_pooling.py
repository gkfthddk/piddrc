"""Unit tests for pooling layers used by point-set models."""

import torch

from pid.models.base import MaskedMaxMeanPool


def test_masked_pool_includes_attention():
    feature_dim = 8
    pool = MaskedMaxMeanPool(feature_dim)
    features = torch.randn(3, 5, feature_dim)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    pooled = pool(features, mask)

    assert isinstance(pool.attention, torch.nn.MultiheadAttention)
    assert pooled.shape == (3, feature_dim * 3)
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled[2], torch.zeros(feature_dim * 3), atol=1e-6)


def test_masked_pool_can_disable_attention():
    feature_dim = 4
    pool = MaskedMaxMeanPool(feature_dim, use_attention=False)
    features = torch.randn(2, 7, feature_dim)
    mask = torch.ones(2, 7, dtype=torch.bool)

    pooled = pool(features, mask)

    assert pooled.shape == (2, feature_dim * 2)
    assert torch.isfinite(pooled).all()
