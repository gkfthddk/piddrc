from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset

from piddrc.engine import Trainer, TrainingConfig
from piddrc.models.mlp import SummaryMLP
from piddrc.models.pointset_mlp import PointSetMLP
from piddrc.models.pointset_mamba import PointSetMamba
from piddrc.models.pointset_transformer import PointSetTransformer
from tests._helpers import (
    SUMMARY_DIM,
    TEST_CLASSES,
    create_synthetic_batch,
)


class _SyntheticDataset(Dataset):
    def __init__(self, batch: dict[str, torch.Tensor], length: int = 2) -> None:
        self._batch = batch
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # pragma: no cover - trivial
        return {key: value.clone() for key, value in self._batch.items()}


def _synthetic_loader(num_batches: int = 2) -> DataLoader:
    batch = create_synthetic_batch()
    dataset = _SyntheticDataset(batch, length=num_batches)
    return DataLoader(dataset, batch_size=None)


def test_models_forward_pass() -> None:
    batch = create_synthetic_batch()
    summary_dim = batch["summary"].shape[-1]
    num_classes = len(TEST_CLASSES)
    feature_dim = batch["points"].shape[-1]
    batch_size = batch["points"].shape[0]

    mlp = SummaryMLP(summary_dim=summary_dim, num_classes=num_classes)
    outputs = mlp(batch)
    assert outputs.logits.shape == (batch_size, num_classes)

    point_mlp = PointSetMLP(in_channels=feature_dim, summary_dim=summary_dim, num_classes=num_classes)
    outputs = point_mlp(batch)
    assert outputs.energy.shape == (batch_size,)

    transformer = PointSetTransformer(
        in_channels=feature_dim,
        summary_dim=summary_dim,
        num_classes=num_classes,
        depth=2,
        num_heads=4,
    )
    outputs = transformer(batch)
    assert outputs.logits.shape == (batch_size, num_classes)

    if importlib.util.find_spec("mamba_ssm") is not None:
        point_mamba = PointSetMamba(in_channels=feature_dim, summary_dim=summary_dim, num_classes=num_classes)
        outputs = point_mamba(batch)
        assert outputs.logits.shape == (batch_size, num_classes)


def test_trainer_step() -> None:
    batch = create_synthetic_batch()
    feature_dim = batch["points"].shape[-1]
    num_classes = len(TEST_CLASSES)

    model = PointSetMLP(in_channels=feature_dim, summary_dim=SUMMARY_DIM, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, device=torch.device("cpu"), config=TrainingConfig(epochs=1, log_every=1, use_amp=False))

    loader = _synthetic_loader()
    history = trainer.fit(loader)
    assert len(history["train"]) == 1
    metrics = history["train"][0]
    assert "loss" in metrics
    assert "accuracy" in metrics
