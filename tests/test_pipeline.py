"""Integration tests that exercise the event pipeline with real dependencies."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

try:
    import numpy as np
    import torch
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise pytest.UsageError(
        "The pipeline integration tests require numpy, torch and h5py. "
        "Install them via 'pip install -r requirements.txt' and re-run pytest."
    ) from exc

from torch.utils.data import DataLoader, Subset

from pid.data import DualReadoutEventDataset, collate_events
from pid.engine import Trainer, TrainingConfig
from pid.models.mlp import SummaryMLP
from pid.models.pointset_mlp import PointSetMLP
from pid.models.pointset_mamba import PointSetMamba
from pid.models.pointset_transformer import PointSetTransformer

import run


HIT_FEATURES = (
            "DRcalo3dHits.amplitude_sum",
            "DRcalo3dHits.type",
            "DRcalo3dHits.time",
            "DRcalo3dHits.time_end",
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
            )


@pytest.fixture
def dummy_h5(tmp_path):
    file_path = tmp_path / "dummy.h5"
    rng = np.random.default_rng(1234)
    with h5py.File(file_path, "w") as f:
        for feature in HIT_FEATURES:
            data = rng.normal(size=(8, 32)).astype(np.float32)
            f.create_dataset(feature, data=data)
        labels = np.array(["11", "211"] * 4, dtype="S")
        energies = rng.uniform(5, 50, size=8).astype(np.float32)
        f.create_dataset("GenParticles.PDG", data=labels)
        f.create_dataset("E_gen", data=energies)
    return file_path


@pytest.fixture
def pipeline_dataset(dummy_h5):
    return DualReadoutEventDataset(
        [str(dummy_h5)],
        hit_features=HIT_FEATURES,
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        max_points=16,
    )


def test_collate_and_models(pipeline_dataset):
    dataset = pipeline_dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_events)
    batch = next(iter(loader))
    assert batch["points"].shape[1] == 16
    summary_dim = batch["summary"].shape[-1]
    num_classes = len(dataset.classes)

    mlp = SummaryMLP(summary_dim=summary_dim, num_classes=num_classes)
    outputs = mlp(batch)
    assert outputs.logits.shape == (4, num_classes)

    point_mlp = PointSetMLP(in_channels=batch["points"].shape[-1], summary_dim=summary_dim, num_classes=num_classes)
    outputs = point_mlp(batch)
    assert outputs.energy.shape == (4,)

    transformer = PointSetTransformer(
        in_channels=batch["points"].shape[-1],
        summary_dim=summary_dim,
        num_classes=num_classes,
        depth=2,
        num_heads=4,
    )
    outputs = transformer(batch)
    assert outputs.logits.shape == (4, num_classes)

    if importlib.util.find_spec("mamba_ssm") is not None:
        point_mamba = PointSetMamba(in_channels=batch["points"].shape[-1], summary_dim=summary_dim, num_classes=num_classes)
        outputs = point_mamba(batch)
        assert outputs.logits.shape == (4, num_classes)


def test_trainer_step(pipeline_dataset):
    dataset = pipeline_dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_events)
    batch = next(iter(loader))
    summary_dim = batch["summary"].shape[-1]
    num_classes = len(dataset.classes)
    model = PointSetMLP(in_channels=batch["points"].shape[-1], summary_dim=summary_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, device=torch.device("cpu"), config=TrainingConfig(epochs=1, log_every=1, use_amp=False))
    history = trainer.fit(loader)
    assert len(history["train"]) == 1
    metrics = history["train"][0]
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_build_datasets_auto_split(dummy_h5):
    args = argparse.Namespace(
        train_files=[dummy_h5],
        val_files=None,
        test_files=None,
        hit_features=HIT_FEATURES,
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        max_points=16,
        val_fraction=0.25,
        test_fraction=0.25,
        split_seed=42,
    )

    base_dataset, train_dataset, val_dataset, test_dataset = run.build_datasets(args)

    assert isinstance(base_dataset, DualReadoutEventDataset)
    assert isinstance(train_dataset, Subset)
    assert isinstance(val_dataset, Subset)
    assert isinstance(test_dataset, Subset)

    total_length = len(train_dataset) + len(val_dataset) + len(test_dataset)
    assert total_length == len(base_dataset)

    train_indices = {int(idx) for idx in train_dataset.indices}
    val_indices = {int(idx) for idx in val_dataset.indices}
    test_indices = {int(idx) for idx in test_dataset.indices}

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    # The split should be deterministic for a given seed
    _, train_dataset_b, val_dataset_b, test_dataset_b = run.build_datasets(args)
    assert isinstance(train_dataset_b, Subset)
    assert train_dataset.indices == train_dataset_b.indices
    assert val_dataset.indices == val_dataset_b.indices
    assert test_dataset.indices == test_dataset_b.indices
