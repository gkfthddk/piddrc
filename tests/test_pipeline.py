import importlib.util
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
import numpy as np
import torch
from torch.utils.data import DataLoader

from piddrc.data import DualReadoutEventDataset, collate_events
from piddrc.engine import Trainer, TrainingConfig
from piddrc.models.mlp import SummaryMLP
from piddrc.models.pointset_mlp import PointSetMLP
from piddrc.models.pointset_mamba import PointSetMamba


def _create_dummy_file(path: Path, num_events: int = 8, num_hits: int = 32) -> None:
    rng = np.random.default_rng(1234)
    with h5py.File(path, "w") as f:
        for feature in ["x", "y", "z", "S", "C", "t"]:
            data = rng.normal(size=(num_events, num_hits)).astype(np.float32)
            f.create_dataset(feature, data=data)
        labels = np.array(["electron", "pion"] * (num_events // 2), dtype="S")
        energies = rng.uniform(5, 50, size=num_events).astype(np.float32)
        f.create_dataset("particle_type", data=labels)
        f.create_dataset("true_energy", data=energies)


def _build_dataset(tmp_path: Path) -> DualReadoutEventDataset:
    file_path = tmp_path / "dummy.h5"
    _create_dummy_file(file_path)
    dataset = DualReadoutEventDataset(
        [str(file_path)],
        hit_features=("x", "y", "z", "S", "C", "t"),
        label_key="particle_type",
        energy_key="true_energy",
        scintillation_key="S",
        cherenkov_key="C",
        depth_key="z",
        time_key="t",
        max_points=16,
    )
    return dataset


def test_collate_and_models(tmp_path):
    dataset = _build_dataset(tmp_path)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_events)
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

    if importlib.util.find_spec("mamba_ssm") is not None:
        point_mamba = PointSetMamba(in_channels=batch["points"].shape[-1], summary_dim=summary_dim, num_classes=num_classes)
        outputs = point_mamba(batch)
        assert outputs.logits.shape == (4, num_classes)


def test_trainer_step(tmp_path):
    dataset = _build_dataset(tmp_path)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_events)
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

