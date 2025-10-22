"""Integration tests that exercise the event pipeline with real dependencies."""

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
from torch.utils.data import DataLoader

from piddrc.data import DualReadoutEventDataset, collate_events
from piddrc.engine import Trainer, TrainingConfig
from piddrc.models.mlp import SummaryMLP
from piddrc.models.pointset_mlp import PointSetMLP
from piddrc.models.pointset_mamba import PointSetMamba
from piddrc.models.pointset_transformer import PointSetTransformer


HIT_FEATURES = ("x", "y", "z", "S", "C", "t")


@pytest.fixture
def dummy_h5(tmp_path):
    file_path = tmp_path / "dummy.h5"
    rng = np.random.default_rng(1234)
    with h5py.File(file_path, "w") as f:
        for feature in HIT_FEATURES:
            data = rng.normal(size=(8, 32)).astype(np.float32)
            f.create_dataset(feature, data=data)
        labels = np.array(["electron", "pion"] * 4, dtype="S")
        energies = rng.uniform(5, 50, size=8).astype(np.float32)
        f.create_dataset("particle_type", data=labels)
        f.create_dataset("true_energy", data=energies)
    return file_path


@pytest.fixture
def pipeline_dataset(dummy_h5):
    return DualReadoutEventDataset(
        [str(dummy_h5)],
        hit_features=HIT_FEATURES,
        label_key="particle_type",
        energy_key="true_energy",
        scintillation_key="S",
        cherenkov_key="C",
        depth_key="z",
        time_key="t",
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
