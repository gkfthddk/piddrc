"""Integration tests that exercise the event pipeline with real dependencies."""

import argparse
import importlib.util
import sys
from pathlib import Path
import json

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

from torch import nn
from torch.utils.data import DataLoader, Subset

from pid.data import DualReadoutEventDataset, collate_events
from pid.engine import Trainer, TrainingConfig
from pid.models.base import ModelOutputs
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
    file_path1 = tmp_path / "dummy1.h5"
    rng = np.random.default_rng(1234)
    with h5py.File(file_path1, "w") as f:
        for feature in HIT_FEATURES:
            data = rng.normal(size=(8, 32)).astype(np.float32)
            f.create_dataset(feature, data=data)
        labels = np.array(["11"] * 8, dtype="S")
        energies = rng.uniform(5, 50, size=8).astype(np.float32)
        c_amp = rng.uniform(100, 500, size=8).astype(np.float32)
        s_amp = rng.uniform(2000, 8000, size=8).astype(np.float32)
        f.create_dataset("C_amp", data=c_amp)
        f.create_dataset("S_amp", data=s_amp)
        f.create_dataset("GenParticles.PDG", data=labels)
        f.create_dataset("E_gen", data=energies)
    file_path2 = tmp_path / "dummy2.h5"
    rng = np.random.default_rng(1234)
    with h5py.File(file_path2, "w") as f:
        for feature in HIT_FEATURES:
            data = rng.normal(size=(8, 32)).astype(np.float32)
            f.create_dataset(feature, data=data)
        labels = np.array(["211"] * 8, dtype="S")
        energies = rng.uniform(5, 50, size=8).astype(np.float32)
        c_amp = rng.uniform(100, 500, size=8).astype(np.float32)
        s_amp = rng.uniform(2000, 8000, size=8).astype(np.float32)
        f.create_dataset("C_amp", data=c_amp)
        f.create_dataset("S_amp", data=s_amp)
        f.create_dataset("GenParticles.PDG", data=labels)
        f.create_dataset("E_gen", data=energies)
    return file_path1, file_path2


@pytest.fixture
def pipeline_dataset(dummy_h5, stats_file):
    return DualReadoutEventDataset(
        files=dummy_h5,
        hit_features=HIT_FEATURES,
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        stat_file=str(stats_file),
        max_points=16,
        amp_sum_key="DRcalo3dHits.amplitude_sum",
        is_cherenkov_key="DRcalo3dHits.type",
        amp_sum_clip_percentile=None,
    )


def test_collate_and_models(pipeline_dataset):
    dataset = pipeline_dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_events)
    batch = next(iter(loader))
    if dataset.max_points is not None:
        assert batch["points"].shape[1] == dataset.max_points
    else:
        # When ``max_points`` is disabled we expect the batch to pad to the
        # maximum number of hits observed in the sampled events, which is the
        # largest number of ``True`` entries in the per-event mask.
        max_hits_in_batch = int(batch["mask"].sum(dim=1).max().item())
        assert batch["points"].shape[1] == max_hits_in_batch
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

    if importlib.util.find_spec("mamba_ssm") is not None and torch.cuda.is_available():
        device = torch.device("cuda")
        point_mamba = PointSetMamba(
            in_channels=batch["points"].shape[-1],
            summary_dim=summary_dim,
            num_classes=num_classes,
        ).to(device)
        cuda_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        outputs = point_mamba(cuda_batch)
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


def test_trainer_eval_computes_auc_by_default():
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            labels = batch["labels"]
            logits = torch.stack([1.0 - labels.float(), labels.float()], dim=1) + self.dummy * 0.0
            return ModelOutputs(
                logits=logits,
                energy=batch["energy"],
                log_sigma=None,
                direction=None,
                direction_log_sigma=None,
                extras={},
            )

    def collate_labels(items):
        labels = torch.tensor(items, dtype=torch.long)
        return {"labels": labels, "energy": torch.ones_like(labels, dtype=torch.float32)}

    model = ToyModel()
    loader = torch.utils.data.DataLoader([0, 1, 0, 1], batch_size=2, collate_fn=collate_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=1, log_every=1, use_amp=False),
    )

    metrics = trainer.evaluate(loader)

    assert "roc_auc" in metrics
    assert np.isfinite(metrics["roc_auc"])


def test_trainer_eval_computes_pid_checkpoint_score():
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            labels = batch["labels"]
            logits = torch.full((labels.shape[0], 4), -1.0, dtype=torch.float32)
            logits[torch.arange(labels.shape[0]), labels] = 2.0
            logits = logits + self.dummy * 0.0
            return ModelOutputs(
                logits=logits,
                energy=batch["energy"],
                log_sigma=None,
                direction=None,
                direction_log_sigma=None,
                extras={},
            )

    def collate_labels(items):
        labels = torch.tensor(items, dtype=torch.long)
        return {"labels": labels, "energy": torch.ones_like(labels, dtype=torch.float32)}

    model = ToyModel()
    loader = torch.utils.data.DataLoader([0, 1, 2, 3, 1, 2], batch_size=3, collate_fn=collate_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingConfig(
            epochs=1,
            log_every=1,
            use_amp=False,
            class_names=("e-", "gamma", "pi0", "pi+"),
        ),
    )

    metrics = trainer.evaluate(loader)

    assert "pi0_gamma_auc" in metrics
    assert "pid_checkpoint_score" in metrics
    assert np.isfinite(metrics["pid_checkpoint_score"])
    assert "macro_recall" in metrics


def test_trainer_eval_computes_pid_checkpoint_score_for_numeric_class_names():
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            labels = batch["labels"]
            logits = torch.full((labels.shape[0], 4), -1.0, dtype=torch.float32)
            logits[torch.arange(labels.shape[0]), labels] = 2.0
            logits = logits + self.dummy * 0.0
            return ModelOutputs(
                logits=logits,
                energy=batch["energy"],
                log_sigma=None,
                direction=None,
                direction_log_sigma=None,
                extras={},
            )

    def collate_labels(items):
        labels = torch.tensor(items, dtype=torch.long)
        return {"labels": labels, "energy": torch.ones_like(labels, dtype=torch.float32)}

    model = ToyModel()
    loader = torch.utils.data.DataLoader([0, 1, 2, 3, 1, 2], batch_size=3, collate_fn=collate_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingConfig(
            epochs=1,
            log_every=1,
            use_amp=False,
            class_names=("11", "22", "111", "211"),
        ),
    )

    metrics = trainer.evaluate(loader)

    assert "pi0_gamma_auc" in metrics
    assert "pid_checkpoint_score" in metrics
    assert np.isfinite(metrics["pid_checkpoint_score"])
    assert "macro_recall" in metrics


def test_trainer_monitor_does_not_fallback_to_loss_for_custom_metric():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=1, log_every=1, use_amp=False),
    )

    metrics = {"loss": 1.23, "accuracy": 0.75}

    assert trainer._select_monitor_value(metrics, "pid_checkpoint_score") is None
    assert trainer._select_monitor_value(metrics, "loss") == pytest.approx(1.23)


def test_amp_summary_does_not_require_type_when_c_amp_and_s_amp_exist():
    dataset = DualReadoutEventDataset.__new__(DualReadoutEventDataset)
    dataset.is_cherenkov_key = "DRcalo3dHits.type"
    dataset.amp_sum_key = "DRcalo3dHits.amplitude_sum"

    points = np.zeros((2, 4, 6), dtype=np.float32)
    index_map = {
        "DRcalo3dHits.amplitude_sum": 0,
        "DRcalo3dHits.time": 1,
        "DRcalo3dHits.time_end": 2,
        "DRcalo3dHits.position.x": 3,
        "DRcalo3dHits.position.y": 4,
        "DRcalo3dHits.position.z": 5,
    }
    c_amp = np.array([1.0, 2.0], dtype=np.float32)
    s_amp = np.array([3.0, 4.0], dtype=np.float32)

    summary = dataset._amp_summary(c_amp, s_amp, points, index_map)

    assert summary.shape == (2, 3)
    np.testing.assert_allclose(summary[:, 0], c_amp)
    np.testing.assert_allclose(summary[:, 1], s_amp)
    np.testing.assert_allclose(summary[:, 2], c_amp + s_amp)


def test_direction_loss_is_clamped_non_negative() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingConfig(
            epochs=1,
            log_every=1,
            use_amp=False,
            use_direction_regression=True,
            freeze_sigma=0,
        ),
    )
    batch = {
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "energy": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "direction": torch.zeros((2, 2), dtype=torch.float32),
    }
    outputs = ModelOutputs(
        logits=torch.zeros((2, 2), dtype=torch.float32),
        energy=torch.tensor([1.0, 2.0], dtype=torch.float32),
        direction=torch.zeros((2, 2), dtype=torch.float32),
        log_sigma=torch.zeros(2, dtype=torch.float32),
        direction_log_sigma=torch.full((2, 2), -3.0, dtype=torch.float32),
        extras={},
    )

    _, _, loss_dir = trainer._compute_losses(outputs, batch)

    assert float(loss_dir.item()) >= 0.0


def test_build_model_infers_mamba_backend_from_model_choice(monkeypatch):
    class RecordingMamba(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.backend = kwargs.get("backend")

        def forward(self, *args, **kwargs):  # pragma: no cover - not used in test
            raise NotImplementedError

    monkeypatch.setattr(run, "PointSetMamba", RecordingMamba)
    monkeypatch.setitem(run.MODEL_REGISTRY, "mamba", RecordingMamba)
    monkeypatch.setitem(run.MODEL_REGISTRY, "mamba2", RecordingMamba)

    class DummyRecord:
        def __init__(self) -> None:
            self.summary = torch.zeros(5)

    class DummyDataset:
        hit_features = ("f1", "f2", "f3")
        classes = ("a", "b")

        def __getitem__(self, index: int) -> DummyRecord:
            return DummyRecord()

    args = argparse.Namespace(
        model="mamba2",
        hidden_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        head_hidden=(128, 64),
        dropout=0.1,
        disable_summary=False,
        disable_uncertainty=False,
    )

    model = run.build_model(args, DummyDataset())

    assert isinstance(model, RecordingMamba)
    assert model.backend == "mamba2"


def test_build_model_can_disable_direction_uncertainty() -> None:
    class DummyRecord:
        def __init__(self) -> None:
            self.summary = torch.zeros(5)

    class DummyDataset:
        hit_features = ("f1", "f2", "f3")
        classes = ("a", "b")

        def __getitem__(self, index: int) -> DummyRecord:
            return DummyRecord()

    args = argparse.Namespace(
        model="mlp",
        hidden_dim=32,
        depth=2,
        num_heads=4,
        mlp_expansion=4.0,
        k_neighbors=16,
        head_hidden=(128, 64),
        dropout=0.1,
        disable_summary=False,
        disable_uncertainty=False,
        disable_direction_uncertainty=True,
        enable_direction_regression=True,
        direction_keys=("theta", "phi"),
    )

    model = run.build_model(args, DummyDataset())
    batch = {
        "points": torch.zeros((2, 4, 3), dtype=torch.float32),
        "mask": torch.ones((2, 4), dtype=torch.bool),
        "summary": torch.zeros((2, 5), dtype=torch.float32),
    }
    outputs = model(batch)

    assert outputs.direction is not None
    assert outputs.log_sigma is not None
    assert outputs.direction_log_sigma is None


def test_instance_name_sets_default_artifact_paths(dummy_h5, monkeypatch):
    original_parse_args = argparse.ArgumentParser.parse_args

    def _parse_args_with_output(self, *args, **kwargs):
        namespace = original_parse_args(self, *args, **kwargs)
        if not hasattr(namespace, "output_json"):
            setattr(namespace, "output_json", getattr(namespace, "output", None))
        return namespace

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _parse_args_with_output)

    args = run.parse_args([
        "--train_files",
        ",".join([str(d) for d in dummy_h5]),
        "--name",
        "experiment_a",
    ])

    expected_base = Path("save") / "experiment_a"
    assert args.history_json == expected_base / "history.json"
    assert args.checkpoint == expected_base / "checkpoint.pt"
    assert args.metrics_json == expected_base / "metrics.json"
    assert args.config_json == expected_base / "config.json"
    assert args.output_json == expected_base / "output.json"


def test_maybe_save_metrics(tmp_path):
    metrics = {"val": {"loss": 0.5, "accuracy": 0.9}}
    output_path = tmp_path / "metrics" / "results.json"

    run.maybe_save_metrics(metrics, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded == metrics


def test_maybe_save_config(tmp_path):
    config_path = tmp_path / "artifacts" / "config.json"
    args = argparse.Namespace(
        train_files=[Path("/data/train.h5")],
        history_json=config_path,
        checkpoint=None,
        metrics_json=None,
        config_json=config_path,
    )

    run.maybe_save_config(args, config_path)

    saved = json.loads(config_path.read_text())
    assert saved["train_files"] == ["/data/train.h5"]
    assert saved["config_json"] == str(config_path)
def test_trainer_evaluate_outputs(pipeline_dataset):
    dataset = pipeline_dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_events)
    summary_dim = dataset[0].summary.shape[-1]
    num_classes = len(dataset.classes)
    model = PointSetMLP(in_channels=dataset[0].points.shape[-1], summary_dim=summary_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=1, log_every=1, use_amp=False),
    )
    metrics, outputs = trainer.evaluate(loader, return_outputs=True)
    assert isinstance(metrics, dict)
    assert isinstance(outputs, list)
    assert len(outputs) == len(dataset)
    first_record = outputs[0]
    assert "event_index" in first_record  # This is the batch-local index
    assert "event_id" in first_record
    assert "energy_pred" in first_record
    assert "logits" in first_record


def test_build_datasets_auto_split(dummy_h5, stats_file):
    args = argparse.Namespace(
        train_files=dummy_h5,
        val_files=None,
        test_files=None,
        hit_features=HIT_FEATURES,
        pos_keys=(
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
        ),
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        stat_file=str(stats_file),
        max_points=16,
        enable_direction_regression=False,
        direction_keys=None,
        filter_amp=False,
        cache_size=16,
        max_cache_chunks=0,
        fixed_split_json=None,
        val_count=None,
        test_count=None,
        val_fraction=0.25,
        test_fraction=0.25, # this is not used
        split_seed=42,
        amp_sum_clip_percentile=None,
        amp_sum_key="DRcalo3dHits.amplitude_sum",
        is_cherenkov_key="DRcalo3dHits.type",
        pool=1,
        balance_train_files=False,
        train_limit=None,
        dataset_progress=False,
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


def test_build_datasets_fixed_split_ignores_mismatched_counts(dummy_h5, stats_file, tmp_path):
    split_path = tmp_path / "fixed_split.json"
    payload = {
        "total_events": 16,
        "train_indices": list(range(10)),
        "val_indices": [10, 11, 12],
        "test_indices": [13, 14, 15],
    }
    split_path.write_text(json.dumps(payload))

    args = argparse.Namespace(
        train_files=dummy_h5,
        val_files=None,
        test_files=None,
        hit_features=HIT_FEATURES,
        pos_keys=(
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
        ),
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        stat_file=str(stats_file),
        max_points=16,
        enable_direction_regression=False,
        direction_keys=None,
        filter_amp=False,
        cache_size=16,
        max_cache_chunks=0,
        val_fraction=0.25,
        test_fraction=0.25,
        val_count=1,
        test_count=1,
        split_seed=42,
        fixed_split_json=split_path,
        amp_sum_clip_percentile=None,
        amp_sum_key="DRcalo3dHits.amplitude_sum",
        is_cherenkov_key="DRcalo3dHits.type",
        pool=1,
        balance_train_files=False,
        train_limit=None,
        dataset_progress=False,
    )

    _, train_dataset, val_dataset, test_dataset = run.build_datasets(args)

    assert isinstance(train_dataset, Subset)
    assert isinstance(val_dataset, Subset)
    assert isinstance(test_dataset, Subset)
    assert len(train_dataset) == 10
    assert len(val_dataset) == 3
    assert len(test_dataset) == 3
    assert list(train_dataset.indices) == payload["train_indices"]
    assert list(val_dataset.indices) == payload["val_indices"]
    assert list(test_dataset.indices) == payload["test_indices"]


def test_build_datasets_projects_classwise_split_to_file_subset(tmp_path, stats_file):
    def _write_file(path: Path, label: str) -> None:
        rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
        with h5py.File(path, "w") as f:
            for feature in HIT_FEATURES:
                data = rng.normal(size=(4, 16)).astype(np.float32)
                f.create_dataset(feature, data=data)
            f.create_dataset("GenParticles.PDG", data=np.array([label] * 4, dtype="S"))
            f.create_dataset("E_gen", data=rng.uniform(5, 50, size=4).astype(np.float32))
            f.create_dataset("C_amp", data=rng.uniform(100, 500, size=4).astype(np.float32))
            f.create_dataset("S_amp", data=rng.uniform(2000, 8000, size=4).astype(np.float32))

    e_file = tmp_path / "e-_toy.h5"
    g_file = tmp_path / "gamma_toy.h5"
    p0_file = tmp_path / "pi0_toy.h5"
    pp_file = tmp_path / "pi+_toy.h5"
    _write_file(e_file, "11")
    _write_file(g_file, "22")
    _write_file(p0_file, "111")
    _write_file(pp_file, "211")

    split_path = tmp_path / "classwise_split.json"
    payload = {
        "schema": "classwise_fixed_split_v1",
        "total_events": 16,
        "class_order": ["e-", "gamma", "pi0", "pi+"],
        "per_class": {
            "e-": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 0, "offset_end": 4},
            "gamma": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 4, "offset_end": 8},
            "pi0": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 8, "offset_end": 12},
            "pi+": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 12, "offset_end": 16},
        },
        "train_indices": [0, 3, 4, 7, 8, 11, 12, 15],
        "val_indices": [1, 5, 9, 13],
        "test_indices": [2, 6, 10, 14],
    }
    split_path.write_text(json.dumps(payload))

    args = argparse.Namespace(
        train_files=[g_file, p0_file],
        val_files=None,
        test_files=None,
        hit_features=HIT_FEATURES,
        pos_keys=(
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
        ),
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        stat_file=str(stats_file),
        max_points=16,
        enable_direction_regression=False,
        direction_keys=None,
        filter_amp=False,
        cache_size=16,
        max_cache_chunks=0,
        val_fraction=0.25,
        test_fraction=0.25,
        val_count=999,
        test_count=999,
        split_seed=42,
        fixed_split_json=split_path,
        amp_sum_clip_percentile=None,
        amp_sum_key="DRcalo3dHits.amplitude_sum",
        is_cherenkov_key="DRcalo3dHits.type",
        pool=1,
        balance_train_files=False,
        train_limit=None,
        dataset_progress=False,
    )

    base_dataset, train_dataset, val_dataset, test_dataset = run.build_datasets(args)

    assert isinstance(train_dataset, Subset)
    assert isinstance(val_dataset, Subset)
    assert isinstance(test_dataset, Subset)
    assert len(train_dataset) == 4
    assert len(val_dataset) == 2
    assert len(test_dataset) == 2

    train_events = {tuple(base_dataset._indices[int(i)]) for i in train_dataset.indices}
    val_events = {tuple(base_dataset._indices[int(i)]) for i in val_dataset.indices}
    test_events = {tuple(base_dataset._indices[int(i)]) for i in test_dataset.indices}

    assert train_events == {(0, 0), (0, 3), (1, 0), (1, 3)}
    assert val_events == {(0, 1), (1, 1)}
    assert test_events == {(0, 2), (1, 2)}


def test_build_datasets_classwise_train_limit_is_per_class(tmp_path, stats_file):
    def _write_file(path: Path, label: str) -> None:
        rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
        with h5py.File(path, "w") as f:
            for feature in HIT_FEATURES:
                data = rng.normal(size=(4, 16)).astype(np.float32)
                f.create_dataset(feature, data=data)
            f.create_dataset("GenParticles.PDG", data=np.array([label] * 4, dtype="S"))
            f.create_dataset("E_gen", data=rng.uniform(5, 50, size=4).astype(np.float32))
            f.create_dataset("C_amp", data=rng.uniform(100, 500, size=4).astype(np.float32))
            f.create_dataset("S_amp", data=rng.uniform(2000, 8000, size=4).astype(np.float32))

    e_file = tmp_path / "e-_toy.h5"
    g_file = tmp_path / "gamma_toy.h5"
    p0_file = tmp_path / "pi0_toy.h5"
    pp_file = tmp_path / "pi+_toy.h5"
    _write_file(e_file, "11")
    _write_file(g_file, "22")
    _write_file(p0_file, "111")
    _write_file(pp_file, "211")

    split_path = tmp_path / "classwise_split.json"
    payload = {
        "schema": "classwise_fixed_split_v1",
        "total_events": 16,
        "class_order": ["e-", "gamma", "pi0", "pi+"],
        "per_class": {
            "e-": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 0, "offset_end": 4},
            "gamma": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 4, "offset_end": 8},
            "pi0": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 8, "offset_end": 12},
            "pi+": {"total": 4, "train": 2, "val": 1, "test": 1, "offset_start": 12, "offset_end": 16},
        },
        "train_indices": [0, 3, 4, 7, 8, 11, 12, 15],
        "val_indices": [1, 5, 9, 13],
        "test_indices": [2, 6, 10, 14],
    }
    split_path.write_text(json.dumps(payload))

    args = argparse.Namespace(
        train_files=[e_file, g_file, p0_file, pp_file],
        val_files=None,
        test_files=None,
        hit_features=HIT_FEATURES,
        pos_keys=(
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
        ),
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        stat_file=str(stats_file),
        max_points=16,
        enable_direction_regression=False,
        direction_keys=None,
        filter_amp=False,
        cache_size=16,
        max_cache_chunks=0,
        val_fraction=0.25,
        test_fraction=0.25,
        val_count=999,
        test_count=999,
        split_seed=42,
        fixed_split_json=split_path,
        amp_sum_clip_percentile=None,
        amp_sum_key="DRcalo3dHits.amplitude_sum",
        is_cherenkov_key="DRcalo3dHits.type",
        pool=1,
        balance_train_files=False,
        train_limit=1,
        dataset_progress=False,
    )

    base_dataset, train_dataset, val_dataset, test_dataset = run.build_datasets(args)

    assert isinstance(train_dataset, Subset)
    assert isinstance(val_dataset, Subset)
    assert isinstance(test_dataset, Subset)
    assert len(train_dataset) == 4
    assert len(val_dataset) == 4
    assert len(test_dataset) == 4

    train_events = [tuple(base_dataset._indices[int(i)]) for i in train_dataset.indices]
    train_file_ids = sorted(int(file_id) for file_id, _ in train_events)
    assert train_file_ids == [0, 1, 2, 3]
