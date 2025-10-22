"""Tests for :mod:`piddrc.data` that require NumPy/HDF5 support."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

from piddrc.data import DualReadoutEventDataset
from tests._helpers import MAX_POINTS, SUMMARY_DIM, TEST_CLASSES, TEST_HIT_FEATURES

_NUM_EVENTS = 6
_NUM_HITS = 32


def _create_dummy_file(path: Path, *, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as handle:
        for feature in TEST_HIT_FEATURES:
            data = rng.normal(size=(_NUM_EVENTS, _NUM_HITS)).astype(np.float32)
            handle.create_dataset(feature, data=data)
        labels = np.asarray(list(TEST_CLASSES) * (_NUM_EVENTS // len(TEST_CLASSES)), dtype="S")
        energies = rng.uniform(5.0, 50.0, size=_NUM_EVENTS).astype(np.float32)
        handle.create_dataset("particle_type", data=labels)
        handle.create_dataset("true_energy", data=energies)


def _build_dataset(tmp_path: Path) -> DualReadoutEventDataset:
    file_path = tmp_path / "events.h5"
    _create_dummy_file(file_path)
    return DualReadoutEventDataset(
        [str(file_path)],
        hit_features=TEST_HIT_FEATURES,
        label_key="particle_type",
        energy_key="true_energy",
        scintillation_key="S",
        cherenkov_key="C",
        depth_key="z",
        time_key="t",
        max_points=MAX_POINTS,
        cache_file_handles=False,
    )


@pytest.fixture()
def dataset(tmp_path: Path) -> DualReadoutEventDataset:
    return _build_dataset(tmp_path)


def test_class_discovery(dataset: DualReadoutEventDataset) -> None:
    assert dataset.classes == TEST_CLASSES
    assert dataset.class_to_index == {name: idx for idx, name in enumerate(TEST_CLASSES)}


def test_len_and_indices(dataset: DualReadoutEventDataset) -> None:
    assert len(dataset) == _NUM_EVENTS
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    assert first.event_id == (0, 0)
    assert last.event_id == (0, _NUM_EVENTS - 1)


def test_record_shapes(dataset: DualReadoutEventDataset) -> None:
    record = dataset[0]
    assert record.points.shape == (MAX_POINTS, len(TEST_HIT_FEATURES))
    assert record.summary.shape == (SUMMARY_DIM,)
    assert record.energy.shape == (1,)
    assert 0 <= record.label < len(TEST_CLASSES)


def test_summary_matches_default(dataset: DualReadoutEventDataset) -> None:
    record = dataset[0]
    points = record.points.numpy()
    expected = dataset._default_summary(points, dataset.feature_to_index)
    assert np.allclose(record.summary.numpy(), expected)


def test_multiple_indices_are_distinct(dataset: DualReadoutEventDataset) -> None:
    records = [dataset[i] for i in range(3)]
    event_ids = {record.event_id for record in records}
    assert len(event_ids) == len(records)
