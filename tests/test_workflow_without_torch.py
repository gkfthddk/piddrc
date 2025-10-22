"""Tests that dataset utilities operate with a minimal torch stub."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

_FEATURES = ("x", "y", "z", "S", "C", "t")
_CLASSES = ("electron", "pion")
_MAX_POINTS = 6
_SUMMARY_DIM = 8


class _FakeTensor:
    """NumPy-backed tensor that mimics a small subset of ``torch.Tensor``."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.array(array, copy=False)

    def numpy(self) -> np.ndarray:
        return np.array(self._array, copy=True)

    def __array__(self, dtype=None) -> np.ndarray:  # pragma: no cover - NumPy protocol
        if dtype is None:
            return self._array
        return self._array.astype(dtype)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._array)

    def __getitem__(self, key):
        return _FakeTensor(self._array[key])

    def __setitem__(self, key, value) -> None:
        self._array[key] = np.array(value, copy=False)

    def squeeze(self, axis=None) -> "_FakeTensor":
        return _FakeTensor(np.squeeze(self._array, axis=axis))

    def astype(self, dtype) -> "_FakeTensor":  # pragma: no cover - defensive
        return _FakeTensor(self._array.astype(dtype))

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self._array.copy())

    def item(self):  # pragma: no cover - defensive
        return self._array.item()

    @property
    def dtype(self):  # pragma: no cover - trivial
        return self._array.dtype

    @property
    def shape(self):  # pragma: no cover - trivial
        return self._array.shape

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_FakeTensor({self._array!r})"


def _resolve_dtype(requested):
    if requested is None:
        return None
    if requested is _FAKE_TORCH.bool:
        return np.bool_
    if requested is _FAKE_TORCH.long:
        return np.int64
    return requested


def _fake_from_numpy(array) -> _FakeTensor:
    return _FakeTensor(np.array(array, copy=False))


def _fake_zeros(shape, *, dtype=None) -> _FakeTensor:
    np_dtype = _resolve_dtype(dtype) or np.float32
    return _FakeTensor(np.zeros(shape, dtype=np_dtype))


def _fake_stack(tensors, dim: int = 0) -> _FakeTensor:
    arrays = [np.array(tensor) for tensor in tensors]
    return _FakeTensor(np.stack(arrays, axis=dim))


def _fake_tensor(data, *, dtype=None) -> _FakeTensor:
    np_dtype = _resolve_dtype(dtype)
    return _FakeTensor(np.array(data, dtype=np_dtype))


class _FakeDatasetBase:  # pragma: no cover - marker base class
    pass


_FAKE_TORCH = SimpleNamespace()
_FAKE_TORCH.Tensor = _FakeTensor
_FAKE_TORCH.from_numpy = _fake_from_numpy
_FAKE_TORCH.zeros = _fake_zeros
_FAKE_TORCH.stack = _fake_stack
_FAKE_TORCH.tensor = _fake_tensor
_FAKE_TORCH.bool = np.bool_
_FAKE_TORCH.long = np.int64

_FAKE_UTILS_DATA = SimpleNamespace(Dataset=_FakeDatasetBase)
_FAKE_UTILS = SimpleNamespace(data=_FAKE_UTILS_DATA)
_FAKE_TORCH.utils = _FAKE_UTILS


def _load_dataset_module(monkeypatch: pytest.MonkeyPatch):
    """Load ``piddrc.data`` with the fake torch module installed."""

    # Ensure no previously imported torch modules leak into the stub.
    for name in [name for name in sys.modules if name.startswith("torch")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setitem(sys.modules, "torch", _FAKE_TORCH)
    monkeypatch.setitem(sys.modules, "torch.utils", _FAKE_UTILS)
    monkeypatch.setitem(sys.modules, "torch.utils.data", _FAKE_UTILS_DATA)

    module_name = "piddrc.data_without_torch"
    data_path = Path(__file__).resolve().parents[1] / "piddrc" / "data.py"
    spec = importlib.util.spec_from_file_location(module_name, data_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _create_dummy_file(path: Path) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        for feature in _FEATURES:
            data = rng.normal(size=(4, 12)).astype(np.float32)
            handle.create_dataset(feature, data=data)
        labels = np.asarray([_CLASSES[i % len(_CLASSES)] for i in range(4)], dtype="S")
        energy = rng.uniform(5.0, 20.0, size=4).astype(np.float32)
        handle.create_dataset("particle_type", data=labels)
        handle.create_dataset("true_energy", data=energy)


def test_dataset_workflow_without_real_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_module = _load_dataset_module(monkeypatch)

    file_path = tmp_path / "events.h5"
    _create_dummy_file(file_path)

    dataset = data_module.DualReadoutEventDataset(
        [str(file_path)],
        hit_features=_FEATURES,
        label_key="particle_type",
        energy_key="true_energy",
        max_points=_MAX_POINTS,
        cache_file_handles=False,
    )

    assert len(dataset) == 4

    record = dataset[0]
    assert isinstance(record.points, _FakeTensor)
    assert record.points.shape == (_MAX_POINTS, len(_FEATURES))
    assert isinstance(record.summary, _FakeTensor)
    assert record.summary.shape == (_SUMMARY_DIM,)
    assert isinstance(record.energy, _FakeTensor)
    assert record.energy.shape == (1,)

    batch = data_module.collate_events([dataset[i] for i in range(2)])
    expected_keys = {"points", "mask", "summary", "labels", "energy", "event_id"}
    assert set(batch) == expected_keys
    for key in expected_keys:
        assert isinstance(batch[key], _FakeTensor)

    mask = batch["mask"].numpy()
    assert mask.dtype == np.bool_
    assert mask.shape == (2, _MAX_POINTS)
    assert mask.any(axis=1).all()

    labels = batch["labels"].numpy()
    assert labels.dtype == np.int64
    assert set(labels.tolist()) <= set(range(len(_CLASSES)))
