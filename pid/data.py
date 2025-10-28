"""Data loading utilities for dual-readout calorimeter studies.

The module implements an :class:`~torch.utils.data.Dataset` that reads
hit-level information from HDF5 files, performs lightweight feature
engineering, and prepares batched tensors that are suitable for
point-cloud style neural networks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - helpful message during unit tests
    raise ModuleNotFoundError("The h5py package is required to use the data utilities.") from exc

SummaryFn = Callable[[np.ndarray, Mapping[str, int]], np.ndarray]


def _decode_label(value: np.ndarray) -> str:
    """Decode a scalar value coming from HDF5 into a Python string."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        if value.dtype.kind in {"S", "a"}:  # byte string
            return value.astype(str)
        if value.dtype.kind in {"U"}:
            return str(value)
        return value.item()  # type: ignore[return-value]
    return str(value)


@dataclass
class EventRecord:
    """Container returned by :class:`DualReadoutEventDataset` for one event."""

    points: torch.Tensor
    summary: torch.Tensor
    label: int
    energy: torch.Tensor
    event_id: Tuple[int, int]


class DualReadoutEventDataset(Dataset):
    """Dataset that loads dual-readout calorimeter events from HDF5 files.

    Parameters
    ----------
    files:
        Sequence of paths to HDF5 files. Each file is expected to contain
        datasets named after the entries listed in ``hit_features``, the
        classification label ``label_key`` and the regression target
        ``energy_key``.
    hit_features:
        Feature names to read per hit. They must all share the shape
        ``(num_events, num_hits)`` within the HDF5 file.
    label_key:
        Dataset name containing the per-event classification labels.
    energy_key:
        Dataset name containing the regression target (true energy).
    max_points:
        If not ``None``, each event is randomly down-sampled to this
        number of points using a deterministic RNG seeded by the event
        index. The operation preserves ordering for reproducibility.
    scintillation_key / cherenkov_key:
        Names of the features that correspond to the scintillation (S)
        and Cherenkov (C) signals. They must be present in
        ``hit_features``. They are required for the default summary
        statistics.
    depth_key:
        Optional name of the feature describing the longitudinal depth of
        a hit (usually the ``z`` coordinate).
    time_key:
        Optional time-of-arrival feature. When provided, the mean
        arrival time is computed as part of the summary features.
    amp_sum_clip_percentile:
        When greater than zero, the dataset will estimate an adaptive
        clipping threshold for the per-event sum of
        ``amp_sum_key`` by sampling from the provided files and taking the
        requested percentile. Events whose total amplitude exceeds the
        resulting threshold (after multiplying by
        ``amp_sum_clip_multiplier``) or contains non-finite amplitudes
        will be removed from the dataset so that downstream consumers
        never observe the problematic entries. Set to ``0`` or ``None`` to
        disable the overflow-based filtering while keeping the
        non-finite guard in place.
    amp_sum_clip_multiplier:
        Multiplicative safety margin applied on top of the sampled
        percentile. This is useful to avoid clipping valid, but rare,
        events while still removing obvious outliers. Must be strictly
        positive.
    amp_sum_clip_sample_size:
        Maximum number of events to keep in the sampling reservoir when
        estimating the percentile. Larger values provide a more accurate
        threshold at the cost of additional I/O and memory. Set to a
        non-positive value to sample all events (not recommended for very
        large datasets).
    summary_fn:
        Custom callable that receives the point array of shape
        ``(num_hits, num_features)`` and a mapping from feature name to
        column index, and returns a 1-D numpy array of summary features.
        By default a physics-motivated summary vector consisting of
        ``[S_sum, C_sum, total, C_over_S, S_minus_C, depth_mean,
        depth_std, time_mean]`` is produced.
    class_names:
        Optional sequence defining the label ordering. When ``None`` the
        dataset determines the unique labels by scanning the input files.
    cache_file_handles:
        When ``True`` (default) the HDF5 files are kept open per worker
        process to avoid frequent reopen/close operations.
    balance_files:
        When ``True`` the dataset truncates each input file so that all files
        contribute the same number of events, matching the smallest available
        file after filtering. This is useful to avoid class imbalance when
        individual files correspond to different particle species.
    max_events:
        Optional cap on the number of events retained per file after applying
        filtering and (optional) balancing. When provided, each input file is
        truncated to at most this many events which is handy for quick smoke
        tests with a reduced sample size.
    """

    def __init__(
        self,
        files: Sequence[str],
        *,
        hit_features: Sequence[str],
        label_key: str,
        energy_key: str,
        max_points: Optional[int] = None,
        is_cherenkov_key: str = "DRcalo3dHits.type",
        amp_sum_key: str = "DRcalo3dHits.amplitude_sum",
        depth_key: Optional[str] = "z",
        time_key: Optional[str] = "t",
        amp_sum_clip_percentile: Optional[float] = 0.999,
        amp_sum_clip_multiplier: float = 1.5,
        amp_sum_clip_sample_size: int = 16_384,
        summary_fn: Optional[SummaryFn] = None,
        class_names: Optional[Sequence[str]] = None,
        cache_file_handles: bool = True,
        balance_files: bool = False,
        max_events: Optional[int] = None,
    ) -> None:
        if len(files) == 0:
            raise ValueError("At least one input file must be provided.")
        self.files: Tuple[str, ...] = tuple(files)
        self.hit_features: Tuple[str, ...] = tuple(hit_features)
        if len(self.hit_features) == 0:
            raise ValueError("hit_features must contain at least one feature name")
        self.feature_to_index: Dict[str, int] = {name: i for i, name in enumerate(self.hit_features)}

        self.label_key = label_key
        self.energy_key = energy_key
        self.max_points = max_points
        self.is_cherenkov_key = is_cherenkov_key
        self.amp_sum_key = amp_sum_key
        self.depth_key = depth_key
        self.time_key = time_key
        self.summary_fn = summary_fn or self._default_summary
        self.cache_file_handles = cache_file_handles
        self._balance_files = balance_files
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be non-negative when provided")
        self._max_events = max_events

        self._amp_sum_clip_percentile = amp_sum_clip_percentile
        self._amp_sum_clip_multiplier = amp_sum_clip_multiplier
        self._amp_sum_clip_sample_size = amp_sum_clip_sample_size
        self._amp_sum_threshold: float = float("inf")

        self._file_handles: MutableMapping[int, h5py.File] = {}
        self._indices: List[Tuple[int, int]] = []

        if class_names is not None:
            self.classes: Tuple[str, ...] = tuple(class_names)
        else:
            self.classes = self._discover_classes()
        self.class_to_index: Mapping[str, int] = {name: i for i, name in enumerate(self.classes)}

        self._amp_sum_threshold = self._estimate_amplitude_sum_threshold()
        print("amp_sum_threshold",self._amp_sum_threshold)
        self._build_index()

    # ------------------------------------------------------------------
    # Dataset protocol implementation
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._indices)

    def __getitem__(self, index: int) -> EventRecord:
        file_id, event_id = self._indices[index]
        with self._get_handle(file_id) as handle:
            return self._load_event(handle, file_id, event_id)

    def _load_event(self, handle: h5py.File, file_id: int, event_id: int) -> EventRecord:
        hits = [np.asarray(handle[feature][event_id], dtype=np.float32) for feature in self.hit_features]
        points = np.stack(hits, axis=-1)
        if self.max_points is not None and points.shape[0] > self.max_points:
            #rng = np.random.default_rng(seed=hash((file_id, event_id)) & 0xFFFF_FFFF)
            #choice = np.sort(rng.choice(points.shape[0], self.max_points, replace=False))
            points = points[:self.max_points]

        summary = self.summary_fn(points, self.feature_to_index)
        label_value = handle[self.label_key][event_id]
        label_name = _decode_label(label_value)
        if label_name not in self.class_to_index:
            raise KeyError(f"Encountered label '{label_name}' that is not part of the class map {self.class_to_index}")
        label = self.class_to_index[label_name]
        energy = np.asarray(handle[self.energy_key][event_id], dtype=np.float32)
        energy = np.atleast_1d(energy)

        return EventRecord(
            points=torch.from_numpy(points),
            summary=torch.from_numpy(summary.astype(np.float32)),
            label=label,
            energy=torch.from_numpy(energy),
            event_id=(file_id, event_id),
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _discover_classes(self) -> Tuple[str, ...]:
        labels: List[str] = []
        for file_path in self.files:
            with h5py.File(file_path, "r") as handle:
                data = handle[self.label_key][:]
                decoded = [_decode_label(value) for value in data]
                labels.extend(decoded)
        unique_labels = sorted(set(labels))
        return tuple(unique_labels)

    def _build_index(self) -> None:
        self._indices.clear()
        filter_amp = self.amp_sum_key in self.hit_features
        threshold = self._amp_sum_threshold

        per_file_indices: List[List[Tuple[int, int]]] = []

        for file_id, file_path in enumerate(self.files):
            with h5py.File(file_path, "r") as handle:
                num_events = len(handle[self.label_key])

                if not filter_amp or self.amp_sum_key not in handle:
                    file_indices = [(file_id, event_id) for event_id in range(num_events)]
                    per_file_indices.append(file_indices)
                    continue

                amp_dataset = handle[self.amp_sum_key]
                if num_events == 0:
                    per_file_indices.append([])
                    continue

                chunk_size = min(max(num_events // 32, 1), 1024)
                file_indices: List[Tuple[int, int]] = []
                for start in range(0, num_events, chunk_size):
                    stop = min(start + chunk_size, num_events)
                    amp_chunk = np.asarray(amp_dataset[start:stop], dtype=np.float64)
                    finite = np.isfinite(amp_chunk)
                    sanitized = np.where(finite, amp_chunk, 0.0)

                    if amp_chunk.ndim == 1:
                        finite_rows = finite
                        totals = sanitized.astype(np.float64)
                    else:
                        finite_rows = np.all(finite, axis=1)
                        totals = np.sum(sanitized, axis=1, dtype=np.float64)

                    keep_mask = np.logical_and(finite_rows, totals <= threshold)
                    if(len(keep_mask)!= np.sum(keep_mask)):
                        print("File",file_path,"Events",start,"to",stop,"removed",np.sum(~keep_mask),"out of",len(keep_mask))
                    for offset, keep in enumerate(keep_mask):
                        if keep:
                            file_indices.append((file_id, start + offset))

                per_file_indices.append(file_indices)

        if per_file_indices and self._balance_files:
            min_events = min(len(entries) for entries in per_file_indices)
            per_file_indices = [entries[:min_events] for entries in per_file_indices]

        if per_file_indices and self._max_events is not None:
            per_file_indices = [entries[: self._max_events] for entries in per_file_indices]

        self._indices = [entry for entries in per_file_indices for entry in entries]

    def _estimate_amplitude_sum_threshold(self) -> float:
        """Estimate a robust amplitude sum threshold for masking outliers."""

        percentile = self._amp_sum_clip_percentile
        if percentile is None or percentile <= 0:
            return float("inf")
        if not (0.0 < percentile < 1.0):
            raise ValueError("amp_sum_clip_percentile must lie in (0, 1)")

        if self.amp_sum_key not in self.feature_to_index:
            return float("inf")

        sample_size = self._amp_sum_clip_sample_size
        rng = np.random.default_rng(12345)
        reservoir: List[float] = []
        seen = 0

        for file_path in self.files:
            with h5py.File(file_path, "r") as handle:
                dataset = handle[self.amp_sum_key]
                if dataset.shape[0] == 0:
                    continue
                chunk_size = min(max(dataset.shape[0] // 32, 1), 1024)
                for start in range(0, dataset.shape[0], chunk_size):
                    stop = min(start + chunk_size, dataset.shape[0])
                    chunk = np.asarray(dataset[start:stop], dtype=np.float64)
                    chunk = np.where(np.isfinite(chunk), chunk, 0.0)
                    sums = np.sum(chunk, axis=1)
                    for value in sums:
                        seen += 1
                        if sample_size is not None and sample_size > 0:
                            if len(reservoir) < sample_size:
                                reservoir.append(float(value))
                            else:
                                j = int(rng.integers(0, seen))
                                if j < sample_size:
                                    reservoir[j] = float(value)
                        else:
                            reservoir.append(float(value))

        if not reservoir:
            return float("inf")

        baseline = float(np.quantile(np.asarray(reservoir, dtype=np.float64), percentile))
        if not np.isfinite(baseline):
            return float("inf")
        multiplier = self._amp_sum_clip_multiplier
        if multiplier <= 0:
            raise ValueError("amp_sum_clip_multiplier must be positive")
        threshold = baseline * multiplier
        if threshold <= 0:
            return float("inf")
        return threshold

    @contextmanager
    def _get_handle(self, file_id: int) -> Iterator[h5py.File]:
        if not self.cache_file_handles:
            with h5py.File(self.files[file_id], "r") as handle:
                yield handle
            return
        if file_id not in self._file_handles:
            self._file_handles[file_id] = h5py.File(self.files[file_id], "r")
        yield self._file_handles[file_id]

    def close(self) -> None:
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary feature engineering
    # ------------------------------------------------------------------
    def _default_summary(self, points: np.ndarray, index_map: Mapping[str, int]) -> np.ndarray:
        """Compute physics-motivated summary features for one event."""

        is_cherenkov = points[:, index_map[self.is_cherenkov_key]].astype(bool)
        c = points[is_cherenkov, index_map[self.amp_sum_key]]
        s = points[~is_cherenkov, index_map[self.amp_sum_key]]
        c_sum = float(np.sum(c))
        s_sum = float(np.sum(s))
        total_sum = c_sum+s_sum
        ratio = float(c_sum / (s_sum + 1e-6))
        s_minus_c = float(s_sum - c_sum)

        stats: List[float] = [s_sum, c_sum, total_sum, ratio, s_minus_c]

        if self.depth_key and self.depth_key in index_map:
            z_c = points[is_cherenkov, index_map[self.depth_key]]
            if c_sum > 0:
                depth_mean_c = float(np.average(z_c, weights=c))
                depth_std_c = float(np.sqrt(np.average((z_c - depth_mean_c) ** 2, weights=c)))
            else:
                depth_mean_c = float(np.mean(z_c))
                depth_std_c = float(np.std(z_c))
            stats.extend([depth_mean_c, depth_std_c])
            z_s = points[~is_cherenkov, index_map[self.depth_key]]
            if s_sum > 0:
                depth_mean_s = float(np.average(z_s, weights=s))
                depth_std_s = float(np.sqrt(np.average((z_s - depth_mean_s) ** 2, weights=s)))
            else:
                depth_mean_s = float(np.mean(z_s))
                depth_std_s = float(np.std(z_s))
            stats.extend([depth_mean_s, depth_std_s])
        else:
            stats.extend([0.0, 0.0, 0.0, 0.0])

        if self.time_key and self.time_key in index_map:
            t_c = points[is_cherenkov, index_map[self.time_key]]
            if c_sum > 0:
                time_mean_c = float(np.average(t_c, weights=c))
            else:
                time_mean_c = float(np.mean(t_c))
            stats.append(time_mean_c)
            t_s = points[~is_cherenkov, index_map[self.time_key]]
            if s_sum > 0:
                time_mean_s = float(np.average(t_s, weights=s))
            else:
                time_mean_s = float(np.mean(t_s))
            stats.append(time_mean_s)
        else:
            stats.extend([0.0, 0.0])

        return np.asarray(stats, dtype=np.float32)


def collate_events(batch: Sequence[EventRecord]) -> Dict[str, torch.Tensor]:
    """Custom ``DataLoader`` collation for variable-length point clouds."""

    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch")

    max_hits = max(record.points.shape[0] for record in batch)
    feature_dim = batch[0].points.shape[1]
    batch_size = len(batch)

    points = torch.zeros(batch_size, max_hits, feature_dim, dtype=batch[0].points.dtype)
    mask = torch.zeros(batch_size, max_hits, dtype=torch.bool)
    summary = torch.stack([record.summary for record in batch], dim=0)
    labels = torch.tensor([record.label for record in batch], dtype=torch.long)
    energy = torch.stack([record.energy for record in batch], dim=0).squeeze(-1)
    event_id = torch.tensor([record.event_id for record in batch], dtype=torch.long)

    for i, record in enumerate(batch):
        num_hits = record.points.shape[0]
        points[i, :num_hits] = record.points
        mask[i, :num_hits] = True

    return {
        "points": points,
        "mask": mask,
        "summary": summary,
        "labels": labels,
        "energy": energy,
        "event_id": event_id,
    }


__all__ = ["DualReadoutEventDataset", "collate_events", "EventRecord"]
