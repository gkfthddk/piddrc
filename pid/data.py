"""Data loading utilities for dual-readout calorimeter studies.

The module implements an :class:`~torch.utils.data.Dataset` that reads
hit-level information from HDF5 files, performs lightweight feature
engineering, and prepares batched tensors that are suitable for
point-cloud style neural networks.
"""

from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None
from tqdm.auto import tqdm
try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - helpful message during unit tests
    raise ModuleNotFoundError("The h5py package is required to use the data utilities.") from exc

SummaryFn = Callable[[np.ndarray, Mapping[str, int]], np.ndarray]
AmpAwareSummaryFn = Callable[[float, float, np.ndarray, Mapping[str, int]], np.ndarray]
SummaryFnType = Union[SummaryFn, AmpAwareSummaryFn]


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
    pos: torch.Tensor | None = None


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
    stat_file:
        Path to a YAML file containing per-feature statistics used for
        normalization and amplitude threshold estimation.
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
        Custom callable that receives the Cherenkov and scintillation
        amplitude summaries (as floats), the point array of shape
        ``(num_hits, num_features)`` and a mapping from feature name to
        column index, and returns a 1-D numpy array of summary
        features. Legacy callables that only accept ``(points,
        index_map)`` are still supported and the amplitude arguments are
        ignored. By default a physics-motivated summary vector
        consisting of ``[S_sum, C_sum, total, C_over_S, S_minus_C,
        depth_mean, depth_std, time_mean]`` is produced.
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
    progress:
        When ``False`` suppress informational progress messages that are printed
        while the dataset scans the input files during initialization.
    """

    def __init__(
        self,
        files: Sequence[str],
        *,
        hit_features: Sequence[str],
        label_key: str,
        energy_key: str,
        stat_file: str,
        max_points: Optional[int] = None,
        pool: int = 1,
        is_cherenkov_key: str = "DRcalo3dHits.type",
        amp_sum_key: str = "DRcalo3dHits.amplitude_sum",
        pos_keys: Optional[List[str]] = ["DRcalo3dHits.position.x", "DRcalo3dHits.position.y", "DRcalo3dHits.position.z","DRcalo3dHits.time"],
        amp_sum_clip_percentile: Optional[float] = 0.999,
        amp_sum_clip_multiplier: float = 1.5,
        amp_sum_clip_sample_size: int = 16_384,
        summary_fn: Optional[SummaryFnType] = None,
        class_names: Optional[Sequence[str]] = None,
        cache_file_handles: bool = True,
        balance_files: bool = False,
        max_events: Optional[int] = None,
        progress: bool = True,
    ) -> None:
        if len(files) == 0:
            raise ValueError("At least one input file must be provided.")
        self.files: Tuple[str, ...] = tuple(files)
        self.stat_file = os.fspath(stat_file)
        self.hit_features: Tuple[str, ...] = tuple(hit_features)
        if len(self.hit_features) == 0:
            raise ValueError("hit_features must contain at least one feature name")
        self.feature_to_index: Dict[str, int] = {name: i for i, name in enumerate(self.hit_features)}
        self.feature_stat: Dict[str, Mapping[str, float]] = {}
        self.feature_max: Dict[str, float] = {}
        if yaml is None:
            with open(self.stat_file, "r", encoding="utf-8") as yf:
                try:
                    loaded = json.load(yf)
                except json.JSONDecodeError as exc:
                    raise ModuleNotFoundError(
                        "PyYAML is required to load stat_file. Install PyYAML to read YAML content."
                    ) from exc
        else:
            with open(self.stat_file, "r", encoding="utf-8") as yf:
                loaded = yaml.safe_load(yf) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("stat_file must contain a mapping from feature name to statistics")
        channel_stat: Mapping[str, Any] | None = loaded
        for name in self.hit_features:
            stats = channel_stat.get(name, {}) if isinstance(channel_stat, Mapping) else {}
            if not isinstance(stats, Mapping):
                stats = {}
            self.feature_stat[name] = stats
            max_val = float(stats.get("max", 0.0)) if "max" in stats else 0.0
            min_val = float(stats.get("min", 0.0)) if "min" in stats else 0.0
            scale = max(abs(max_val), abs(min_val))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            self.feature_max[name] = scale
        self._channel_stat = channel_stat

        self.label_key = label_key
        self.energy_key = energy_key
        self.max_points = max_points
        if(pool > 1):
            is_cherenkov_key=is_cherenkov_key.replace("DRcalo3dHits",f"DRcalo3dHits{pool}")
            amp_sum_key=amp_sum_key.replace("DRcalo3dHits",f"DRcalo3dHits{pool}")
            pos_keys=[key.replace("DRcalo3dHits",f"DRcalo3dHits{pool}") for key in pos_keys]

        self.is_cherenkov_key = is_cherenkov_key
        self.amp_sum_key = amp_sum_key
        self.pos_keys = pos_keys
        if summary_fn is None:
            self.summary_fn: AmpAwareSummaryFn = self._amp_summary
        else:
            self.summary_fn = self._coerce_summary_fn(summary_fn)
        self.cache_file_handles = cache_file_handles
        self._balance_files = balance_files
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be non-negative when provided")
        self._max_events = max_events

        self._progress_enabled = progress

        self._log(f"Initializing dataset from {len(self.files)} file(s)")

        self._amp_sum_clip_percentile = amp_sum_clip_percentile
        self._amp_sum_clip_multiplier = amp_sum_clip_multiplier
        self._amp_sum_clip_sample_size = amp_sum_clip_sample_size
        self._amp_sum_threshold: float = float("inf")

        self._file_handles: MutableMapping[int, h5py.File] = {}
        self._indices: List[Tuple[int, int]] = []

        if class_names is not None:
            self.classes: Tuple[str, ...] = tuple(class_names)
        else:
            self._log("Discovering label set across input files")
            self.classes = self._discover_classes()
        self.class_to_index: Mapping[str, int] = {name: i for i, name in enumerate(self.classes)}

        self._log(f"Discovered {len(self.classes)} class(es)")

        self._log("Estimating amplitude-sum threshold")
        #self._amp_sum_threshold = self._estimate_amplitude_sum_threshold()
        stats = None
        if isinstance(self._channel_stat, Mapping):
            stats = self._channel_stat.get("S_amp")
        if isinstance(stats, Mapping):
            max_value = stats.get("max")
            if isinstance(max_value, (int, float)) and np.isfinite(max_value):
                self._amp_sum_threshold = float(max_value) * 1.5
            else:
                self._amp_sum_threshold = self._estimate_amplitude_sum_threshold()
        else:
            self._amp_sum_threshold = self._estimate_amplitude_sum_threshold()
        if np.isfinite(self._amp_sum_threshold):
            self._log(f"  Using threshold {self._amp_sum_threshold:,.3f}")
        else:
            self._log("  No finite threshold computed (disabled or insufficient data)")
        self._log("Building event index")
        self._build_index()
        self._log(f"Index contains {len(self._indices):,} event(s)")

    # ------------------------------------------------------------------
    # Summary function helpers
    # ------------------------------------------------------------------
    def _coerce_summary_fn(self, summary_fn: SummaryFnType) -> AmpAwareSummaryFn:
        """Wrap legacy summary functions to accept amplitude scalars."""

        try:
            parameter_count = len(inspect.signature(summary_fn).parameters)
        except (TypeError, ValueError):
            parameter_count = None

        if parameter_count == 2:

            def _wrapped(
                c_amp: float,
                s_amp: float,
                points: np.ndarray,
                index_map: Mapping[str, int],
            ) -> np.ndarray:
                return summary_fn(points, index_map)  # type: ignore[misc]

            return _wrapped

        return cast(AmpAwareSummaryFn, summary_fn)

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
        hits = [np.asarray(handle[feature][event_id]/self.feature_max[feature], dtype=np.float32) for feature in self.hit_features]
        points = np.stack(hits, axis=-1)
        pos_tensor: torch.Tensor | None = None
        if self.pos_keys:
            coord_indices: List[int] = []
            for key in self.pos_keys:
                if key in self.feature_to_index:
                    coord_indices.append(self.feature_to_index[key])
                if len(coord_indices) >= 3:
                    break
            if len(coord_indices) >= 3:
                coords = np.asarray(points[:, coord_indices[:3]], dtype=np.float32).copy()
                pos_tensor = torch.from_numpy(coords)
        if self.max_points is not None and points.shape[0] > self.max_points:
            #rng = np.random.default_rng(seed=hash((file_id, event_id)) & 0xFFFF_FFFF)
            #choice = np.sort(rng.choice(points.shape[0], self.max_points, replace=False))
            points = points[:self.max_points]
            if pos_tensor is not None:
                pos_tensor = pos_tensor[: self.max_points]

        # Some lightweight conversion helpers so tests using minimal HDF5 stubs do
        # not have to materialize the pre-computed amplitude summaries. When the
        # datasets are absent we fall back to zeros and let the summary function
        # derive the values from the hit-level features instead.
        def _read_optional(dataset_name: str) -> float:
            if dataset_name not in handle:
                return 0.0
            value = handle[dataset_name][event_id]
            array = np.asarray(value, dtype=np.float32)
            if array.size == 0:
                return 0.0
            return float(array.reshape(-1)[0])

        c_amp = _read_optional("C_amp")
        s_amp = _read_optional("S_amp")

        summary = self.summary_fn(c_amp, s_amp, points, self.feature_to_index)
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
            pos=pos_tensor,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _discover_classes(self) -> Tuple[str, ...]:
        labels: List[str] = []
        for file_path in self.files:
            self._log(f"  Reading labels from {file_path}")
            with h5py.File(file_path, "r") as handle:
                if(self._max_events is None):
                    data = handle[self.label_key][:]
                else:
                    data = handle[self.label_key][:self._max_events]
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
            self._log(
                f"  Indexing file {file_id + 1}/{len(self.files)}: {file_path}"
            )
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

                # If max_events is set, we can limit the number of chunks to scan
                scan_num_events = num_events
                if self._max_events is not None:
                    # Estimate the number of events to scan. Add a buffer.
                    scan_num_events = min(num_events, int(self._max_events * 1.2))

                chunk_iterator = range(0, scan_num_events, chunk_size)
                if self._progress_enabled:
                    chunk_iterator = tqdm(
                        chunk_iterator,
                        desc="    Scanning chunks",
                        unit="chunk",
                        leave=False,
                        total=len(chunk_iterator),
                    )
                for start in chunk_iterator:
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
                    if len(keep_mask) != int(np.sum(keep_mask)):
                        pass
                    for offset, keep in enumerate(keep_mask):
                        if keep:
                            file_indices.append((file_id, start + offset))
                    if self._max_events is not None and len(file_indices) >= self._max_events:
                        break

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
            self._log(f"  Sampling amplitude sums from {file_path}")
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
            self._log("  No samples collected while estimating threshold")
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

    def _log(self, message: str) -> None:
        if not self._progress_enabled:
            return
        print(f"[DualReadoutEventDataset] {message}", flush=True)

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
    def _amp_summary(self, C_amp, S_amp, points: np.ndarray, index_map: Mapping[str, int]) -> np.ndarray:
        """Compute simple amplitude-based summary features for one event."""

        is_cherenkov = points[:, index_map[self.is_cherenkov_key]].astype(bool)
        if(C_amp>0):
            c_sum = float(C_amp)
        else:
            c = points[is_cherenkov, index_map[self.amp_sum_key]]
            c_sum = float(np.sum(c))
        if(S_amp>0):
            s_sum = float(S_amp)
        else:
            s = points[~is_cherenkov, index_map[self.amp_sum_key]]
            s_sum = float(np.sum(s))
        stats= [s_sum, c_sum, s_sum + c_sum]
        return np.asarray(stats, dtype=np.float32)
    
    def _dist_summary(self, C_amp, S_amp, points: np.ndarray, index_map: Mapping[str, int]) -> np.ndarray:
        """Compute physics-motivated summary features for one event."""

        is_cherenkov = points[:, index_map[self.is_cherenkov_key]].astype(bool)
        c = points[is_cherenkov, index_map[self.amp_sum_key]]
        s = points[~is_cherenkov, index_map[self.amp_sum_key]]
        if(C_amp>0):
            c_sum = float(C_amp)
        else:
            c_sum = float(np.sum(c))
        if(S_amp>0):
            s_sum = float(S_amp)
        else:
            s_sum = float(np.sum(s))
        total_sum = c_sum+s_sum
        ratio = float(c_sum / (s_sum + 1e-6))
        s_minus_c = float(s_sum - c_sum)

        stats: List[float] = [s_sum, c_sum, total_sum, ratio, s_minus_c]

        for key in self.pos_keys:
            if key and key in index_map:
                coord_c = points[is_cherenkov, index_map[key]]
                if c_sum > 0:
                    mean_c = float(np.average(coord_c, weights=c))
                    std_c = float(np.sqrt(np.average((coord_c - mean_c) ** 2, weights=c)))
                    peak_c = float(coord_c[np.argmax(c)])
                else:
                    mean_c = float(np.mean(coord_c))
                    std_c = float(np.std(coord_c))
                    peak_c = mean_c
                coord_s = points[~is_cherenkov, index_map[key]]
                if s_sum > 0:
                    mean_s = float(np.average(coord_s, weights=s))
                    std_s = float(np.sqrt(np.average((coord_s - mean_s) ** 2, weights=s)))
                    peak_s = float(coord_s[np.argmax(s)])
                else:
                    mean_s = float(np.mean(coord_s))
                    std_s = float(np.std(coord_s))
                    peak_s = mean_s
                stats.extend([mean_c, std_c, peak_c, mean_s, std_s, peak_s])
            else:
                stats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
    pos_template = next((record.pos for record in batch if record.pos is not None), None)
    pos = None
    if pos_template is not None:
        pos = torch.zeros(
            batch_size,
            max_hits,
            pos_template.shape[-1],
            dtype=pos_template.dtype,
        )

    for i, record in enumerate(batch):
        num_hits = record.points.shape[0]
        points[i, :num_hits] = record.points
        mask[i, :num_hits] = True
        if pos is not None and record.pos is not None:
            pos[i, :num_hits] = record.pos

    collated = {
        "points": points,
        "mask": mask,
        "summary": summary,
        "labels": labels,
        "energy": energy,
        "event_id": event_id,
    }
    if pos is not None:
        collated["pos"] = pos
    return collated


__all__ = ["DualReadoutEventDataset", "collate_events", "EventRecord"]
