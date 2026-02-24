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
from collections import OrderedDict
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


def _trim_left_to_cid_boundary(cid: torch.Tensor, cut: int) -> int:
    """Move ``cut`` to the previous readout boundary so no cid is partially kept."""
    if cut <= 0:
        return 0
    if cut >= cid.shape[0]:
        return int(cut)
    if cid[cut - 1] != cid[cut]:
        return int(cut)
    boundary_diff = torch.nonzero(cid[:cut] != cid[cut - 1], as_tuple=False)
    if boundary_diff.numel() == 0:
        return 0
    return int(boundary_diff[-1].item()) + 1


def _adaptive_keep_fraction(
    energy_value: float,
    min_frac: float,
    max_frac: float,
    e_min: float,
    e_max: float,
) -> float:
    """Energy-adaptive keep fraction in [min_frac, max_frac], linear in log10(E)."""
    if not np.isfinite(energy_value) or energy_value <= 0:
        return max_frac
    if e_max <= e_min:
        return max_frac

    lo = min(min_frac, max_frac)
    hi = max(min_frac, max_frac)
    log_e = np.log10(max(energy_value, e_min))
    t = (log_e - np.log10(e_min)) / (np.log10(e_max) - np.log10(e_min))
    t = float(np.clip(t, 0.0, 1.0))
    frac = hi - (hi - lo) * t
    return float(np.clip(frac, lo, hi))


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
    direction: torch.Tensor | None
    event_id: Tuple[int, int]
    pos: torch.Tensor | None = None
    cid: torch.Tensor | None = None


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
    cache_size:
        Number of events to read per on-disk chunk when filling the in-memory
        cache. Larger values amortize disk access at the cost of more temporary
        memory per cached chunk.
    max_cache_chunks:
        Maximum number of cached chunks to retain per worker. Set to ``None``
        (the default) to disable eviction and keep all loaded chunks, matching
        the original unbounded caching behaviour.
    """

    def __init__(
        self,
        files: Sequence[str],
        *,
        hit_features: Sequence[str],
        label_key: str,
        energy_key: str,
        direction_keys: Optional[Sequence[str]] = None,
        stat_file: str,
        max_points: Optional[int] = None,
        pool: int = 1,
        is_cherenkov_key: str = "DRcalo3dHits.type",
        amp_sum_key: str = "DRcalo3dHits.amplitude_sum",
        cid_key: str = "DRcalo3dHits.cellID",
        pos_keys: Optional[List[str]] = [
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
        ],
        amp_sum_clip_percentile: Optional[float] = 0.999,
        amp_sum_clip_multiplier: float = 1.5,
        amp_sum_clip_sample_size: int = 16_384,
        summary_fn: Optional[SummaryFnType] = None,
        class_names: Optional[Sequence[str]] = None,
        balance_files: bool = False,
        max_events: Optional[int] = None,
        progress: bool = True,
        cache_size: int = 1024,
        max_cache_chunks: Optional[int] = None,
    ) -> None:
        if len(files) == 0:
            raise ValueError("At least one input file must be provided.")
        store = "/store/ml/dual-readout/h5s"

        def _resolve_path(path_like: str) -> str:
            raw = os.fspath(path_like)
            # Backward-compatibility guard: older code could accidentally build
            # '/store/...//tmp/...' when joining store with an absolute tmp path.
            if raw.startswith(f"{store}//"):
                raw = raw[len(store) + 1 :]
            if os.path.isabs(raw) or os.path.exists(raw):
                return raw
            return os.path.join(store, raw)

        self.files: Tuple[str, ...] = tuple(_resolve_path(f) for f in files)
        self.stat_file = _resolve_path(stat_file)
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
        self.direction_keys: Tuple[str, ...] = tuple(direction_keys or ())
        self.max_points = max_points
        if(pool > 1):
            is_cherenkov_key=is_cherenkov_key.replace("DRcalo3dHits",f"DRcalo3dHits{pool}")
            amp_sum_key=amp_sum_key.replace("DRcalo3dHits",f"DRcalo3dHits{pool}")
            cid_key=cid_key.replace("DRcalo3dHits",f"DRcalo3dHits{pool}")
            pos_keys=[key.replace("DRcalo3dHits",f"DRcalo3dHits{pool}") for key in pos_keys]

        self.is_cherenkov_key = is_cherenkov_key
        self.amp_sum_key = amp_sum_key
        self.cid_key = cid_key
        self.pos_keys = pos_keys
        if summary_fn is None:
            self.summary_fn: AmpAwareSummaryFn = self._amp_summary
        else:
            self.summary_fn = self._coerce_summary_fn(summary_fn)
        self.cache_size = cache_size
        if max_cache_chunks is not None:
            if max_cache_chunks <= 0:
                raise ValueError("max_cache_chunks must be positive or None to disable caching limits")
            self._max_cached_chunks: Optional[int] = int(max_cache_chunks)
        else:
            self._max_cached_chunks = None
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

        self._indices: List[Tuple[int, int]] = []
        self._cache: MutableMapping[int, OrderedDict[Tuple[int, int], Mapping[str, Any]]] = {}

        if class_names is not None:
            self.classes: Tuple[str, ...] = tuple(class_names)
        else:
            self._log("Discovering label set across input files")
            self.classes = self._discover_classes()
        self.class_to_index: Mapping[str, int] = {name: i for i, name in enumerate(self.classes)}

        self._log(f"Discovered {len(self.classes)} class(es)")

        self._log("Estimating amplitude-sum threshold")
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
        chunk_start = (event_id // self.cache_size) * self.cache_size
        cache_key = (file_id, chunk_start)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if worker_id not in self._cache:
            # Initialize cache for this worker (this is safe in multiprocessing)
            self._cache[worker_id] = OrderedDict()
        worker_cache = self._cache[worker_id]
        # --- 2. Load chunk into cache if needed ---
        if cache_key not in worker_cache:
            try:
                with h5py.File(self.files[file_id], "r") as handle:
                    # Store in *worker-local* cache
                    worker_cache[cache_key] = self._get_chunk(handle, chunk_start)
                    worker_cache.move_to_end(cache_key)
                    if self._max_cached_chunks is not None:
                        while len(worker_cache) > self._max_cached_chunks:
                            worker_cache.popitem(last=False)
                # File handle is now closed. This is safe.
            except Exception as e:
                self._log(f"ERROR: Worker {worker_id} failed to load chunk {cache_key} from {self.files[file_id]}: {e}")
                # Fallback: get a different item (use modulo for safety)
                return self.__getitem__((index + 1) % len(self))
        else:
            worker_cache.move_to_end(cache_key)
        # --- 3. Retrieve event from cache ---
        chunk_data = worker_cache[cache_key]
        local_index = event_id - chunk_start
        
        try:
            # Get the specific event data
            return self._load_event(chunk_data, local_index, file_id, event_id)
        except IndexError as e:
            self._log(f"ERROR: Worker {worker_id} had an index error. "
                      f"Attempted local_index {local_index} in chunk {cache_key} with shape {chunk_data['points'].shape}. "
                      f"Original index {index}. Error: {e}")
            return self.__getitem__((index + 1) % len(self))

    def _get_chunk(self, handle: h5py.File, chunk_start: int) -> Mapping[str, Any]:
        start = chunk_start
        stop = min(start + self.cache_size, len(handle[self.label_key]))
        hits = [np.asarray(handle[feature][start:stop,:self.max_points]/self.feature_max[feature], dtype=np.float32) for feature in self.hit_features]
        cid=handle[self.cid_key][start:stop,:self.max_points]
        points = np.stack(hits, axis=-1)

        def _read_optional(dataset_name: str) -> float:
            if dataset_name not in handle:
                return None
            value = handle[dataset_name][start:stop]
            array = np.asarray(value, dtype=np.float32)
            if array.size == 0:
                return None
            return array

        c_amp = _read_optional("C_amp")
        s_amp = _read_optional("S_amp")
        summary = self.summary_fn(c_amp, s_amp, points, self.feature_to_index)
        label_value = handle[self.label_key][start:stop]
        energy = handle[self.energy_key][start:stop]
        direction = None
        if self.direction_keys:
            direction_components: List[np.ndarray] = []
            for key in self.direction_keys:
                if key not in handle:
                    raise KeyError(
                        f"Direction key '{key}' not found in file '{handle.filename}'. "
                        "Disable direction regression or choose valid keys."
                    )
                component = np.asarray(handle[key][start:stop], dtype=np.float32).reshape(-1)
                direction_components.append(component)
            direction = np.stack(direction_components, axis=-1)
        chunk={
            "start": start, 
            "points": points,
            "cid": cid,
            "summary": summary,
            "label_value": label_value,
            "energy": energy,
            "direction": direction,
            }   
        return chunk

    def _load_event(self, chunk: Mapping[str, Any], local_index: int, file_id: int, event_id: int) -> EventRecord:
        points = chunk["points"][local_index]
        cid = chunk["cid"][local_index]
        label_value = chunk["label_value"][local_index]
        summary = chunk["summary"][local_index]
        energy = chunk["energy"][local_index]
        direction = chunk["direction"][local_index] if chunk.get("direction") is not None else None

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

        # Some lightweight conversion helpers so tests using minimal HDF5 stubs do
        # not have to materialize the pre-computed amplitude summaries. When the
        # datasets are absent we fall back to zeros and let the summary function

        label_name = _decode_label(label_value)
        if label_name not in self.class_to_index:
            raise KeyError(f"Encountered label '{label_name}' that is not part of the class map {self.class_to_index}")
        label = self.class_to_index[label_name]
        
        energy = np.atleast_1d(energy)
        return EventRecord(
            points=torch.from_numpy(points),
            summary=torch.from_numpy(summary.astype(np.float32)),
            label=label,
            energy=torch.from_numpy(energy),
            direction=torch.from_numpy(np.atleast_1d(direction)) if direction is not None else None,
            event_id=(file_id, event_id),
            pos=pos_tensor,
            cid=torch.from_numpy(np.asarray(cid)),
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _discover_classes(self) -> Tuple[str, ...]:
        labels: List[str] = []
        for file_path in self.files:
            with h5py.File(file_path, "r") as handle:
                if(self._max_events is None):
                    data = handle[self.label_key][:1]
                else:
                    data = handle[self.label_key][:1]
                decoded = [_decode_label(value) for value in set(data)]
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
                
                # If max_events is set, we can limit the number of chunks to scan
                scan_num_events = num_events
                if self._max_events is not None:
                    # Estimate the number of events to scan. Add a buffer.
                    scan_num_events = min(num_events, self._max_events)

                if not filter_amp or self.amp_sum_key not in handle:
                    file_indices = [(file_id, event_id) for event_id in range(scan_num_events)]
                    per_file_indices.append(file_indices)
                    continue
                
                if scan_num_events == 0:
                    per_file_indices.append([])
                    continue

                chunk_size = min(max(scan_num_events // 32, 1), 1024)
                file_indices: List[Tuple[int, int]] = []
                
                #amp_dataset = handle[self.amp_sum_key][:scan_num_events]
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
                    stop = min(start + chunk_size, scan_num_events)
                    amp_chunk = np.asarray(handle[self.amp_sum_key][start:stop], dtype=np.float64)
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
        rng = np.random.default_rng(12345)
        cursor = np.zeros(len(per_file_indices), dtype=int)
        labels = list(range(len(per_file_indices)))
        total = sum(len(entries) for entries in per_file_indices)
        while len(self._indices) < total:
            l= rng.choice(labels)
            if cursor[l] < len(per_file_indices[l]):
                i = cursor[l]
                self._indices.append(per_file_indices[l][i])
                cursor[l] += 1

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

    def close(self) -> None:
        self._cache.clear()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary feature engineering
    # ------------------------------------------------------------------
    def _amp_summary(self, C_amp: list, S_amp: list, points: np.ndarray, index_map: Mapping[str, int]) -> np.ndarray:
        """Compute simple amplitude-based summary features for one event."""

        is_cherenkov = points[:, index_map[self.is_cherenkov_key]].astype(bool)
        if C_amp is not None:
            c_sum = C_amp
        else:
            c = points[is_cherenkov, index_map[self.amp_sum_key]]
            c_sum = float(np.sum(c))
        if S_amp is not None:
            s_sum = S_amp
        else:
            s = points[~is_cherenkov, index_map[self.amp_sum_key]]
            s_sum = float(np.sum(s))
        stats= [c_sum, s_sum, s_sum + c_sum]
        return np.asarray(stats, dtype=np.float32).transpose()
    
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


def collate_events(
    batch: Sequence[EventRecord],
    *,
    use_energy_adaptive_trim: Optional[bool] = None,
    trim_amp_feature_index: Optional[int] = None,
    trim_keep_frac_min: Optional[float] = None,
    trim_keep_frac_max: Optional[float] = None,
    trim_energy_min: Optional[float] = None,
    trim_energy_max: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Custom ``DataLoader`` collation for variable-length point clouds."""

    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch")

    if use_energy_adaptive_trim is None:
        use_energy_adaptive_trim = os.environ.get("PID_ENERGY_ADAPTIVE_TRIM", "1").lower() in {"1", "true", "yes", "on"}
    if trim_keep_frac_min is None:
        trim_keep_frac_min = float(os.environ.get("PID_TRIM_KEEP_FRAC_MIN", "0.70"))
    if trim_keep_frac_max is None:
        trim_keep_frac_max = float(os.environ.get("PID_TRIM_KEEP_FRAC_MAX", "1.00"))
    if trim_energy_min is None:
        trim_energy_min = float(os.environ.get("PID_TRIM_E_MIN", "1.0"))
    if trim_energy_max is None:
        trim_energy_max = float(os.environ.get("PID_TRIM_E_MAX", "100.0"))

    max_hits = max(record.points.shape[0] for record in batch)
    feature_dim = batch[0].points.shape[1]
    batch_size = len(batch)

    points = torch.zeros(batch_size, max_hits, feature_dim, dtype=batch[0].points.dtype)
    mask = torch.zeros(batch_size, max_hits, dtype=torch.bool)
    summary = torch.stack([record.summary for record in batch], dim=0)
    labels = torch.tensor([record.label for record in batch], dtype=torch.long)
    energy = torch.stack([record.energy for record in batch], dim=0).squeeze(-1)
    direction_template = next((record.direction for record in batch if record.direction is not None), None)
    direction = None
    if direction_template is not None:
        direction = torch.stack(
            [
                record.direction
                if record.direction is not None
                else torch.zeros_like(direction_template)
                for record in batch
            ],
            dim=0,
        )
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
        cid = None
        if record.cid is not None and num_hits > 0:
            cid = record.cid[:num_hits].reshape(-1)
            if cid.is_floating_point():
                valid = torch.isfinite(cid) & (cid != 0)
            else:
                valid = cid != 0
            valid_idx = torch.nonzero(valid, as_tuple=False)
            if valid_idx.numel() == 0:
                num_hits = 0
            else:
                num_hits = min(num_hits, int(valid_idx[-1].item()) + 1)
                cid = cid[:num_hits]

        if use_energy_adaptive_trim and num_hits > 0:
            energy_value = float(record.energy.reshape(-1)[0].item())
            keep_frac = _adaptive_keep_fraction(
                energy_value,
                min_frac=trim_keep_frac_min,
                max_frac=trim_keep_frac_max,
                e_min=trim_energy_min,
                e_max=trim_energy_max,
            )
            target_hits = num_hits
            if trim_amp_feature_index is not None and 0 <= trim_amp_feature_index < record.points.shape[1]:
                amp = record.points[:num_hits, trim_amp_feature_index]
                amp = torch.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0).abs()
                total_amp = float(torch.sum(amp).item())
                if total_amp > 0.0:
                    cumulative = torch.cumsum(amp, dim=0)
                    threshold = keep_frac * total_amp
                    reached = torch.nonzero(cumulative >= threshold, as_tuple=False)
                    if reached.numel() > 0:
                        target_hits = int(reached[0].item()) + 1
            target_hits = max(1, min(target_hits, num_hits))
            if cid is not None:
                target_hits = _trim_left_to_cid_boundary(cid, target_hits)
            num_hits = target_hits
            if cid is not None:
                cid = cid[:num_hits]

        points[i, :num_hits] = record.points[:num_hits]
        mask[i, :num_hits] = True
        if pos is not None and record.pos is not None:
            pos[i, :num_hits] = record.pos[:num_hits]

    collated = {
        "points": points,
        "mask": mask,
        "summary": summary,
        "labels": labels,
        "energy": energy,
        "event_id": event_id,
    }
    if direction is not None:
        collated["direction"] = direction
    if pos is not None:
        collated["pos"] = pos
    return collated


__all__ = ["DualReadoutEventDataset", "collate_events", "EventRecord"]
