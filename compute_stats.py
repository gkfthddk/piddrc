"""Utility script to compute normalization statistics for calorimeter channels.

This script scans one or more HDF5 datasets and aggregates simple statistics for
all configured channels.  The resulting maxima or high quantiles can be fed
back into ``get_channel_config`` (loaded from a configurable module) to populate
``CHANNELMAX`` with realistic values.  When a configuration module is not
available, you can provide channel names explicitly or allow the script to
discover them from the input files.
"""
from __future__ import annotations

import argparse
import random
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import h5py
import numpy as np
from typing import List as TypingList

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None


DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_PERCENTILES = (0.5, 0.9, 0.99)
DEFAULT_DIR = "h5s"
DEFAULT_FILES = (
    "e-_1-100GeV_1",
    "gamma_1-100GeV_1",
    "pi0_1-100GeV_1",
    "pi+_1-100GeV_1",
)
CHUNK_ROWS = 2048

_BASE_CHANNELS = [
    "C_amp",
    "C_raw",
    "DRcalo3dHits.amplitude",
    "DRcalo3dHits.amplitude_sum",
    "DRcalo3dHits.cellID",
    "DRcalo3dHits.position.x",
    "DRcalo3dHits.position.y",
    "DRcalo3dHits.position.z",
    "DRcalo3dHits.time",
    "DRcalo3dHits.time_end",
    "DRcalo3dHits.type",
    "DRcalo2dHits.amplitude",
    "DRcalo2dHits.cellID",
    "DRcalo2dHits.position.x",
    "DRcalo2dHits.position.y",
    "DRcalo2dHits.position.z",
    "DRcalo2dHits.type",
    "Reco3dHits_C.amplitude",
    "Reco3dHits_C.position.x",
    "Reco3dHits_C.position.y",
    "Reco3dHits_C.position.z",
    "Reco3dHits_S.amplitude",
    "Reco3dHits_S.position.x",
    "Reco3dHits_S.position.y",
    "Reco3dHits_S.position.z",
    "E_dep",
    "E_gen",
    "E_leak",
    "GenParticles.momentum.phi",
    "GenParticles.momentum.theta",
    "seed",
    "S_amp",
    "S_raw",
    "angle2",
]

_POOL_CHANNEL_TEMPLATES = (
    "Reco3dHits{pool}_C.amplitude",
    "Reco3dHits{pool}_C.position.x",
    "Reco3dHits{pool}_C.position.y",
    "Reco3dHits{pool}_C.position.z",
    "Reco3dHits{pool}_S.amplitude",
    "Reco3dHits{pool}_S.position.x",
    "Reco3dHits{pool}_S.position.y",
    "Reco3dHits{pool}_S.position.z",
    "DRcalo3dHits{pool}.amplitude",
    "DRcalo3dHits{pool}.amplitude_sum",
    "DRcalo3dHits{pool}.cellID",
    "DRcalo3dHits{pool}.position.x",
    "DRcalo3dHits{pool}.position.y",
    "DRcalo3dHits{pool}.position.z",
    "DRcalo3dHits{pool}.time",
    "DRcalo3dHits{pool}.time_end",
    "DRcalo3dHits{pool}.type",
    "DRcalo2dHits{pool}.amplitude",
    "DRcalo2dHits{pool}.cellID",
    "DRcalo2dHits{pool}.position.x",
    "DRcalo2dHits{pool}.position.y",
    "DRcalo2dHits{pool}.position.z",
    "DRcalo2dHits{pool}.type",
)
POOLSET=[4,5,6,7,8,9,10,11,12,13,14,16,28,56]
def _build_write_keys() -> List[str]:
    """Generates the list of all dataset keys to be written."""
    keys = list(_BASE_CHANNELS)  # Start with a copy of base channels
    for pool in POOLSET:
        for template in _POOL_CHANNEL_TEMPLATES:
            keys.append(template.format(pool=pool))
    return keys

DEFAULT_DATASETS = _build_write_keys()


class StreamingStats:
    """Keeps running aggregates and a bounded reservoir sample for percentiles."""

    def __init__(self, sample_size: int, percentiles: Sequence[float]):
        self.sample_size = max(sample_size, 0)
        self.percentiles = percentiles
        self.count: int = 0
        self.sum: float = 0.0
        self.sumsq: float = 0.0
        self.min_val: float = math.inf
        self.max_val: float = -math.inf
        self._sample: TypingList[float] = []

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return

        flat = np.asarray(values, dtype=np.float64).ravel()
        start_count = self.count
        self.count += flat.size
        self.sum += float(np.sum(flat))
        self.sumsq += float(np.sum(flat * flat))
        self.min_val = min(self.min_val, float(np.min(flat)))
        self.max_val = max(self.max_val, float(np.max(flat)))

        if self.sample_size <= 0:
            return

        # Reservoir sampling to keep an unbiased subset for percentile estimation.
        for idx, value in enumerate(flat):
            global_index = start_count + idx
            if len(self._sample) < self.sample_size:
                self._sample.append(float(value))
            else:
                replace_position = random.randint(0, global_index)
                if replace_position < self.sample_size:
                    self._sample[replace_position] = float(value)

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "min": math.nan,
                "max": math.nan,
                "mean": math.nan,
                "std": math.nan,
                "percentiles": {p: math.nan for p in self.percentiles},
            }

        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(variance)

        if self.sample_size > 0 and self._sample:
            sample_array = np.array(self._sample, dtype=np.float64)
            percentile_values = {
                p: float(np.quantile(sample_array, p, method="linear"))
                for p in self.percentiles
            }
        else:
            percentile_values = {p: math.nan for p in self.percentiles}

        return {
            "count": int(self.count),
            "min": float(self.min_val),
            "max": float(self.max_val),
            "mean": float(mean),
            "std": float(std),
            "percentiles": percentile_values,
        }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan HDF5 datasets and compute summary statistics for configured "
            "detector channels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DIR,
        help="Directory that contains <dataset>.h5py files",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_FILES),
        help=(
            "Dataset names (without .h5py extension) to include in the scan. "
            "Each name will be resolved against --data-dir."
        ),
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=DEFAULT_DATASETS,
        help=(
            "Explicit dataset names to include in addition to the configured "
            "channel groups.  When omitted and no configuration module is "
            "provided, the script will attempt to discover channels by "
            "inspecting the input files."
        ),
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Maximum number of samples retained for percentile estimation. "
            "Reservoir sampling is used once the cap is exceeded."
        ),
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=500,
        help=(
            "Maximum number of points to sample from each event."
        ),
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="*",
        default=list(DEFAULT_PERCENTILES),
        help="Percentiles to report (values between 0 and 1).",
    )
    parser.add_argument(
        "--output",
        default="h5s/stats_1-100GeV_1.yaml",
        help=(
            "Optional path where the statistics will be written.  The format is "
            "chosen by file extension (.json or .yml/.yaml).  If omitted, the "
            "results are printed to stdout."
        ),
    )
    return parser.parse_args(argv)

def compute_stats(values: np.ndarray, percentiles: Sequence[float]) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0, "min": math.nan, "max": math.nan,
            "mean": math.nan, "std": math.nan,
            "percentiles": {p: math.nan for p in percentiles},
        }

    # Promote to float64 for numerical stability.
    flat = values.astype(np.float64, copy=False).ravel()
    size = flat.size
    summed = float(np.sum(flat))
    sumsq = float(np.sum(flat * flat))
    min_val = min(math.inf, float(np.min(flat)))
    max_val = max(-math.inf, float(np.max(flat)))

    mean = summed / size
    variance = max(sumsq / size - mean * mean, 0.0)
    std = math.sqrt(variance)

    percentile_values: Dict[float, float]
    percentile_values = {p: float(np.quantile(flat, p, method="linear")) for p in percentiles}

    return {
        "count": int(size),
        "min": float(min_val),
        "max": float(max_val),
        "mean": float(mean),
        "std": float(std),
        "percentiles": percentile_values
    }

def _iter_dataset(
    ds: h5py.Dataset,
    sample_size: int,
    max_points: Optional[int],
    chunk_rows: int = CHUNK_ROWS,
):
    """Yield chunks from the first `sample_size` rows (and optional columns) of a dataset."""
    if sample_size <= 0:
        return

    total_rows = min(sample_size, ds.shape[0])
    idx = 0
    while idx < total_rows:
        end = min(idx + chunk_rows, total_rows)
        if max_points is not None and ds.ndim > 1:
            yield ds[idx:end, :max_points]
        else:
            yield ds[idx:end]
        idx = end


def scan_files(
    data_dir: str,
    file_names: Sequence[str],
    dataset_names: Optional[Sequence[str]] = None,
    *,
    key: Optional[str] = None,
    sample_size: int,
    percentiles: Sequence[float],
    max_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute stats one dataset at a time to keep memory bounded.

    Trade-off: files are reopened per dataset, but resident memory is limited to
    one reservoir sample plus a small chunk buffer.
    """
    if key is not None:
        if dataset_names is not None:
            raise ValueError("Provide either 'key' or 'dataset_names', not both.")
        dataset_names = [key]
        single_key_mode = True
    else:
        if dataset_names is None:
            raise ValueError("Either 'dataset_names' or 'key' must be provided.")
        single_key_mode = False

    results: Dict[str, Any] = {}
    iterator = (
        tqdm(dataset_names, desc="Datasets", leave=False)
        if tqdm is not None
        else dataset_names
    )

    for key in iterator:
        if tqdm is not None:
            iterator.set_postfix_str(key)
        stats = StreamingStats(sample_size=sample_size, percentiles=percentiles)

        file_iter = (
            tqdm(file_names, desc=f"{key}", leave=False)
            if tqdm is not None
            else file_names
        )

        for name in file_iter:
            path = os.path.join(data_dir, f"{name}.h5py")
            if not os.path.exists(path):
                warnings.warn(f"File not found: {path}. Skipping.")
                continue

            with h5py.File(path, "r") as handle:
                if key not in handle:
                    continue
                ds = handle[key]
                chunk_iter = _iter_dataset(ds, sample_size=sample_size, max_points=max_points)
                if tqdm is not None:
                    # Total is approximate: rows/chunk_rows up to sample_size.
                    total_chunks = max(1, min(sample_size, ds.shape[0]) // CHUNK_ROWS + 1)
                    chunk_iter = tqdm(chunk_iter, total=total_chunks, desc="chunks", leave=False)
                for chunk in chunk_iter:
                    stats.update(chunk)

        results[key] = stats.finalize()

    if single_key_mode:
        return results[key]  # type: ignore[index]
    return results

def _dump_results(results: Mapping[str, Mapping[str, float]], output: str | None) -> None:
    if output is None:
        if yaml is not None:
            yaml.safe_dump(
                results,
                stream=sys.stdout,
                sort_keys=False,
                default_flow_style=False,
            )
        else:
            json.dump(results, sys.stdout, indent=2)
            sys.stdout.write("\n")
        return

    os.makedirs(os.path.dirname(output), exist_ok=True) if os.path.dirname(output) else None

    ext = os.path.splitext(output)[1].lower()
    if ext == ".json":
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    elif ext in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to write YAML output. Install it or choose a .json file."
            )
        with open(output, "w", encoding="utf-8") as fh:
            yaml.safe_dump(results, fh, sort_keys=False, default_flow_style=False)
    else:
        raise ValueError(
            "Unsupported output format. Use .json, .yml, or .yaml extensions."
        )

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    for value in args.percentiles:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Percentile values must lie in [0, 1]. Received {value}."
            )
    if not args.dataset_names:
        raise ValueError("No datasets selected for scanning.")
    if not args.files:
        raise ValueError("No input files specified for scanning.")
    results = scan_files(
        data_dir=args.data_dir,
        file_names=args.files,
        dataset_names=args.dataset_names,
        sample_size=args.sample_size,
        max_points=args.max_points,
        percentiles=args.percentiles,
    )

    _dump_results(results, args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
