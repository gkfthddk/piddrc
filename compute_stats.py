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
    "e-_1-100GeV",
    "gamma_1-100GeV",
    "pi0_1-100GeV",
    "pi+_1-100GeV",
)

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

def _build_write_keys() -> List[str]:
    """Generates the list of all dataset keys to be written."""
    keys = list(_BASE_CHANNELS)  # Start with a copy of base channels
    for pool in [4, 7, 8, 14, 28, 56]:
        for template in _POOL_CHANNEL_TEMPLATES:
            keys.append(template.format(pool=pool))
    return keys

DEFAULT_DATASETS = _build_write_keys()


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
        "--percentiles",
        type=float,
        nargs="*",
        default=list(DEFAULT_PERCENTILES),
        help="Percentiles to report (values between 0 and 1).",
    )
    parser.add_argument(
        "--output",
        default="h5s/stats_1-100GeV.yaml",
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

def scan_files(
    data_dir: str,
    file_names: Sequence[str],
    key: str,
    sample_size: int,
    percentiles: Sequence[float],
) -> Dict[str, Any]:
    dataset=[]
    for name in file_names:
        path = os.path.join(data_dir, f"{name}.h5py")
        if not os.path.exists(path):
            warnings.warn(f"File not found: {path}. Skipping.")
            continue
        
        with h5py.File(path, "r") as handle:
            if key in handle:
                dataset.extend(handle[key][:sample_size])
    return compute_stats(np.array(dataset), percentiles)

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
    results={}
    iterator = tqdm(args.dataset_names) if tqdm is not None else args.dataset_names
    for key in iterator:
        results[key] = scan_files(
            data_dir=args.data_dir,
            file_names=args.files,
            key=key,
            sample_size=args.sample_size,
            percentiles=args.percentiles,
        )

    _dump_results(results, args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
