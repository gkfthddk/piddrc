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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

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


DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_PERCENTILES = (0.5, 0.9, 0.99)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_DATA_DIR = "h5s"
DEFAULT_DATASETS = (
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

DEFAULT_CHANNEL_CONFIG: Mapping[str, Mapping[str, object]] = {
    "summary": {
        "DESCRIPTION": "Core per-event features written by toh5.py",
        "CHANNEL": tuple(_BASE_CHANNELS),
    }
}
for _pool in (4, 8, 14, 28, 56):
    DEFAULT_CHANNEL_CONFIG[f"pool_{_pool}"] = {
        "DESCRIPTION": f"Pooled reconstruction features with pool size {_pool}",
        "CHANNEL": tuple(template.format(pool=_pool) for template in _POOL_CHANNEL_TEMPLATES),
    }
del _pool

_default_channel_sequence: List[str] = []
for _config in DEFAULT_CHANNEL_CONFIG.values():
    _default_channel_sequence.extend(_config["CHANNEL"])
DEFAULT_CHANNELS = tuple(dict.fromkeys(_default_channel_sequence))
del _default_channel_sequence, _config


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan HDF5 datasets and compute summary statistics for configured "
            "detector channels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory that contains <dataset>.h5py files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=(
            "Dataset names (without .h5py extension) to include in the scan. "
            "Each name will be resolved against --data-dir."
        ),
    )
    parser.add_argument(
        "--channel-groups",
        nargs="*",
        default=None,
        help=(
            "Optional subset of channel groups defined in get_channel_config(). "
            "If omitted, all configured groups are scanned."
        ),
    )
    parser.add_argument(
        "--config-module",
        default="config",
        help=(
            "Python module that exposes get_channel_config(). Set to 'none' to "
            "skip loading channel groups from a module. If the module cannot be "
            "imported the script will fall back to discovery."
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=DEFAULT_CHANNELS,
        help=(
            "Explicit dataset names to include in addition to the configured "
            "channel groups.  When omitted and no configuration module is "
            "provided, the script will attempt to discover channels by "
            "inspecting the input files."
        ),
    )
    parser.add_argument(
        "--default-config",
        action="store_true",
        help=(
            "Use the built-in channel configuration derived from toh5.py. "
            "This is automatically enabled when no configuration module is "
            "provided or cannot be imported."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of events to read per batch from each dataset.",
    )
    parser.add_argument(
        "--sample-size",
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
        default=DEFAULT_PERCENTILES,
        help="Percentiles to report (values between 0 and 1).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional path where the statistics will be written.  The format is "
            "chosen by file extension (.json or .yml/.yaml).  If omitted, the "
            "results are printed to stdout."
        ),
    )
    return parser.parse_args(argv)


@dataclass
class StreamingStats:
    """Keep track of aggregate statistics for a channel."""

    sample_size: int
    percentiles: Sequence[float]
    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min_val: float = math.inf
    max_val: float = -math.inf
    _sample: List[float] = field(default_factory=list)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return

        # Promote to float64 for numerical stability.
        flat = values.astype(np.float64, copy=False).ravel()
        start_count = self.count
        self.count += flat.size
        self.sum += float(np.sum(flat))
        self.sumsq += float(np.sum(flat * flat))
        self.min_val = min(self.min_val, float(np.min(flat)))
        self.max_val = max(self.max_val, float(np.max(flat)))

        if self.sample_size <= 0:
            return

        # Fill the reservoir until it reaches the desired size.
        if len(self._sample) < self.sample_size:
            needed = self.sample_size - len(self._sample)
            take = min(needed, flat.size)
            self._sample.extend(map(float, flat[:take]))
            flat = flat[take:]
            total_seen = start_count + take
        else:
            total_seen = start_count

        if flat.size == 0:
            return

        # Reservoir sampling for the remaining values.
        for value in flat:
            total_seen += 1
            j = np.random.randint(0, total_seen)
            if j < self.sample_size:
                self._sample[j] = float(value)

    def finalize(self) -> Mapping[str, float]:
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

        percentile_values: Dict[float, float]
        if self.sample_size > 0 and self._sample:
            sample_array = np.asarray(self._sample, dtype=np.float64)
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


def _load_channel_config(
    module_name: Optional[str],
    use_default: bool,
) -> Optional[Mapping[str, Mapping[str, object]]]:
    if module_name is None:
        return DEFAULT_CHANNEL_CONFIG if use_default else None

    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        warnings.warn(
            "Channel configuration module '%s' could not be imported; "
            "falling back to the built-in defaults." % module_name
        )
        return DEFAULT_CHANNEL_CONFIG

    if not hasattr(module, "get_channel_config"):
        raise AttributeError(
            f"Module '{module_name}' does not define get_channel_config()."
        )

    channel_config, _, _ = module.get_channel_config()
    if use_default:
        merged: Dict[str, Mapping[str, object]] = dict(DEFAULT_CHANNEL_CONFIG)
        merged.update(channel_config)
        return merged
    return channel_config


def _discover_channels(
    data_dir: str, dataset_names: Sequence[str]
) -> List[str]:
    """Infer channel names by inspecting the provided datasets."""

    discovered: List[str] = []
    seen = set()

    for name in dataset_names:
        path = os.path.join(data_dir, f"{name}.h5py")
        if not os.path.exists(path):
            continue

        try:
            with h5py.File(path, "r") as handle:
                for key in handle.keys():
                    if key not in seen and isinstance(handle[key], h5py.Dataset):
                        seen.add(key)
                        discovered.append(key)
        except OSError as exc:
            warnings.warn(f"Failed to inspect {path}: {exc}")

    return discovered


def _collect_channels(
    channel_config: Optional[Mapping[str, Mapping[str, object]]],
    groups: Iterable[str] | None,
    extra_channels: Iterable[str] | None,
    data_dir: str,
    dataset_names: Sequence[str],
) -> List[str]:
    channels: List[str] = []

    if channel_config is not None:
        if groups is None:
            selected_groups = channel_config.keys()
        else:
            missing = [name for name in groups if name not in channel_config]
            if missing:
                raise ValueError(
                    f"Unknown channel groups: {', '.join(missing)}. "
                    "Check get_channel_config() in the provided module."
                )
            selected_groups = groups

        for group in selected_groups:
            channels.extend(channel_config[group]["CHANNEL"])
    elif groups:
        raise ValueError(
            "Channel groups were requested but no configuration module was loaded."
        )

    if extra_channels:
        channels.extend(extra_channels)

    # Preserve order but drop duplicates.
    seen = set()
    unique_channels = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)

    if not unique_channels:
        discovered = _discover_channels(data_dir, dataset_names)
        if discovered:
            warnings.warn(
                "No channels were provided explicitly; discovered channels from "
                f"datasets: {', '.join(discovered)}"
            )
        unique_channels = discovered

    return unique_channels


def _iter_batches(dataset: h5py.Dataset, chunk_size: int) -> Iterable[np.ndarray]:
    length = dataset.shape[0]
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        yield dataset[start:end]


def _progress(iterable: Iterable, **kwargs):
    if tqdm is None:
        for item in iterable:
            yield item
        return

    progress_bar = tqdm(iterable, **kwargs)
    try:
        for item in progress_bar:
            yield item
    finally:
        progress_bar.close()


def scan_datasets(
    data_dir: str,
    dataset_names: Sequence[str],
    channels: Sequence[str],
    chunk_size: int,
    sample_size: int,
    percentiles: Sequence[float],
) -> MutableMapping[str, StreamingStats]:
    stats = {
        channel: StreamingStats(sample_size=sample_size, percentiles=percentiles)
        for channel in channels
    }

    dataset_iter = _progress(dataset_names, desc="Datasets", unit="file")
    for name in dataset_iter:
        path = os.path.join(data_dir, f"{name}.h5py")
        if not os.path.exists(path):
            warnings.warn(f"File not found: {path}. Skipping.")
            continue

        with h5py.File(path, "r") as handle:
            channel_iter = _progress(
                channels,
                desc=f"{name} channels",
                unit="channel",
                leave=False,
            )
            for channel in channel_iter:
                if channel not in handle:
                    warnings.warn(
                        f"Dataset '{channel}' not found in file {path}. Skipping this channel."
                    )
                    continue
                dataset = handle[channel]
                for batch in _iter_batches(dataset, chunk_size):
                    stats[channel].update(np.asarray(batch))
    return stats


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

    module_name: Optional[str]
    if args.config_module and args.config_module.lower() != "none":
        module_name = args.config_module
    else:
        module_name = None

    use_default_config = args.default_config or module_name is None
    channel_config = _load_channel_config(module_name, use_default=use_default_config)

    channels = _collect_channels(
        channel_config,
        args.channel_groups,
        args.channels,
        data_dir=args.data_dir,
        dataset_names=args.datasets,
    )
    if not channels:
        raise ValueError("No channels selected for scanning.")

    stats = scan_datasets(
        data_dir=args.data_dir,
        dataset_names=args.datasets,
        channels=channels,
        chunk_size=args.chunk_size,
        sample_size=args.sample_size,
        percentiles=args.percentiles,
    )

    results = {channel: stat.finalize() for channel, stat in stats.items()}
    _dump_results(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
