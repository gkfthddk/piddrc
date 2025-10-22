"""Utility script to compute normalization statistics for calorimeter channels.

This script scans one or more HDF5 datasets and aggregates simple statistics for
all configured channels.  The resulting maxima or high quantiles can be fed
back into ``config.get_channel_config`` to populate ``CHANNELMAX`` with
realistic values.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import h5py
import numpy as np
import yaml

from config import get_channel_config


DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_PERCENTILES = (0.5, 0.9, 0.99)
DEFAULT_CHUNK_SIZE = 512


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan HDF5 datasets and compute summary statistics for configured "
            "detector channels."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory that contains <dataset>.h5py files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
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
        "--channels",
        nargs="*",
        default=None,
        help=(
            "Explicit dataset names to include in addition to the configured "
            "channel groups."
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


def _collect_channels(
    groups: Iterable[str] | None, extra_channels: Iterable[str] | None
) -> List[str]:
    channel_config, _, _ = get_channel_config()

    if groups is None:
        selected_groups = channel_config.keys()
    else:
        missing = [name for name in groups if name not in channel_config]
        if missing:
            raise ValueError(
                f"Unknown channel groups: {', '.join(missing)}. "
                "Check config.get_channel_config()."
            )
        selected_groups = groups

    channels = []
    for group in selected_groups:
        channels.extend(channel_config[group]["CHANNEL"])

    if extra_channels:
        channels.extend(extra_channels)

    # Preserve order but drop duplicates.
    seen = set()
    unique_channels = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)
    return unique_channels


def _iter_batches(dataset: h5py.Dataset, chunk_size: int) -> Iterable[np.ndarray]:
    length = dataset.shape[0]
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        yield dataset[start:end]


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

    for name in dataset_names:
        path = os.path.join(data_dir, f"{name}.h5py")
        if not os.path.exists(path):
            warnings.warn(f"File not found: {path}. Skipping.")
            continue

        with h5py.File(path, "r") as handle:
            for channel in channels:
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
        yaml.safe_dump(
            results,
            stream=sys.stdout,
            sort_keys=False,
            default_flow_style=False,
        )
        return

    os.makedirs(os.path.dirname(output), exist_ok=True) if os.path.dirname(output) else None

    ext = os.path.splitext(output)[1].lower()
    if ext == ".json":
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    elif ext in {".yml", ".yaml"}:
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

    channels = _collect_channels(args.channel_groups, args.channels)
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
