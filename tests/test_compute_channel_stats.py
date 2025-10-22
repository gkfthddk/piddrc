"""Unit tests for the channel statistics helper script."""

from __future__ import annotations

import math

import h5py
import numpy as np
from compute_channel_stats import (
    DEFAULT_CHANNELS,
    DEFAULT_CHANNEL_CONFIG,
    DEFAULT_DATASETS,
    DEFAULT_DATA_DIR,
    _parse_args,
    scan_datasets,
)


def test_parse_args_populates_defaults():
    args = _parse_args([])
    assert args.data_dir == DEFAULT_DATA_DIR
    assert tuple(args.datasets) == DEFAULT_DATASETS
    assert tuple(args.channels) == DEFAULT_CHANNELS
    assert args.default_config is False


def test_default_channel_config_matches_expectations():
    assert "summary" in DEFAULT_CHANNEL_CONFIG
    summary_channels = DEFAULT_CHANNEL_CONFIG["summary"]["CHANNEL"]
    assert "C_amp" in summary_channels
    assert "S_raw" in summary_channels

    for pool in (4, 8, 14, 28, 56):
        group_key = f"pool_{pool}"
        assert group_key in DEFAULT_CHANNEL_CONFIG
        group_channels = DEFAULT_CHANNEL_CONFIG[group_key]["CHANNEL"]
        assert any(name.startswith(f"Reco3dHits{pool}_C") for name in group_channels)
        assert any(name.startswith(f"DRcalo3dHits{pool}") for name in group_channels)

    # DEFAULT_CHANNELS should be deduplicated and contain the union.
    assert len(DEFAULT_CHANNELS) == len(set(DEFAULT_CHANNELS))
    assert "C_amp" in DEFAULT_CHANNELS


def test_scan_datasets_computes_statistics(tmp_path):
    data_dir = tmp_path
    dataset_name = "sample"
    values = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    flat = values.ravel()

    with h5py.File(data_dir / f"{dataset_name}.h5py", "w") as handle:
        handle.create_dataset("C_amp", data=values)

    stats = scan_datasets(
        data_dir=str(data_dir),
        dataset_names=[dataset_name],
        channels=["C_amp"],
        chunk_size=1,
        sample_size=32,
        percentiles=(0.25, 0.5, 0.75),
    )

    result = stats["C_amp"].finalize()
    assert result["count"] == flat.size
    assert math.isclose(result["mean"], float(np.mean(flat)), rel_tol=1e-6)
    assert math.isclose(result["std"], float(np.std(flat)), rel_tol=1e-6)
    for quantile, expected in zip((0.25, 0.5, 0.75), np.quantile(flat, (0.25, 0.5, 0.75))):
        assert math.isclose(result["percentiles"][quantile], float(expected), rel_tol=1e-6)
