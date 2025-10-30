"""Unit tests for the channel statistics helper script."""

from __future__ import annotations

import math
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from compute_stats import (
    DEFAULT_DATASETS,
    DEFAULT_DIR,
    DEFAULT_FILES,
    DEFAULT_PERCENTILES,
    DEFAULT_SAMPLE_SIZE,
    _parse_args,
    compute_stats,
    scan_files,
)


def test_parse_args_populates_defaults():
    args = _parse_args([])
    assert args.data_dir == DEFAULT_DATA_DIR
    assert args.files == list(DEFAULT_FILES)
    assert args.dataset_names == DEFAULT_DATASETS
    assert args.sample_size == DEFAULT_SAMPLE_SIZE
    assert args.percentiles == list(DEFAULT_PERCENTILES)


def test_compute_stats_with_data():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    percentiles = [0.25, 0.5, 0.75]
    result = compute_stats(values, percentiles)
    assert result["count"] == 4
    assert math.isclose(result["mean"], 2.5)
    assert math.isclose(result["std"], np.std(values))
    assert math.isclose(result["min"], 1.0)
    assert math.isclose(result["max"], 4.0)
    assert math.isclose(result["percentiles"][0.5], 2.5)


def test_compute_stats_with_empty_data():
    values = np.array([])
    percentiles = [0.5]
    result = compute_stats(values, percentiles)
    assert result["count"] == 0
    assert math.isnan(result["mean"])
    assert math.isnan(result["percentiles"][0.5])


def test_scan_files_computes_statistics(tmp_path):
    data_dir = tmp_path
    file_name = "sample"
    values = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    flat = values.ravel()

    with h5py.File(data_dir / f"{file_name}.h5py", "w") as handle:
        handle.create_dataset("C_amp", data=values)

    result = scan_files(
        data_dir=str(data_dir),
        files_names=[file_name],
        key="C_amp",
        sample_size=32,
        percentiles=(0.25, 0.5, 0.75),
    )

    assert result["count"] == flat.size
    assert math.isclose(result["mean"], float(np.mean(flat)), rel_tol=1e-6)
    assert math.isclose(result["std"], float(np.std(flat)), rel_tol=1e-6)
    quantiles = (0.25, 0.5, 0.75)
    expected_quantiles = np.quantile(flat, quantiles)
    for quantile, expected in zip(quantiles, expected_quantiles):
        assert math.isclose(result["percentiles"][quantile], float(expected), rel_tol=1e-6)
