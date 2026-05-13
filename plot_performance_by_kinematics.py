"""Plot run performance as a function of E_gen, theta, and phi.

This script compares multiple run directories (e.g. test2/test3/test4/test5)
using:
- pairwise ROC-AUC in bins of E_gen/theta/phi
- overall accuracy in bins of E_gen/theta/phi

Legends include differing config fields so runs with different settings are
visually identified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from utils.plot_helpers import (
    PAIR_LABELS,
    PAIR_SPECS,
    config_label,
    differing_config_keys,
    infer_class_to_label,
    pair_key,
    resolve_run_dir,
    safe_name,
    set_publication_style,
)

set_publication_style()


KIN_VARS = {
    "E_gen": "E_gen",
    "theta": "GenParticles.momentum.theta",
    "phi": "GenParticles.momentum.phi",
}
CONFIG_DIFF_EXCLUDE = {
    "cache_size",
    "max_cache_chunks",
    "eval_batch_size",
    "num_workers",
    "log_batch_lengths",
    "log_cuda_memory",
}
FIXED_ENERGY_EDGES = np.asarray([1.0, 10.0, 30.0, 70.0, 120.0], dtype=np.float64)
MIN_CONDITIONAL_EVENTS = 50
DEFAULT_BOOTSTRAP_SAMPLES = 32
DEFAULT_BOOTSTRAP_SEED = 12345
DEFAULT_CONDITIONAL_ENERGY_CURVE_BINS = 16
DEFAULT_CONDITIONAL_THETA_CURVE_BINS = 16
EFFICIENCY_REFERENCE_LINES = {
    "pi+ vs e-": [
        (0.99, "tab:green", "99% eff"),
        (0.90, "tab:purple", "90% eff"),
    ],
    "pi0 vs gamma": [
        (0.90, "tab:green", "90% eff"),
        (0.80, "tab:purple", "80% eff"),
    ],
}
FIXED_TPR_TARGETS = {
    "pi+ vs e-": [0.99, 0.90],
    "pi0 vs gamma": [0.90, 0.80],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot performance vs E_gen/theta/phi across runs.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run names or directories.")
    parser.add_argument(
        "--legend-labels",
        nargs="*",
        default=None,
        help=(
            "Optional legend labels, in the same order as --run-dirs. "
            "If provided, these override the auto-generated config labels."
        ),
    )
    parser.add_argument("--base-dir", default="save", help="Base dir when run names are provided.")
    parser.add_argument("--out-dir", default="plots", help="Output directory.")
    parser.add_argument(
        "--out-subdir",
        default=None,
        help="Optional subdirectory under --out-dir where plots will be written.",
    )
    parser.add_argument("--bins", type=int, default=20, help="Number of bins.")
    parser.add_argument(
        "--binning",
        choices=["quantile", "uniform"],
        default="quantile",
        help="Binning strategy for x variables.",
    )
    parser.add_argument(
        "--barrel-theta-min",
        type=float,
        default=0.61,
        help="Barrel lower theta bound in radians.",
    )
    parser.add_argument(
        "--barrel-theta-max",
        type=float,
        default=np.pi-0.61,
        help="Barrel upper theta bound in radians.",
    )
    parser.add_argument(
        "--theta-binning",
        choices=["data", "tower"],
        default="data",
        help="Use standard data-driven bins or tower-geometry theta bins.",
    )
    parser.add_argument(
        "--tower-geometry-xml",
        default="/users/yulee/dream/tools/dr_modules/DRcalo.xml",
        help="Geometry XML used to derive theta tower bin edges when --theta-binning tower is selected.",
    )
    parser.add_argument(
        "--theta-tower-alignment",
        choices=["edge", "shifted", "center"],
        default="edge",
        help="Align theta bins to tower edges or use a shifted variant where tower boundaries fall near bin centers.",
    )
    parser.add_argument(
        "--cache-dir",
        default="plots/.perf_cache",
        help="Directory used to cache extracted kinematics arrays across runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk caching of extracted kinematics arrays.",
    )
    parser.add_argument(
        "--show-hdf5-progress",
        action="store_true",
        help="Show detailed HDF5 file/chunk progress bars while loading kinematics.",
    )
    parser.add_argument(
        "--theta-plot-max",
        type=float,
        default=None,
        help="Optional upper x-limit for theta plots. Defaults to the observed data range.",
    )
    parser.add_argument(
        "--conditional-min-events",
        type=int,
        default=MIN_CONDITIONAL_EVENTS,
        help="Minimum events required in a conditional bin before the metric is computed.",
    )
    parser.add_argument(
        "--conditional-bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples used to estimate conditional AUC error bars.",
    )
    parser.add_argument(
        "--conditional-bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for conditional metric bootstrap resampling.",
    )
    parser.add_argument(
        "--conditional-energy-curve-bins",
        type=int,
        default=DEFAULT_CONDITIONAL_ENERGY_CURVE_BINS,
        help=(
            "Number of finer energy bins used for the AUC-vs-energy curves inside each theta slice. "
            "This is independent of the coarse 4-bin heatmap."
        ),
    )
    parser.add_argument(
        "--conditional-theta-curve-bins",
        type=int,
        default=DEFAULT_CONDITIONAL_THETA_CURVE_BINS,
        help=(
            "Number of finer theta bins used for the AUC-vs-theta curves inside each energy slice. "
            "This is independent of the coarse 4-bin heatmap."
        ),
    )
    return parser.parse_args()


def _resolve_h5_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        Path(".") / raw,
        Path("h5s") / raw,
        Path("/store/ml/dual-readout/h5s") / raw,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


def _extract_event_ids(records: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    file_ids = np.full(len(records), -1, dtype=np.int64)
    event_ids = np.full(len(records), -1, dtype=np.int64)
    for i, rec in enumerate(records):
        eid = rec.get("event_id", [])
        if isinstance(eid, list) and len(eid) >= 2:
            file_ids[i] = int(eid[0])
            event_ids[i] = int(eid[1])
    return file_ids, event_ids


def _load_vars_from_h5(
    records: Sequence[Mapping[str, object]],
    files: Sequence[str],
    dataset_keys: Sequence[str],
    cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], bytes, bytes], Dict[str, np.ndarray]] | None = None,
    *,
    desc: str | None = None,
    chunk_size: int = 8192,
) -> Dict[str, np.ndarray]:
    file_ids, event_ids = _extract_event_ids(records)
    cache_key = (
        tuple(dataset_keys),
        tuple(str(f) for f in files),
        file_ids.tobytes(),
        event_ids.tobytes(),
    )
    if cache is not None and cache_key in cache:
        return {key: value.copy() for key, value in cache[cache_key].items()}

    n = len(records)
    out = {key: np.full(n, np.nan, dtype=np.float64) for key in dataset_keys}

    idx_by_file: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        if file_ids[i] >= 0 and event_ids[i] >= 0:
            idx_by_file[int(file_ids[i])].append(i)

    file_items = list(idx_by_file.items())
    if desc is not None:
        file_items = tqdm(file_items, desc=desc, unit="file", leave=False)

    for fidx, rec_indices in file_items:
        if not (0 <= fidx < len(files)):
            continue
        h5_path = _resolve_h5_path(str(files[fidx]))
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h5:
            ev = np.asarray([event_ids[i] for i in rec_indices], dtype=np.int64)
            order = np.argsort(ev, kind="stable")
            ev_sorted = ev[order]
            rec_idx_sorted = np.asarray(rec_indices, dtype=np.int64)[order]
            chunk_count = (ev_sorted.size + chunk_size - 1) // chunk_size
            for dataset_key in dataset_keys:
                if dataset_key not in h5:
                    continue
                ds = h5[dataset_key]
                chunk_iter = range(0, ev_sorted.size, chunk_size)
                if desc is not None and chunk_count > 1:
                    chunk_iter = tqdm(
                        chunk_iter,
                        total=chunk_count,
                        desc=f"{Path(str(files[fidx])).name}:{dataset_key}",
                        unit="chunk",
                        leave=False,
                    )
                for start in chunk_iter:
                    stop = min(start + chunk_size, ev_sorted.size)
                    out[dataset_key][rec_idx_sorted[start:stop]] = np.asarray(
                        ds[ev_sorted[start:stop]],
                        dtype=np.float64,
                    ).reshape(-1)
    if cache is not None:
        cache[cache_key] = {key: value.copy() for key, value in out.items()}
    return out


def make_bins(values: np.ndarray, bins: int, mode: str) -> np.ndarray:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([], dtype=np.float64)
    if mode == "uniform":
        lo, hi = float(np.min(v)), float(np.max(v))
        if lo == hi:
            return np.array([lo, hi + 1e-12], dtype=np.float64)
        return np.linspace(lo, hi, bins + 1)
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(v, q)
    edges = np.unique(edges)
    if edges.size < 2:
        x = float(v[0])
        return np.array([x, x + 1e-12], dtype=np.float64)
    return edges


def _kin_cache_path(
    run_dir: Path,
    config_path: Path,
    output_path: Path,
    files: Sequence[str],
    dataset_keys: Sequence[str],
    cache_dir: Path,
) -> Path:
    payload = {
        "run_dir": str(run_dir.resolve()),
        "config_mtime_ns": config_path.stat().st_mtime_ns,
        "output_mtime_ns": output_path.stat().st_mtime_ns,
        "files": [str(f) for f in files],
        "dataset_keys": [str(k) for k in dataset_keys],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{safe_name(run_dir.name)}_{digest}.npz"


def load_run_kinematics(
    run_dir: Path,
    config_path: Path,
    output_path: Path,
    records: Sequence[Mapping[str, object]],
    files: Sequence[str],
    dataset_keys: Sequence[str],
    memory_cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], bytes, bytes], Dict[str, np.ndarray]] | None,
    *,
    cache_dir: Path | None,
    show_hdf5_progress: bool,
) -> Dict[str, np.ndarray]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _kin_cache_path(run_dir, config_path, output_path, files, dataset_keys, cache_dir)
        if cache_path.exists():
            with np.load(cache_path) as cached:
                return {key: np.asarray(cached[key], dtype=np.float64) for key in dataset_keys}

    out = _load_vars_from_h5(
        records,
        files,
        dataset_keys,
        memory_cache,
        desc=f"HDF5 {run_dir.name}" if show_hdf5_progress else None,
    )
    if cache_path is not None:
        np.savez_compressed(cache_path, **out)
    return out


def load_kinematics_from_records(
    records: Sequence[Mapping[str, object]],
    source_files: Sequence[str],
    *,
    run_dir: Path,
    config_path: Path,
    output_path: Path,
    memory_cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], bytes, bytes], Dict[str, np.ndarray]] | None,
    cache_dir: Path | None,
    show_hdf5_progress: bool,
) -> Dict[str, np.ndarray]:
    n = len(records)
    out: Dict[str, np.ndarray] = {}
    aliases = {
        "E_gen": ("E_gen", "energy_true"),
        "theta": ("theta",),
        "phi": ("phi",),
    }
    missing_keys: List[str] = []
    for public_key, candidate_keys in aliases.items():
        values = np.full(n, np.nan, dtype=np.float64)
        found = False
        for idx, rec in enumerate(records):
            value = None
            for key in candidate_keys:
                if key in rec:
                    value = rec.get(key)
                    break
            if value is None:
                continue
            try:
                values[idx] = float(value)
                found = True
            except (TypeError, ValueError):
                continue
        out[public_key] = values
        if not found:
            missing_keys.append(KIN_VARS[public_key])

    if missing_keys:
        loaded = load_run_kinematics(
            run_dir,
            config_path,
            output_path,
            records,
            source_files,
            missing_keys,
            memory_cache,
            cache_dir=cache_dir,
            show_hdf5_progress=show_hdf5_progress,
        )
        for public_key, dataset_key in KIN_VARS.items():
            if dataset_key in loaded:
                out[public_key] = loaded[dataset_key]
    return out


def load_theta_tower_geometry(xml_path: Path) -> tuple[np.ndarray, float]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Tower geometry XML not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    barrel = root.find(".//barrel")
    endcap = root.find(".//endcap")
    if barrel is None or endcap is None:
        raise ValueError(f"Missing <barrel> or <endcap> section in {xml_path}")

    barrel_deltas: list[float] = []
    all_deltas: list[float] = []
    for component, target in ((barrel, barrel_deltas), (endcap, None)):
        for child in component.findall("deltatheta"):
            delta = float(child.attrib["deltatheta"])
            if delta <= 0.0:
                raise ValueError(f"Non-positive deltatheta in {xml_path}: {delta}")
            all_deltas.append(delta)
            if target is not None:
                target.append(delta)

    if not all_deltas:
        raise ValueError(f"No deltatheta entries found in {xml_path}")

    theta_mid = math.pi / 2.0
    hemi_desc = [theta_mid]
    current = theta_mid
    for delta in all_deltas:
        current -= delta
        hemi_desc.append(current)
    hemi_edges = np.asarray(sorted(hemi_desc), dtype=np.float64)
    hemi_edges = np.unique(np.round(hemi_edges, decimals=10))
    if hemi_edges.size < 2:
        raise ValueError(f"Failed to build theta tower edges from {xml_path}")

    barrel_theta_min = float(theta_mid - sum(barrel_deltas))
    return hemi_edges, barrel_theta_min


def load_theta_tower_edges(xml_path: Path, *, alignment: str = "edge") -> np.ndarray:
    hemi_edges, _ = load_theta_tower_geometry(xml_path)

    if alignment == "edge":
        return hemi_edges
    if alignment == "center":
        alignment = "shifted"
    if alignment != "shifted":
        raise ValueError(f"Unknown theta tower alignment: {alignment}")

    centers = 0.5 * (hemi_edges[:-1] + hemi_edges[1:])
    shifted = np.empty(centers.size + 1, dtype=np.float64)
    shifted[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    shifted[0] = max(0.0, float(centers[0] - 0.5 * (centers[1] - centers[0])))
    shifted[-1] = min(math.pi, float(centers[-1] + 0.5 * (centers[-1] - centers[-2])))
    shifted = np.unique(np.round(shifted, decimals=10))
    if shifted.size < 2:
        raise ValueError(f"Failed to build shifted theta tower edges from {xml_path}")
    return shifted


def choose_edges(
    kin_name: str,
    values: np.ndarray,
    *,
    bins: int,
    mode: str,
    theta_binning: str,
    theta_tower_edges: np.ndarray | None,
) -> np.ndarray:
    if kin_name == "theta" and theta_binning == "tower":
        if theta_tower_edges is None:
            raise ValueError("theta tower edges are required when --theta-binning tower is selected")
        return theta_tower_edges
    return make_bins(values, bins, mode)


def _efficiency_reference_lines(pair_name: str) -> list[tuple[float, str, str]]:
    return EFFICIENCY_REFERENCE_LINES.get(pair_name, [])


def _add_efficiency_reference_lines(ax: plt.Axes, pair_name: str) -> list[float]:
    levels: list[float] = []
    for level, color, label in _efficiency_reference_lines(pair_name):
        ax.axhline(level, color=color, linestyle="--", linewidth=1.4, alpha=0.95, label=label)
        levels.append(float(level))
    return levels


def _expand_auc_ylim(ymin: float, ymax: float, reference_levels: Sequence[float]) -> tuple[float, float]:
    lo = float(ymin)
    hi = float(ymax)
    if reference_levels:
        ref_lo = min(reference_levels)
        ref_hi = max(reference_levels)
        lo = min(lo, max(0.0, ref_lo - 0.02))
        hi = max(hi, min(1.0, ref_hi + 0.02))
    return lo, hi


def compute_bin_ids(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size < 2:
        return np.full(x.shape, -1, dtype=np.int32)
    out = np.full(x.shape, -1, dtype=np.int32)
    valid = np.isfinite(x) & (x >= edges[0]) & (x <= edges[-1])
    if np.any(valid):
        idx = np.searchsorted(edges, x[valid], side="right") - 1
        idx[idx == len(edges) - 1] = len(edges) - 2
        out[valid] = idx.astype(np.int32)
    return out


def binned_accuracy_from_bins(
    labels: np.ndarray,
    preds: np.ndarray,
    bin_ids: np.ndarray,
    centers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = centers.size
    if n_bins == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    values = np.full(n_bins, np.nan, dtype=np.float64)
    valid = bin_ids >= 0
    if not np.any(valid):
        return centers.copy(), values
    counts = np.bincount(bin_ids[valid], minlength=n_bins)
    correct = np.bincount(
        bin_ids[valid],
        weights=(preds[valid] == labels[valid]).astype(np.float64),
        minlength=n_bins,
    )
    nz = counts > 0
    values[nz] = correct[nz] / counts[nz]
    return centers.copy(), values


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def binned_pair_auc_from_bins(
    labels: np.ndarray,
    probs: np.ndarray,
    bin_ids: np.ndarray,
    centers: np.ndarray,
    class_to_label: Mapping[str, int],
    pos: str,
    neg: str,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bins = centers.size
    if n_bins == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    values = np.full(n_bins, np.nan, dtype=np.float64)
    errors = np.full(n_bins, np.nan, dtype=np.float64)
    if pos not in class_to_label or neg not in class_to_label:
        return centers.copy(), values, errors
    pos_idx = int(class_to_label[pos])
    neg_idx = int(class_to_label[neg])
    score = probs[:, pos_idx]
    mask = (bin_ids >= 0) & np.isin(labels, [pos_idx, neg_idx])
    if extra_mask is not None:
        mask = mask & extra_mask
    if not np.any(mask):
        return centers.copy(), values, errors

    for bin_idx in np.unique(bin_ids[mask]):
        m = mask & (bin_ids == bin_idx)
        if np.count_nonzero(m) < 2:
            continue
        y = (labels[m] == pos_idx).astype(np.int32)
        n_pos = int(np.count_nonzero(y == 1))
        n_neg = int(np.count_nonzero(y == 0))
        if n_pos == 0 or n_neg == 0:
            continue
        auc = float(roc_auc_score(y, score[m]))
        values[int(bin_idx)] = auc
        errors[int(bin_idx)] = auc_standard_error(auc, n_pos, n_neg)
    return centers.copy(), values, errors


def _fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    if y_true.size == 0 or scores.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    if tpr.size == 0 or fpr.size == 0:
        return float("nan")
    if target_tpr <= float(np.min(tpr)):
        return float(np.max(fpr))
    if target_tpr >= float(np.max(tpr)):
        return float(np.min(fpr))
    return float(np.interp(target_tpr, tpr, fpr))


def binned_pair_fpr_at_tpr_from_bins(
    labels: np.ndarray,
    probs: np.ndarray,
    bin_ids: np.ndarray,
    centers: np.ndarray,
    class_to_label: Mapping[str, int],
    pos: str,
    neg: str,
    *,
    target_tpr: float,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = centers.size
    if n_bins == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    values = np.full(n_bins, np.nan, dtype=np.float64)
    if pos not in class_to_label or neg not in class_to_label:
        return centers.copy(), values
    pos_idx = int(class_to_label[pos])
    neg_idx = int(class_to_label[neg])
    score = probs[:, pos_idx]
    mask = (bin_ids >= 0) & np.isin(labels, [pos_idx, neg_idx])
    if extra_mask is not None:
        mask = mask & extra_mask
    if not np.any(mask):
        return centers.copy(), values

    for bin_idx in np.unique(bin_ids[mask]):
        m = mask & (bin_ids == bin_idx)
        if np.count_nonzero(m) < 2:
            continue
        y = (labels[m] == pos_idx).astype(np.int32)
        if np.unique(y).size < 2:
            continue
        fpr_val = _fpr_at_tpr(y, score[m], target_tpr)
        if not np.isfinite(fpr_val):
            continue
        values[int(bin_idx)] = max(float(fpr_val), 1e-6)
    return centers.copy(), values


def bootstrap_auc_std(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> float:
    if n_bootstrap <= 0 or labels.size < 2 or scores.size < 2:
        return float("nan")
    if np.unique(labels).size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, labels.size, size=labels.size)
        y = labels[sample_idx]
        if np.unique(y).size < 2:
            continue
        try:
            stats.append(float(roc_auc_score(y, scores[sample_idx])))
        except ValueError:
            continue
    if len(stats) < 2:
        return float("nan")
    return float(np.std(np.asarray(stats, dtype=np.float64), ddof=1))


def _conditional_metric_grid(
    labels: np.ndarray,
    probs: np.ndarray,
    energy: np.ndarray,
    theta: np.ndarray,
    class_to_label: Mapping[str, int],
    *,
    pos_label: str,
    neg_label: str,
    energy_edges: np.ndarray,
    theta_edges: np.ndarray,
    min_events: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_theta = theta_edges.size - 1
    n_energy = energy_edges.size - 1
    grid = np.full((n_theta, n_energy), np.nan, dtype=np.float64)
    errors = np.full((n_theta, n_energy), np.nan, dtype=np.float64)
    counts = np.zeros((n_theta, n_energy), dtype=np.int32)

    if pos_label not in class_to_label or neg_label not in class_to_label:
        return grid, errors, counts

    pos_idx = int(class_to_label[pos_label])
    neg_idx = int(class_to_label[neg_label])
    valid = (
        np.isfinite(energy)
        & np.isfinite(theta)
        & np.isin(labels, [pos_idx, neg_idx])
    )
    if not np.any(valid):
        return grid, errors, counts

    energy_ids = compute_bin_ids(energy, energy_edges)
    theta_ids = compute_bin_ids(theta, theta_edges)
    score = probs[:, pos_idx]

    for tbin in range(n_theta):
        for ebin in range(n_energy):
            mask = valid & (theta_ids == tbin) & (energy_ids == ebin)
            n_events = int(np.count_nonzero(mask))
            counts[tbin, ebin] = n_events
            if n_events < min_events:
                continue
            y = (labels[mask] == pos_idx).astype(np.int32)
            if np.unique(y).size < 2:
                continue
            try:
                auc = float(roc_auc_score(y, score[mask]))
            except ValueError:
                continue
            grid[tbin, ebin] = auc
            errors[tbin, ebin] = bootstrap_auc_std(
                y,
                score[mask],
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + 1000 * tbin + ebin,
            )
    return grid, errors, counts


def _format_bin_label(lo: float, hi: float) -> str:
    return f"[{lo:g}, {hi:g})"


def make_conditional_theta_edges(barrel_theta_min: float) -> np.ndarray:
    edges = np.asarray([0.15, 0.3, barrel_theta_min, 0.9, 1.57], dtype=np.float64)
    edges = np.unique(np.round(edges, decimals=10))
    if edges.size < 5:
        raise ValueError(
            f"Conditional theta edges collapsed after deduplication: {edges.tolist()}"
        )
    return edges


def _finite_value_limits(*arrays: np.ndarray, lower: float = 0.0, upper: float = 1.0) -> tuple[float, float]:
    finite_parts: list[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size > 0:
            finite_parts.append(flat)
    if not finite_parts:
        return lower, upper
    values = np.concatenate(finite_parts)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return lower, upper
    if lo == hi:
        pad = max(0.005, abs(lo) * 0.01)
    else:
        pad = max(0.005, 0.03 * (hi - lo))
    lo = max(lower, lo - pad)
    hi = min(upper, hi + pad)
    if hi <= lo:
        lo = max(lower, lo - 0.01)
        hi = min(upper, hi + 0.01)
    return lo, hi


def plot_conditional_auc_analysis(
    *,
    out_dir: Path,
    run_payloads: Sequence[Mapping[str, object]],
    labels_for_legend: Sequence[str],
    pos_label: str,
    neg_label: str,
    barrel_theta_min: float,
    min_events: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    energy_curve_bins: int,
    theta_curve_bins: int,
) -> Dict[str, object]:
    energy_edges = FIXED_ENERGY_EDGES.copy()
    theta_edges = make_conditional_theta_edges(barrel_theta_min)
    energy_curve_edges = np.linspace(float(energy_edges[0]), float(energy_edges[-1]), int(energy_curve_bins) + 1)
    theta_curve_edges = np.linspace(float(theta_edges[0]), float(theta_edges[-1]), int(theta_curve_bins) + 1)
    energy_curve_centers = 0.5 * (energy_curve_edges[:-1] + energy_curve_edges[1:])
    theta_curve_centers = 0.5 * (theta_curve_edges[:-1] + theta_curve_edges[1:])

    pair_name = pair_key(pos_label, neg_label)
    safe_pair = safe_name(pair_name)
    run_grids: list[Dict[str, object]] = []
    for run_idx, (label, payload) in enumerate(zip(labels_for_legend, run_payloads)):
        coarse_grid, coarse_err, coarse_counts = _conditional_metric_grid(
            payload["labels"],
            payload["probs"],
            payload["E_gen"],
            payload["theta"],
            payload["class_to_label"],
            pos_label=pos_label,
            neg_label=neg_label,
            energy_edges=energy_edges,
            theta_edges=theta_edges,
            min_events=min_events,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed + 17_389 * (run_idx + 1),
        )
        run_grids.append(
            {
                "label": label,
                "grid": coarse_grid,
                "err": coarse_err,
                "counts": coarse_counts,
            }
        )

    # AUC vs energy: one subplot per theta bin, with both runs overlaid.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    energy_all_values: list[np.ndarray] = []
    energy_all_errors: list[np.ndarray] = []
    for tbin, ax in enumerate(axes_flat):
        theta_lo = theta_edges[tbin]
        theta_hi = theta_edges[tbin + 1]
        ax.set_title(f"$\\theta$ {_format_bin_label(theta_lo, theta_hi)}", fontsize=14)
        subplot_values: list[np.ndarray] = []
        subplot_errors: list[np.ndarray] = []
        for run_idx, run in enumerate(run_payloads):
            grid, err, _ = _conditional_metric_grid(
                run["labels"],
                run["probs"],
                run["E_gen"],
                run["theta"],
                run["class_to_label"],
                pos_label=pos_label,
                neg_label=neg_label,
                energy_edges=energy_curve_edges,
                theta_edges=np.asarray([theta_lo, theta_hi], dtype=np.float64),
                min_events=min_events,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed + 17 * (tbin + 1) + 50_003 * (run_idx + 1),
            )
            values = grid[0]
            yerr = err[0]
            subplot_values.append(values)
            subplot_errors.append(yerr)
            ax.errorbar(
                energy_curve_centers,
                values,
                yerr=yerr,
                fmt="-o",
                linewidth=1.5,
                markersize=4,
                capsize=2.0,
                label=labels_for_legend[run_idx],
                )
        energy_all_values.extend(subplot_values)
        energy_all_errors.extend(subplot_errors)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("AUC")
        ax.tick_params(axis="both", labelsize=12)
    axes_flat[0].legend(fontsize=11)
    energy_ymin, energy_ymax = _finite_value_limits(
        *(v - e for v, e in zip(energy_all_values, energy_all_errors)),
        *(v + e for v, e in zip(energy_all_values, energy_all_errors)),
        lower=0.0,
        upper=1.0,
    )
    for ax in axes_flat:
        ax.set_ylim(energy_ymin, energy_ymax)
    fig.suptitle(f"{PAIR_LABELS.get(pair_name, pair_name)} AUC vs Energy by $\\theta$", fontsize=15)
    fig.tight_layout()
    auc_energy_path = out_dir / f"{safe_pair}_auc_vs_energy_by_theta.png"
    fig.savefig(auc_energy_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {auc_energy_path}")

    # AUC vs theta: one subplot per energy bin, with both runs overlaid.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    theta_all_values: list[np.ndarray] = []
    theta_all_errors: list[np.ndarray] = []
    for ebin, ax in enumerate(axes_flat):
        energy_lo = energy_edges[ebin]
        energy_hi = energy_edges[ebin + 1]
        ax.set_title(f"Energy {_format_bin_label(energy_lo, energy_hi)} GeV", fontsize=14)
        subplot_values = []
        subplot_errors = []
        for run_idx, run in enumerate(run_payloads):
            grid, err, _ = _conditional_metric_grid(
                run["labels"],
                run["probs"],
                run["E_gen"],
                run["theta"],
                run["class_to_label"],
                pos_label=pos_label,
                neg_label=neg_label,
                energy_edges=np.asarray([energy_lo, energy_hi], dtype=np.float64),
                theta_edges=theta_curve_edges,
                min_events=min_events,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed + 31 * (ebin + 1) + 50_003 * (run_idx + 1),
            )
            values = grid[:, 0]
            yerr = err[:, 0]
            subplot_values.append(values)
            subplot_errors.append(yerr)
            ax.errorbar(
                theta_curve_centers,
                values,
                yerr=yerr,
                fmt="-o",
                linewidth=1.5,
                markersize=4,
                capsize=2.0,
                label=labels_for_legend[run_idx],
                )
        theta_all_values.extend(subplot_values)
        theta_all_errors.extend(subplot_errors)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel(r"$\theta$ [rad]")
        ax.set_ylabel("AUC")
        ax.tick_params(axis="both", labelsize=12)
    axes_flat[0].legend(fontsize=11)
    theta_ymin, theta_ymax = _finite_value_limits(
        *(v - e for v, e in zip(theta_all_values, theta_all_errors)),
        *(v + e for v, e in zip(theta_all_values, theta_all_errors)),
        lower=0.0,
        upper=1.0,
    )
    for ax in axes_flat:
        ax.set_ylim(theta_ymin, theta_ymax)
    fig.suptitle(f"{PAIR_LABELS.get(pair_name, pair_name)} AUC vs $\\theta$ by Energy", fontsize=15)
    fig.tight_layout()
    auc_theta_path = out_dir / f"{safe_pair}_auc_vs_theta_by_energy.png"
    fig.savefig(auc_theta_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {auc_theta_path}")

    # Optional extension point:
    # If you have a per-event timing observable (e.g. t50, t90-t10, timing RMS),
    # compute the same fixed-bin grid here and mirror the heatmap / curve plots above.
    # Keep the arrays aligned with `labels`, `E_gen`, and `theta` before calling this hook.

    return {
        "pair": pair_name,
        "energy_edges": energy_edges.tolist(),
        "theta_edges": theta_edges.tolist(),
        "energy_curve_edges": energy_curve_edges.tolist(),
        "theta_curve_edges": theta_curve_edges.tolist(),
        "runs": [
            {
                "label": item["label"],
                "grid": np.asarray(item["grid"], dtype=np.float64).tolist(),
                "err": np.asarray(item["err"], dtype=np.float64).tolist(),
                "counts": np.asarray(item["counts"], dtype=np.int64).tolist(),
            }
            for item in run_grids
        ],
    }


def draw_theta_bin_width_guides(ax: plt.Axes, edges: np.ndarray) -> None:
    if edges.size < 2:
        return
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    y = 0.035
    for lo, hi in zip(edges[:-1], edges[1:]):
        width = float(hi - lo)
        if width <= 0.0:
            continue
        pad = min(width * 0.08, 0.0025)
        left = float(lo + pad)
        right = float(hi - pad)
        if right <= left:
            left, right = float(lo), float(hi)
        ax.plot([left, right], [y, y], color="black", linewidth=1.2, alpha=0.65, transform=trans, solid_capstyle="butt")


def draw_theta_empty_bin_markers(ax: plt.Axes, centers: np.ndarray, values: np.ndarray) -> None:
    if centers.size == 0 or values.size == 0:
        return
    empty = np.isnan(values)
    if not np.any(empty):
        return
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        centers[empty],
        np.full(np.count_nonzero(empty), 0.08, dtype=np.float64),
        linestyle='None',
        marker='x',
        markersize=4.5,
        markeredgewidth=0.9,
        color='crimson',
        alpha=0.75,
        transform=trans,
        clip_on=False,
    )


def draw_theta_region_labels(
    ax: plt.Axes,
    theta_xlim: tuple[float, float],
    theta_min: float,
    theta_max: float,
) -> None:
    x0, x1 = theta_xlim
    span = max(x1 - x0, 1e-9)
    min_width = span * 0.06
    segments = [
        ("Endcap", x0, min(theta_min, x1)),
        ("Barrel", max(theta_min, x0), min(theta_max, x1)),
        ("Endcap", max(theta_max, x0), x1),
    ]
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for label, lo, hi in segments:
        if hi - lo < min_width:
            continue
        ax.text(
            0.5 * (lo + hi),
            0.97,
            label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
            clip_on=False,
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.out_subdir:
        out_dir = out_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)

    run_names: List[str] = []
    cfgs: List[Dict[str, object]] = []
    run_payloads: List[Dict[str, object]] = []
    var_cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], bytes, bytes], Dict[str, np.ndarray]] = {}
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    for raw in tqdm(args.run_dirs, desc="Loading runs", unit="run"):
        t0 = time.perf_counter()
        run_dir = resolve_run_dir(raw, base_dir)
        config_path = run_dir / "config.json"
        output_path = run_dir / "output.json"
        if not (config_path.exists() and output_path.exists()):
            raise FileNotFoundError(f"{run_dir} must contain config.json/output.json")

        cfg = json.loads(config_path.read_text())
        records = json.loads(output_path.read_text())
        labels = np.asarray([int(r["label"]) for r in records], dtype=np.int64)
        logits = np.asarray([r["logits"] for r in records], dtype=np.float64)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        source_files = cfg.get("test_files") or cfg.get("train_files") or []
        class_to_label = infer_class_to_label(records, source_files)

        # Prefer test_files if explicitly provided; fallback to train_files mapping.
        print(f"Loading {run_dir.name}: {len(records)} records")
        kin_values = load_kinematics_from_records(
            records,
            source_files,
            run_dir=run_dir,
            config_path=config_path,
            output_path=output_path,
            memory_cache=var_cache,
            cache_dir=cache_dir,
            show_hdf5_progress=args.show_hdf5_progress,
        )
        e_gen = kin_values["E_gen"]
        theta = kin_values["theta"]
        phi = kin_values["phi"]

        run_names.append(run_dir.name)
        cfgs.append(cfg)
        run_payloads.append(
            {
                "labels": labels,
                "logits": logits,
                "preds": np.argmax(logits, axis=1),
                "probs": probs,
                "class_to_label": class_to_label,
                "E_gen": e_gen,
                "theta": theta,
                "phi": phi,
            }
        )
        print(f"Loaded {run_dir.name} in {time.perf_counter() - t0:.1f}s")

    diff_keys = differing_config_keys(cfgs, CONFIG_DIFF_EXCLUDE)
    if args.legend_labels:
        if len(args.legend_labels) != len(run_names):
            raise ValueError(
                f"--legend-labels must match --run-dirs count "
                f"({len(args.legend_labels)} != {len(run_names)})"
            )
        labels_for_legend = list(args.legend_labels)
    else:
        labels_for_legend = [
            config_label(name, cfg, diff_keys, width=44) for name, cfg in zip(run_names, cfgs)
        ]

    # Barrel/endcap split in theta (radians).
    theta_min = float(args.barrel_theta_min)
    theta_max = float(args.barrel_theta_max)
    theta_tower_edges = None
    if args.theta_binning == "tower":
        tower_xml = Path(args.tower_geometry_xml)
        _, tower_theta_min = load_theta_tower_geometry(tower_xml)
        theta_min = tower_theta_min
        theta_max = math.pi - tower_theta_min
        theta_tower_edges = load_theta_tower_edges(
            tower_xml,
            alignment=args.theta_tower_alignment,
        )
    if not (0.0 <= theta_min < theta_max <= math.pi):
        raise ValueError(
            f"Invalid barrel theta range: [{theta_min}, {theta_max}] must satisfy 0 <= min < max <= pi."
        )

    theta_finite = [payload["theta"][np.isfinite(payload["theta"])] for payload in run_payloads if payload["theta"].size > 0]
    theta_xlim = None
    if theta_finite:
        theta_all = np.concatenate(theta_finite)
        if theta_all.size > 0:
            theta_lo = float(np.min(theta_all))
            theta_hi = float(np.max(theta_all))
            if theta_lo == theta_hi:
                pad = max(1e-3, abs(theta_lo) * 0.01)
            else:
                pad = (theta_hi - theta_lo) * 0.02
            theta_xlim = (theta_lo - pad, theta_hi + pad)
            if args.theta_plot_max is not None:
                theta_xlim = (theta_xlim[0], min(theta_xlim[1], float(args.theta_plot_max)))

    for payload in run_payloads:
        bin_cache: Dict[str, Dict[str, np.ndarray]] = {}
        for kin_name in ("E_gen", "theta", "phi"):
            x = payload[kin_name]
            edges = choose_edges(
                kin_name,
                x,
                bins=args.bins,
                mode=args.binning,
                theta_binning=args.theta_binning,
                theta_tower_edges=theta_tower_edges,
            )
            centers = 0.5 * (edges[:-1] + edges[1:]) if edges.size >= 2 else np.array([], dtype=np.float64)
            bin_cache[kin_name] = {
                "edges": edges,
                "centers": centers,
                "bin_ids": compute_bin_ids(x, edges),
            }
        payload["bin_cache"] = bin_cache

    # Plot overall accuracy vs each kinematic variable.
    for kin_name in tqdm(("E_gen", "theta", "phi"), desc="Accuracy plots", unit="plot"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, payload in zip(labels_for_legend, run_payloads):
            cached = payload["bin_cache"][kin_name]
            edges = cached["edges"]
            if edges.size < 2:
                continue
            xc, yc = binned_accuracy_from_bins(
                payload["labels"],
                payload["preds"],
                cached["bin_ids"],
                cached["centers"],
            )
            if xc.size == 0:
                continue
            ax.plot(xc, yc, marker="o", linewidth=1.5, markersize=4.5, alpha=0.75, label=label)
            if kin_name == "theta" and args.theta_binning == "tower":
                draw_theta_empty_bin_markers(ax, xc, yc)
        ax.set_xlabel(kin_name)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs {kin_name}")
        if kin_name == "theta" and theta_xlim is not None:
            theta_center = math.pi / 2.0
            if theta_xlim[0] <= theta_min <= theta_xlim[1]:
                ax.axvline(theta_min, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
            if theta_xlim[0] <= theta_center <= theta_xlim[1]:
                ax.axvline(theta_center, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
            if theta_xlim[0] <= theta_max <= theta_xlim[1]:
                ax.axvline(theta_max, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
            if args.theta_binning == "tower":
                draw_theta_bin_width_guides(ax, edges)
            draw_theta_region_labels(ax, theta_xlim, theta_min, theta_max)
            ax.set_xlim(*theta_xlim)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=11, labelspacing=0.9)
        fig.tight_layout()
        out_path = out_dir / f"accuracy_vs_{kin_name}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Wrote {out_path}")

    # Plot pairwise ROC-AUC vs each variable.
    pair_jobs = [(kin_name, pos, neg) for kin_name in ("E_gen", "theta", "phi") for pos, neg in PAIR_SPECS]
    for kin_name, pos, neg in tqdm(pair_jobs, desc="Pairwise plots", unit="plot"):
            key = pair_key(pos, neg)
            fig, ax = plt.subplots(figsize=(8, 5))
            for label, payload in zip(labels_for_legend, run_payloads):
                cached = payload["bin_cache"][kin_name]
                edges = cached["edges"]
                if edges.size < 2:
                    continue
                xc, yc, ye = binned_pair_auc_from_bins(
                    payload["labels"],
                    payload["probs"],
                    cached["bin_ids"],
                    cached["centers"],
                    payload["class_to_label"],
                    pos,
                    neg,
                )
                if xc.size == 0:
                    continue
                line, = ax.plot(xc, yc, marker="o", linewidth=1.5, markersize=4.5, alpha=0.75, label=label)
                finite = np.isfinite(yc) & np.isfinite(ye)
                if np.any(finite):
                    ax.errorbar(
                        xc[finite],
                        yc[finite],
                        yerr=ye[finite],
                        fmt='none',
                        ecolor=line.get_color(),
                        elinewidth=0.85,
                        capsize=1.8,
                        alpha=0.8,
                    )
                if kin_name == "theta" and args.theta_binning == "tower":
                    draw_theta_empty_bin_markers(ax, xc, yc)
            ax.set_xlabel(kin_name)
            ax.set_ylabel("ROC-AUC")
            ax.set_title(f"{PAIR_LABELS.get(key, key)} AUC vs {kin_name}")
            if kin_name == "theta" and theta_xlim is not None:
                theta_center = math.pi / 2.0
                if theta_xlim[0] <= theta_min <= theta_xlim[1]:
                    ax.axvline(theta_min, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
                if theta_xlim[0] <= theta_center <= theta_xlim[1]:
                    ax.axvline(theta_center, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
                if theta_xlim[0] <= theta_max <= theta_xlim[1]:
                    ax.axvline(theta_max, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
                if args.theta_binning == "tower":
                    draw_theta_bin_width_guides(ax, edges)
                draw_theta_region_labels(ax, theta_xlim, theta_min, theta_max)
                ax.set_xlim(*theta_xlim)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=11, labelspacing=0.9)
            fig.tight_layout()
            safe_pair = safe_name(key)
            out_path = out_dir / f"{safe_pair}_auc_vs_{kin_name}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Wrote {out_path}")

    # Plot pairwise FPR at fixed TPR vs each variable.
    tpr_jobs = [
        (kin_name, pos, neg, target_tpr)
        for kin_name in ("E_gen", "theta", "phi")
        for pos, neg in PAIR_SPECS
        for target_tpr in FIXED_TPR_TARGETS.get(pair_key(pos, neg), [])
    ]
    for kin_name, pos, neg, target_tpr in tqdm(tpr_jobs, desc="Operating point plots", unit="plot"):
            key = pair_key(pos, neg)
            fig, ax = plt.subplots(figsize=(8, 5))
            all_values: list[np.ndarray] = []
            for label, payload in zip(labels_for_legend, run_payloads):
                cached = payload["bin_cache"][kin_name]
                edges = cached["edges"]
                if edges.size < 2:
                    continue
                xc, yc = binned_pair_fpr_at_tpr_from_bins(
                    payload["labels"],
                    payload["probs"],
                    cached["bin_ids"],
                    cached["centers"],
                    payload["class_to_label"],
                    pos,
                    neg,
                    target_tpr=target_tpr,
                )
                if xc.size == 0:
                    continue
                all_values.append(yc)
                ax.plot(xc, yc, marker="o", linewidth=1.5, markersize=4.5, alpha=0.75, label=label)
                if kin_name == "theta" and args.theta_binning == "tower":
                    draw_theta_empty_bin_markers(ax, xc, yc)
            ax.set_xlabel(kin_name)
            ax.set_ylabel(f"FPR @ TPR={target_tpr:.0%}")
            ax.set_yscale("log")
            ax.set_title(f"{PAIR_LABELS.get(key, key)} operating point vs {kin_name}")
            if kin_name == "theta" and theta_xlim is not None:
                theta_center = math.pi / 2.0
                if theta_xlim[0] <= theta_min <= theta_xlim[1]:
                    ax.axvline(theta_min, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
                if theta_xlim[0] <= theta_center <= theta_xlim[1]:
                    ax.axvline(theta_center, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
                if theta_xlim[0] <= theta_max <= theta_xlim[1]:
                    ax.axvline(theta_max, color="gray", linestyle=":", linewidth=1.2, alpha=0.9)
                if args.theta_binning == "tower":
                    draw_theta_bin_width_guides(ax, edges)
                draw_theta_region_labels(ax, theta_xlim, theta_min, theta_max)
                ax.set_xlim(*theta_xlim)
            y_min, y_max = _finite_value_limits(*all_values, lower=1e-6, upper=1.0)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=11, labelspacing=0.9)
            fig.tight_layout()
            safe_pair = safe_name(key)
            out_path = out_dir / f"{safe_pair}_fpr_at_tpr{int(round(target_tpr * 100)):02d}_vs_{kin_name}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Wrote {out_path}")

    # Dedicated diagnostic: pi0 vs gamma AUC vs E_gen split by barrel/endcap.
    pair_pos, pair_neg = "pi0", "gamma"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    region_specs = [
        ("barrel", f"Barrel: {theta_min:.3f} <= theta <= {theta_max:.3f} rad"),
        ("endcap", f"Endcap: theta < {theta_min:.3f} or theta > {theta_max:.3f} rad"),
    ]
    for ax, (region_key, region_title) in tqdm(
        list(zip(axes, region_specs)),
        desc="Region plots",
        unit="region",
    ):
        for label, payload in zip(labels_for_legend, run_payloads):
            x = payload["E_gen"]
            theta = payload["theta"]
            if x.size == 0 or theta.size == 0:
                continue
            if region_key == "barrel":
                region_mask = np.isfinite(theta) & (theta >= theta_min) & (theta <= theta_max)
            else:
                region_mask = np.isfinite(theta) & ((theta < theta_min) | (theta > theta_max))
            edges = make_bins(x[region_mask], args.bins, args.binning)
            if edges.size < 2:
                continue
            xc, yc, ye = binned_pair_auc_from_bins(
                payload["labels"],
                payload["probs"],
                compute_bin_ids(x, edges),
                0.5 * (edges[:-1] + edges[1:]),
                payload["class_to_label"],
                pair_pos,
                pair_neg,
                extra_mask=region_mask,
            )
            if xc.size == 0:
                continue
            line, = ax.plot(xc, yc, marker="o", linewidth=1.5, markersize=4.5, alpha=0.75, label=label)
            finite = np.isfinite(yc) & np.isfinite(ye)
            if np.any(finite):
                ax.errorbar(
                    xc[finite],
                    yc[finite],
                    yerr=ye[finite],
                    fmt='none',
                    ecolor=line.get_color(),
                    elinewidth=0.85,
                    capsize=1.8,
                alpha=0.8,
            )
        ax.set_xlabel("E_gen")
        ax.set_title(region_title)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(fontsize=11, labelspacing=0.9)
    fig.suptitle(f"{PAIR_LABELS['pi0 vs gamma']} AUC vs E_gen by region")
    fig.tight_layout()
    out_region = out_dir / "pi0_vs_gamma_auc_vs_E_gen_barrel_endcap.png"
    fig.savefig(out_region, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_region}")

    conditional_summary = None
    if len(run_payloads) >= 2:
        conditional_summary = [
            plot_conditional_auc_analysis(
                out_dir=out_dir,
                run_payloads=run_payloads,
                labels_for_legend=labels_for_legend,
                pos_label=pos_label,
                neg_label=neg_label,
                barrel_theta_min=theta_min,
                min_events=int(args.conditional_min_events),
                n_bootstrap=int(args.conditional_bootstrap_samples),
                bootstrap_seed=int(args.conditional_bootstrap_seed) + 50_000 * offset,
                energy_curve_bins=int(args.conditional_energy_curve_bins),
                theta_curve_bins=int(args.conditional_theta_curve_bins),
            )
            for offset, (pos_label, neg_label) in enumerate((("pi0", "gamma"), ("pi+", "e-")))
        ]

    # Save summary metadata.
    summary = {
        "runs": run_names,
        "config_diff_keys": diff_keys,
        "labels": labels_for_legend,
        "variables": list(KIN_VARS.keys()),
        "binning": args.binning,
        "bins": args.bins,
        "barrel_theta_min": theta_min,
        "barrel_theta_max": theta_max,
        "theta_binning": args.theta_binning,
        "theta_tower_alignment": (
            ("shifted" if args.theta_tower_alignment == "center" else args.theta_tower_alignment)
            if args.theta_binning == "tower" else None
        ),
        "tower_geometry_xml": args.tower_geometry_xml if args.theta_binning == "tower" else None,
        "theta_plot_max": args.theta_plot_max,
        "conditional_analysis": conditional_summary,
    }
    out_json = out_dir / "perf_by_kinematics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
