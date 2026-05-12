#!/usr/bin/env python
"""Plot nearest-neighbor distance distributions for gamma and pi0 samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_common import (
    ENERGY_KEY,
    PHI_KEY,
    SIM3D_E,
    SIM3D_X,
    SIM3D_Y,
    SIM3D_Z,
    THETA_KEY,
    TIME_END_KEY,
    TIME_KEY,
    VERTEX_X,
    VERTEX_Y,
    VERTEX_Z,
    default_sample_label,
    event_passes_selection,
    resolve_paths,
    unit_vector,
    tower_head_distance_mm,
)
from utils.plot_helpers import set_publication_style

set_publication_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot normalized nearest-neighbor distance distributions for gamma and pi0."
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=["h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"],
        help="Input HDF5 files. Default compares gamma and pi0 samples.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --files order. Defaults to file stem names.",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=40000,
        help="Maximum number of events to scan per file (<=0 means all).",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=20,
        help="Ignore events with fewer than this many valid hits.",
    )
    p.add_argument(
        "--min-hit-energy",
        type=float,
        default=0.5,
        help="Drop hits with amplitude_sum below this threshold.",
    )
    p.add_argument(
        "--min-theta",
        type=float,
        default=1.5,
        help="Only keep events with GenParticles.momentum.theta above this threshold.",
    )
    p.add_argument(
        "--min-e-gen",
        type=float,
        default=10.0,
        help="Only keep events with E_gen above this threshold in GeV.",
    )
    p.add_argument(
        "--range-percentile",
        type=float,
        default=99.5,
        help="Upper percentile used to set the shared histogram x-range.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Fallback number of histogram bins if automatic binning is degenerate.",
    )
    p.add_argument(
        "--out-dir",
        default="plots/nearest_neighbor_distance_distribution",
        help="Output directory.",
    )
    p.add_argument(
        "--out-name",
        default="gamma_vs_pi0_nearest_neighbor_distance.png",
        help="Output filename.",
    )
    return p.parse_args()


def _raw_row(handle: h5py.File, key: str, idx: int) -> np.ndarray:
    if key not in handle:
        return np.empty(0, dtype=np.float64)
    return np.asarray(handle[key][idx], dtype=np.float64).reshape(-1)


def _first_finite_value(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite[0])


def _projected_depths_for_event(
    handle: h5py.File,
    idx: int,
    *,
    min_hit_energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = [SIM3D_E, SIM3D_X, SIM3D_Y, SIM3D_Z, THETA_KEY, PHI_KEY, VERTEX_X, VERTEX_Y, VERTEX_Z]
    if any(key not in handle for key in required):
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    e = _raw_row(handle, SIM3D_E, idx)
    x = _raw_row(handle, SIM3D_X, idx)
    y = _raw_row(handle, SIM3D_Y, idx)
    z = _raw_row(handle, SIM3D_Z, idx)
    t0 = _raw_row(handle, TIME_KEY, idx)
    t1 = _raw_row(handle, TIME_END_KEY, idx) if TIME_END_KEY in handle else np.empty(0, dtype=np.float64)

    n = min(e.size, x.size, y.size, z.size, t0.size)
    if n < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    e = e[:n]
    x = x[:n]
    y = y[:n]
    z = z[:n]
    t0 = t0[:n]
    if t1.size >= n:
        t1 = t1[:n]
        time = 0.5 * (t0 + t1)
        valid_time = np.isfinite(t0) & np.isfinite(t1) & np.isfinite(time)
    else:
        time = t0
        valid_time = np.isfinite(time)

    valid = (
        np.isfinite(e)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & valid_time
        & (e > float(min_hit_energy))
    )
    if int(np.sum(valid)) < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    theta = _first_finite_value(_raw_row(handle, THETA_KEY, idx))
    phi = _first_finite_value(_raw_row(handle, PHI_KEY, idx))
    vx = _first_finite_value(_raw_row(handle, VERTEX_X, idx))
    vy = _first_finite_value(_raw_row(handle, VERTEX_Y, idx))
    vz = _first_finite_value(_raw_row(handle, VERTEX_Z, idx))
    if not np.all(np.isfinite([theta, phi, vx, vy, vz])):
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    axis = unit_vector(theta, phi)
    head = tower_head_distance_mm(theta)
    if not np.isfinite(head):
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    x = x[valid]
    y = y[valid]
    z = z[valid]
    time = time[valid]
    e = e[valid]

    pos = np.column_stack([x, y, z])
    depth = (pos - np.asarray([vx, vy, vz], dtype=np.float64)) @ axis - head
    if depth.size < 2 or time.size < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    time = time - float(np.min(time))
    return (
        np.asarray(depth, dtype=np.float64),
        np.asarray(time, dtype=np.float64),
        np.asarray(e, dtype=np.float64),
    )


def _event_weighted_distance(
    handle: h5py.File,
    idx: int,
    *,
    min_hits: int,
    min_hit_energy: float,
) -> float:
    depth, time, energy = _projected_depths_for_event(
        handle,
        idx,
        min_hit_energy=min_hit_energy,
    )
    if depth.size < int(min_hits) or time.size < int(min_hits) or energy.size < int(min_hits):
        return float("nan")

    z_scale = float(np.std(depth, ddof=0))
    t_scale = float(np.std(time, ddof=0))
    if not np.isfinite(z_scale) or not np.isfinite(t_scale) or z_scale <= 0.0 or t_scale <= 0.0:
        return float("nan")

    valid = np.isfinite(depth) & np.isfinite(time) & np.isfinite(energy) & (energy > 0.0)
    if int(np.sum(valid)) < int(min_hits):
        return float("nan")
    depth = depth[valid]
    time = time[valid]
    weights = energy[valid]
    points = np.column_stack([depth / z_scale, time / t_scale])
    if points.shape[0] < int(min_hits) or points.shape[0] < 2:
        return float("nan")

    sw = float(np.sum(weights))
    if not np.isfinite(sw) or sw <= 0.0:
        return float("nan")
    weights = weights / sw

    tree = KDTree(points, leaf_size=40, metric="euclidean")
    distances, _ = tree.query(points, k=2)
    if distances.shape[1] < 2:
        return float("nan")
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    valid = np.isfinite(nearest) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    nearest = nearest[valid]
    weights = weights[valid]
    sw = float(np.sum(weights))
    if not np.isfinite(sw) or sw <= 0.0:
        return float("nan")
    weights = weights / sw
    return float(np.sum(weights * nearest))


def _shared_range(arrays: list[np.ndarray], percentile: float) -> tuple[float, float]:
    finite = [np.asarray(a, dtype=np.float64).ravel() for a in arrays if np.asarray(a).size > 0]
    finite = [a[np.isfinite(a)] for a in finite if np.isfinite(a).any()]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    if values.size == 0:
        return 0.0, 1.0
    hi = float(np.percentile(values, float(np.clip(percentile, 0.0, 100.0))))
    if not np.isfinite(hi) or hi <= 0.0:
        max_val = float(np.max(values))
        hi = max_val if np.isfinite(max_val) and max_val > 0.0 else 1.0
    return 0.0, hi


def _automatic_bins(values: np.ndarray, xlim: tuple[float, float], fallback_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(xlim[0], xlim[1], fallback_bins + 1)
    clipped = finite[(finite >= xlim[0]) & (finite <= xlim[1])]
    source = clipped if clipped.size >= 2 else finite
    try:
        edges = np.histogram_bin_edges(source, bins="auto", range=xlim)
    except ValueError:
        edges = np.linspace(xlim[0], xlim[1], fallback_bins + 1)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(xlim[0], xlim[1], fallback_bins + 1)
    return edges


def _format_label(label: str, mean: float, rms: float) -> str:
    return f"{label}  (μ={mean:.4g}, RMS={rms:.4g})"


def _plot_hist(
    ax: plt.Axes,
    series: list[tuple[str, np.ndarray]],
    *,
    bins: np.ndarray,
    xlim: tuple[float, float],
) -> None:
    for label, values in series:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean = float(np.mean(arr))
        rms = float(np.std(arr, ddof=0))
        ax.hist(
            arr,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=_format_label(label, mean, rms),
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Event-level weighted nearest-neighbor distance $D_{\mathrm{event}}$")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(*xlim)
    ax.legend(loc="best", fontsize=11)


def main() -> None:
    args = parse_args()
    files = resolve_paths(args.files)
    if not files:
        raise FileNotFoundError("No input files found.")
    if args.labels is not None and len(args.labels) != len(files):
        raise ValueError("--labels must match --files count.")

    labels = list(args.labels) if args.labels is not None else [default_sample_label(Path(str(p))) for p in files]

    series: list[tuple[str, np.ndarray]] = []
    for label, file_path in zip(labels, files):
        distances: list[float] = []
        with h5py.File(file_path, "r") as handle:
            if THETA_KEY not in handle or ENERGY_KEY not in handle:
                raise KeyError(f"Missing required event-level keys in {file_path}")
            n_events = int(handle[THETA_KEY].shape[0])
            take = n_events if int(args.max_events) <= 0 else min(n_events, int(args.max_events))
            for idx in tqdm(range(take), desc=f"{label}", unit="event", leave=False):
                if not event_passes_selection(
                    handle,
                    idx,
                    min_theta=float(args.min_theta),
                    min_e_gen=float(args.min_e_gen),
                ):
                    continue
                d = _event_weighted_distance(
                    handle,
                    idx,
                    min_hits=int(args.min_hits),
                    min_hit_energy=float(args.min_hit_energy),
                )
                if np.isfinite(d):
                    distances.append(float(d))
        if distances:
            series.append((label, np.asarray(distances, dtype=np.float64)))
        else:
            series.append((label, np.asarray([], dtype=np.float64)))

    all_distances = [dist for _, dist in series if dist.size > 0]
    if not all_distances:
        raise RuntimeError("No valid nearest-neighbor distances found.")

    xlim = _shared_range(all_distances, float(args.range_percentile))
    combined = np.concatenate(all_distances)
    bins = _automatic_bins(combined, xlim, int(args.bins))

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    _plot_hist(ax, series, bins=bins, xlim=xlim)
    ax.set_title(r"Energy-weighted nearest-neighbor structure: $\gamma$ vs $\pi^0$")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
