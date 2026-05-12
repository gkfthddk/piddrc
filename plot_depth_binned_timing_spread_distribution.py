#!/usr/bin/env python
"""Plot event-wise max depth-binned timing spread distributions for gamma and pi0."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plot_common import SIM3D_E, SIM3D_Z, default_sample_label, read_row, read_shifted_time_row, resolve_paths, shared_range
from utils.plot_helpers import set_publication_style

set_publication_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot event-wise max depth-binned timing spread distributions for gamma and pi0."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"],
        help="Input HDF5 files. Default compares gamma and pi0 samples.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --files order. Defaults to file stem names.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=20000,
        help="Maximum number of events to scan per file (<=0 means all).",
    )
    parser.add_argument(
        "--n-depth-bins",
        type=int,
        default=24,
        help="Number of fixed depth bins used within each event.",
    )
    parser.add_argument(
        "--min-hits-per-bin",
        type=int,
        default=5,
        help="Ignore depth bins with fewer than this many valid hits.",
    )
    parser.add_argument(
        "--min-valid-bins",
        type=int,
        default=3,
        help="Ignore events with fewer than this many valid depth bins.",
    )
    parser.add_argument(
        "--range-percentile",
        type=float,
        default=99.0,
        help="Central percentile span used to define the shared depth range.",
    )
    parser.add_argument(
        "--depth-range-events",
        type=int,
        default=5000,
        help="Number of early events per file used to estimate the shared depth range.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Fallback number of histogram bins if automatic binning is degenerate.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/depth_binned_timing_spread_distribution",
        help="Output directory.",
    )
    parser.add_argument(
        "--out-name",
        default="gamma_vs_pi0_depth_binned_timing_spread_max.png",
        help="Output filename.",
    )
    return parser.parse_args()


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    sw = float(np.sum(weights))
    if not np.isfinite(sw) or sw <= 0.0:
        return float("nan")
    weights = weights / sw
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    if not np.isfinite(var) or var < 0.0:
        return float("nan")
    return float(np.sqrt(var))


def _depth_range(
    file_paths: list[Path],
    labels: list[str],
    *,
    max_events: int,
    depth_range_events: int,
) -> tuple[float, float]:
    lo = np.inf
    hi = -np.inf
    for label, file_path in zip(labels, file_paths):
        with h5py.File(file_path, "r") as handle:
            if SIM3D_Z not in handle:
                raise KeyError(f"{SIM3D_Z} missing in {file_path}")
            n_events = int(handle[SIM3D_Z].shape[0])
            if int(max_events) <= 0:
                take = n_events
            else:
                take = min(n_events, int(max_events))
            if int(depth_range_events) > 0:
                take = min(take, int(depth_range_events))
            for idx in tqdm(range(take), desc=f"{label} depth range", unit="event", leave=False):
                z = read_row(handle, SIM3D_Z, idx)
                e = read_row(handle, SIM3D_E, idx)
                n = min(z.size, e.size)
                if n == 0:
                    continue
                z = z[:n]
                e = e[:n]
                valid = np.isfinite(z) & np.isfinite(e) & (e > 0.0)
                if not np.any(valid):
                    continue
                zv = np.asarray(z[valid], dtype=np.float64)
                lo = min(lo, float(np.min(zv)))
                hi = max(hi, float(np.max(zv)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0, 1.0
    pad = 0.05 * max(1.0, abs(hi - lo))
    return float(lo - pad), float(hi + pad)


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


def _event_depth_binned_timing_spread(
    handle: h5py.File,
    idx: int,
    *,
    min_hits_per_bin: int,
    min_valid_bins: int,
    depth_edges: np.ndarray,
) -> float:
    z = read_row(handle, SIM3D_Z, idx)
    t = read_shifted_time_row(handle, idx)
    e = read_row(handle, SIM3D_E, idx)
    n = min(z.size, t.size, e.size)
    if n < int(min_hits_per_bin):
        return float("nan")

    z = z[:n]
    t = t[:n]
    e = e[:n]
    valid = np.isfinite(z) & np.isfinite(t) & np.isfinite(e) & (e > 0.0)
    if int(np.sum(valid)) < int(min_hits_per_bin):
        return float("nan")

    z = z[valid]
    t = t[valid]
    w = e[valid].astype(np.float64, copy=False)
    if z.size < int(min_hits_per_bin):
        return float("nan")

    bin_ids = np.digitize(z, depth_edges, right=False) - 1
    spreads: list[float] = []
    for ibin in range(int(depth_edges.size - 1)):
        mask = bin_ids == ibin
        if int(np.sum(mask)) < int(min_hits_per_bin):
            continue
        tb = t[mask]
        wb = w[mask]
        sw = float(np.sum(wb))
        if not np.isfinite(sw) or sw <= 0.0:
            continue
        wb = wb / sw
        sigma_t = _weighted_std(tb, wb)
        if np.isfinite(sigma_t):
            spreads.append(float(sigma_t))

    if len(spreads) < int(min_valid_bins):
        return float("nan")
    return float(np.max(np.asarray(spreads, dtype=np.float64)))


def _collect_event_level_values(
    file_paths: list[Path],
    labels: list[str],
    *,
    max_events: int,
    min_hits_per_bin: int,
    min_valid_bins: int,
    depth_edges: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    series: list[tuple[str, np.ndarray]] = []
    for label, file_path in zip(labels, file_paths):
        residuals: list[float] = []
        with h5py.File(file_path, "r") as handle:
            n_events = int(handle[SIM3D_Z].shape[0])
            take = n_events if int(max_events) <= 0 else min(n_events, int(max_events))
            for idx in tqdm(range(take), desc=f"{label} events", unit="event", leave=False):
                r = _event_depth_binned_timing_spread(
                    handle,
                    idx,
                    min_hits_per_bin=min_hits_per_bin,
                    min_valid_bins=min_valid_bins,
                    depth_edges=depth_edges,
                )
                if np.isfinite(r):
                    residuals.append(float(r))
        series.append((label, np.asarray(residuals, dtype=np.float64)))
    return series


def _plot_hist(ax: plt.Axes, series: list[tuple[str, np.ndarray]], *, bins: np.ndarray, xlim: tuple[float, float]) -> None:
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
    ax.set_xlabel(r"Event-level timing spread $R_{\mathrm{event}}$ [ns]")
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

    depth_lo, depth_hi = _depth_range(
        files,
        labels,
        max_events=int(args.max_events),
        depth_range_events=int(args.depth_range_events),
    )
    depth_edges = np.linspace(depth_lo, depth_hi, int(args.n_depth_bins) + 1)

    series = _collect_event_level_values(
        files,
        labels,
        max_events=int(args.max_events),
        min_hits_per_bin=int(args.min_hits_per_bin),
        min_valid_bins=int(args.min_valid_bins),
        depth_edges=depth_edges,
    )
    all_values = [values for _, values in series if values.size > 0]
    xlim = shared_range(all_values, float(args.range_percentile))
    combined = np.concatenate(all_values) if all_values else np.array([], dtype=np.float64)
    bins = _automatic_bins(combined, xlim, int(args.bins))

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    _plot_hist(ax, series, bins=bins, xlim=xlim)
    ax.set_title(r"Max depth-binned timing spread: $\gamma$ vs $\pi^0$")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
