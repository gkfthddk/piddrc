#!/usr/bin/env python
"""Plot event-wise weighted depth-time residual distributions for e-, gamma, and pi0."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plot_common import SIM3D_E, SIM3D_Z, default_sample_label, read_row, read_shifted_time_row, resolve_paths, weighted_quantile
from utils.plot_helpers import set_publication_style

set_publication_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot weighted linear-regression residual distributions for e-, gamma, and pi0."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["h5s/e-_1-120GeV.h5py", "h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"],
        help="Input HDF5 files. Default compares e-, gamma, and pi0 samples.",
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
        default=40000,
        help="Maximum number of events to scan per file (<=0 means all).",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=10,
        help="Ignore events with fewer than this many valid hits.",
    )
    parser.add_argument(
        "--trim-quantile",
        type=float,
        default=0.95,
        help="Residual quantile used for one robust refit pass.",
    )
    parser.add_argument(
        "--range-percentile",
        type=float,
        default=99.0,
        help="Central percentile span used to define the shared histogram range.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Fallback number of histogram bins if automatic binning is degenerate.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/depth_time_residual_distribution",
        help="Output directory.",
    )
    parser.add_argument(
        "--out-name",
        default="e_gamma_pi0_depth_time_residual.png",
        help="Output filename.",
    )
    return parser.parse_args()


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        return float("nan"), float("nan")
    w = w / sw
    xbar = float(np.sum(w * x))
    ybar = float(np.sum(w * y))
    dx = x - xbar
    dy = y - ybar
    denom = float(np.sum(w * dx * dx))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum(w * dx * dy) / denom)
    intercept = ybar - slope * xbar
    return slope, intercept


def _robust_weighted_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, *, trim_quantile: float) -> tuple[float, float]:
    slope, intercept = _weighted_linear_fit(x, y, w)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return float("nan"), float("nan")

    resid = y - (slope * x + intercept)
    abs_resid = np.abs(resid)
    cutoff = weighted_quantile(abs_resid, w, float(trim_quantile))
    if not np.isfinite(cutoff):
        return slope, intercept

    keep = np.isfinite(abs_resid) & (abs_resid <= cutoff)
    if int(np.sum(keep)) < 3 or int(np.sum(keep)) >= int(x.size):
        return slope, intercept

    slope2, intercept2 = _weighted_linear_fit(x[keep], y[keep], w[keep])
    if not np.isfinite(slope2) or not np.isfinite(intercept2):
        return slope, intercept
    return slope2, intercept2


def _event_residual(handle: h5py.File, idx: int, *, min_hits: int, trim_quantile: float) -> float:
    z = read_row(handle, SIM3D_Z, idx)
    t = read_shifted_time_row(handle, idx)
    e = read_row(handle, SIM3D_E, idx)
    n = min(z.size, t.size, e.size)
    if n < int(min_hits):
        return float("nan")

    z = z[:n]
    t = t[:n]
    e = e[:n]
    valid = np.isfinite(z) & np.isfinite(t) & np.isfinite(e) & (e > 0.0)
    if int(np.sum(valid)) < int(min_hits):
        return float("nan")

    z = z[valid]
    t = t[valid]
    w = e[valid].astype(np.float64, copy=False)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        return float("nan")
    w = w / sw

    slope, intercept = _robust_weighted_fit(z, t, w, trim_quantile=trim_quantile)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return float("nan")

    resid = t - (slope * z + intercept)
    if resid.size == 0:
        return float("nan")
    return float(np.sum(w * resid * resid))


def _shared_range(arrays: list[np.ndarray], percentile: float) -> tuple[float, float]:
    finite = [np.asarray(a, dtype=np.float64).ravel() for a in arrays if np.asarray(a).size > 0]
    finite = [a[np.isfinite(a)] for a in finite if np.isfinite(a).any()]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    if values.size == 0:
        return 0.0, 1.0
    pct = float(np.clip(percentile, 0.0, 100.0))
    lo_q = 0.5 * (100.0 - pct)
    hi_q = 100.0 - lo_q
    lo, hi = np.percentile(values, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.min(values))
        hi = float(np.max(values))
    if lo == hi:
        pad = max(1.0, abs(lo) * 0.05)
        lo -= pad
        hi += pad
    return float(lo), float(hi)


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
    ax.set_yscale("log")
    ax.set_xlabel(r"Weighted residual $R$ [ns$^2$]")
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

    residual_series: list[tuple[str, np.ndarray]] = []
    for label, file_path in zip(labels, files):
        residuals: list[float] = []
        with h5py.File(file_path, "r") as handle:
            n_events = int(handle[SIM3D_Z].shape[0])
            take = n_events if int(args.max_events) <= 0 else min(n_events, int(args.max_events))
            for idx in tqdm(range(take), desc=f"{label}", unit="event", leave=False):
                r = _event_residual(
                    handle,
                    idx,
                    min_hits=int(args.min_hits),
                    trim_quantile=float(args.trim_quantile),
                )
                if np.isfinite(r):
                    residuals.append(float(r))
        residual_series.append((label, np.asarray(residuals, dtype=np.float64)))

    all_residuals = [r for _, r in residual_series if r.size > 0]
    xlim = _shared_range(all_residuals, float(args.range_percentile))
    combined = np.concatenate(all_residuals) if all_residuals else np.array([], dtype=np.float64)
    bins = _automatic_bins(combined, xlim, int(args.bins))

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    _plot_hist(ax, residual_series, bins=bins, xlim=xlim)
    ax.set_title(r"Depth-time residual consistency: $e^-$ / $\gamma$ / $\pi^0$")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
