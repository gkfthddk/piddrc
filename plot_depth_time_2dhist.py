#!/usr/bin/env python
"""Plot depth-time density maps for gamma and pi0."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plot_common import SIM3D_E, SIM3D_Z, TIME_KEY, default_sample_label, projected_shower_depths_mm, read_row, resolve_paths
from utils.plot_helpers import set_publication_style

set_publication_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gamma and pi0 depth-time density maps.")
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
        default=10000,
        help="Maximum number of events to scan per file (<=0 means all).",
    )
    parser.add_argument(
        "--range-events",
        type=int,
        default=1000,
        help="Number of early events per file used to estimate common axis ranges.",
    )
    parser.add_argument(
        "--depth-range-percentile",
        type=float,
        default=99.0,
        help="Central percentile span used to define common depth range.",
    )
    parser.add_argument(
        "--time-range-percentile",
        type=float,
        default=99.0,
        help="Central percentile span used to define common timing range.",
    )
    parser.add_argument(
        "--bins-x",
        type=int,
        default=40,
        help="Number of depth bins.",
    )
    parser.add_argument(
        "--bins-y",
        type=int,
        default=40,
        help="Number of timing bins.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian smoothing sigma in bins. Default disables smoothing.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/depth_time_2dhist",
        help="Output directory.",
    )
    parser.add_argument(
        "--out-name",
        default="gamma_vs_pi0_depth_time_density.png",
        help="Output filename.",
    )
    return parser.parse_args()


def _estimate_ranges(
    file_paths: list[Path],
    labels: list[str],
    *,
    max_events: int,
    range_events: int,
    depth_percentile: float,
    time_percentile: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    depth_values: list[np.ndarray] = []
    time_values: list[np.ndarray] = []
    for label, file_path in zip(labels, file_paths):
        with h5py.File(file_path, "r") as handle:
            if SIM3D_Z not in handle:
                raise KeyError(f"{SIM3D_Z} missing in {file_path}")
            n_events = int(handle[SIM3D_Z].shape[0])
            take = n_events if int(max_events) <= 0 else min(n_events, int(max_events))
            if int(range_events) > 0:
                take = min(take, int(range_events))
            for idx in tqdm(range(take), desc=f"{label} range", unit="event", leave=False):
                depth, e = projected_shower_depths_mm(handle, idx)
                t = read_row(handle, TIME_KEY, idx)
                n = min(depth.size, t.size, e.size)
                if n == 0:
                    continue
                depth = depth[:n]
                t = t[:n]
                e = e[:n]
                valid = np.isfinite(depth) & np.isfinite(t) & np.isfinite(e) & (e > 0.0)
                if not np.any(valid):
                    continue
                depth_values.append(np.asarray(depth[valid], dtype=np.float64))
                tv = np.asarray(t[valid], dtype=np.float64)
                if tv.size == 0:
                    continue
                time_values.append(tv - float(np.min(tv)))

    def _percentile_range(values: list[np.ndarray], percentile: float) -> tuple[float, float]:
        finite = [np.asarray(v, dtype=np.float64).ravel() for v in values if np.asarray(v).size > 0]
        finite = [v[np.isfinite(v)] for v in finite if np.isfinite(v).any()]
        if not finite:
            return 0.0, 1.0
        merged = np.concatenate(finite)
        if merged.size == 0:
            return 0.0, 1.0
        pct = float(np.clip(percentile, 0.0, 100.0))
        lo_q = 0.5 * (100.0 - pct)
        hi_q = 100.0 - lo_q
        lo, hi = np.percentile(merged, [lo_q, hi_q])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.min(merged))
            hi = float(np.max(merged))
        if lo == hi:
            pad = max(1.0, abs(lo) * 0.05)
            lo -= pad
            hi += pad
        return float(lo), float(hi)

    return _percentile_range(depth_values, depth_percentile), _percentile_range(time_values, time_percentile)


def _accumulate_hist2d(
    file_path: Path,
    *,
    max_events: int,
    xbins: np.ndarray,
    ybins: np.ndarray,
) -> np.ndarray:
    hist = np.zeros((xbins.size - 1, ybins.size - 1), dtype=np.float64)
    with h5py.File(file_path, "r") as handle:
        if SIM3D_Z not in handle:
            raise KeyError(f"{SIM3D_Z} missing in {file_path}")
        n_events = int(handle[SIM3D_Z].shape[0])
        take = n_events if int(max_events) <= 0 else min(n_events, int(max_events))
        for idx in tqdm(range(take), desc=file_path.stem, unit="event", leave=False):
            depth, e = projected_shower_depths_mm(handle, idx)
            t = read_row(handle, TIME_KEY, idx)
            n = min(depth.size, t.size, e.size)
            if n == 0:
                continue
            depth = depth[:n]
            t = t[:n]
            e = e[:n]
            valid = np.isfinite(depth) & np.isfinite(t) & np.isfinite(e) & (e > 0.0)
            if not np.any(valid):
                continue
            zv = np.asarray(depth[valid], dtype=np.float64)
            tv = np.asarray(t[valid], dtype=np.float64)
            if tv.size == 0:
                continue
            tv -= float(np.min(tv))
            h, _, _ = np.histogram2d(zv, tv, bins=[xbins, ybins])
            hist += h
    total = float(np.sum(hist))
    if total > 0.0:
        hist /= total
    return hist


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    norm = float(np.sum(kernel))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    return kernel / norm


def _smooth_hist2d(hist: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return hist
    pad = kernel.size // 2
    padded = np.pad(hist, ((pad, pad), (pad, pad)), mode="edge")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, tmp)
    total = float(np.sum(smoothed))
    if total > 0.0:
        smoothed = smoothed / total
    return smoothed


def main() -> None:
    args = parse_args()
    files = resolve_paths(args.files)
    if not files:
        raise FileNotFoundError("No input files found.")
    if args.labels is not None and len(args.labels) != len(files):
        raise ValueError("--labels must match --files count.")

    labels = list(args.labels) if args.labels is not None else [default_sample_label(Path(str(p))) for p in files]

    depth_range, time_range = _estimate_ranges(
        files,
        labels,
        max_events=int(args.max_events),
        range_events=int(args.range_events),
        depth_percentile=float(args.depth_range_percentile),
        time_percentile=float(args.time_range_percentile),
    )

    xbins = np.linspace(depth_range[0], depth_range[1], int(args.bins_x) + 1)
    ybins = np.linspace(time_range[0], time_range[1], int(args.bins_y) + 1)

    histograms = []
    for file_path in files:
        hist = _accumulate_hist2d(file_path, max_events=int(args.max_events), xbins=xbins, ybins=ybins)
        if float(args.gaussian_sigma) > 0.0:
            hist = _smooth_hist2d(hist, float(args.gaussian_sigma))
        histograms.append(hist)

    vmax = max((float(np.max(h)) for h in histograms if h.size > 0), default=1.0)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-12))

    fig, axes = plt.subplots(1, len(files), figsize=(6.5 * len(files), 5.6), constrained_layout=True, sharex=True, sharey=True)
    if len(files) == 1:
        axes = [axes]

    meshes = []
    for ax, label, hist in zip(axes, labels, histograms):
        mesh = ax.pcolormesh(
            xbins,
            ybins,
            hist.T,
            shading="auto",
            norm=norm,
            cmap="viridis",
        )
        ax.set_title(label)
        ax.set_xlabel("Shower depth [mm]")
        ax.set_ylabel("Time [ns]")
        meshes.append(mesh)

    cbar = fig.colorbar(meshes[0], ax=axes, pad=0.02)
    cbar.set_label("Normalized density")
    fig.suptitle(r"Depth-time density after event-wise time shift: $\gamma$ vs $\pi^0$")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
