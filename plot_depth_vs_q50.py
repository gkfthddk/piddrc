#!/usr/bin/env python
"""Plot longitudinal shower depth versus reconstructed timing q50.

This figure is intended as a compact, model-independent check that the
reconstructed timing observable tracks longitudinal shower development.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from plot_timing_quantiles import _build_interval_waveform, _waveform_quantile_from_bins
from utils.plot_common import (
    default_sample_label,
    event_passes_selection,
    compute_shower_depth_mm,
    TIME_KEY,
    TIME_END_KEY,
    AMP_KEY,
    read_row,
    resolve_paths,
)
from utils.plot_helpers import set_publication_style

set_publication_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Sim3d shower depth versus reconstructed timing q50.")
    p.add_argument(
        "--files",
        nargs="+",
        default=[
            "h5s/e-_1-120GeV.h5py",
            "h5s/pi+_1-120GeV.h5py",
            "h5s/gamma_1-120GeV.h5py",
            "h5s/pi0_1-120GeV.h5py",
        ],
        help="Input HDF5 files. Default compares e-, pi+, gamma, and pi0 samples.",
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
        "--bin-width",
        type=float,
        default=1.0,
        help="Time bin width used for waveform-based q50 extraction.",
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
        "--out-dir",
        default="plots/depth_vs_q50",
        help="Output directory.",
    )
    p.add_argument(
        "--out-name",
        default="depth_vs_q50",
        help="Output basename. A .png/.pdf suffix is accepted and stripped.",
    )
    p.add_argument(
        "--hexbin-gridsize",
        type=int,
        default=30,
        help="Hexbin grid size for the 2D density view.",
    )
    p.add_argument(
        "--axis-percentile",
        type=float,
        default=99.0,
        help="Central percentile span used to set shared x/y limits across panels.",
    )
    p.add_argument("--summary-json", action="store_true", help="Also write numeric summary JSON.")
    return p.parse_args()


def _event_q50(handle: h5py.File, idx: int, *, bin_width: float) -> float:
    if TIME_KEY not in handle or TIME_END_KEY not in handle or AMP_KEY not in handle:
        return float("nan")
    starts = read_row(handle, TIME_KEY, idx)
    ends = read_row(handle, TIME_END_KEY, idx)
    amps = read_row(handle, AMP_KEY, idx)
    n = min(starts.size, ends.size, amps.size)
    if n == 0:
        return float("nan")
    starts = starts[:n]
    ends = ends[:n]
    amps = amps[:n]
    valid = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(amps) & (amps > 0.0)
    if not np.any(valid):
        return float("nan")
    edges, binned = _build_interval_waveform(starts[valid], ends[valid], amps[valid], bin_width=bin_width)
    if edges.size < 2 or binned.size == 0 or not np.any(np.isfinite(binned) & (binned > 0.0)):
        return float("nan")
    return float(_waveform_quantile_from_bins(edges, binned, 0.5))


def _collect_series(
    path: Path,
    *,
    max_events: int,
    bin_width: float,
    min_theta: float,
    min_e_gen: float,
) -> tuple[np.ndarray, np.ndarray]:
    depths: list[float] = []
    q50s: list[float] = []
    with h5py.File(path, "r") as handle:
        if TIME_KEY not in handle:
            raise KeyError(f"{TIME_KEY} missing in {path}")
        n_events = int(handle[TIME_KEY].shape[0])
        take = n_events if max_events <= 0 else min(n_events, max_events)
        for idx in tqdm(range(take), desc=path.name, unit="evt", leave=False):
            if not event_passes_selection(handle, idx, min_theta=min_theta, min_e_gen=min_e_gen):
                continue
            depth = compute_shower_depth_mm(handle, idx)
            q50 = _event_q50(handle, idx, bin_width=bin_width)
            if not (np.isfinite(depth) and np.isfinite(q50)):
                continue
            depths.append(float(depth))
            q50s.append(float(q50))
    return np.asarray(depths, dtype=np.float64), np.asarray(q50s, dtype=np.float64)


def _plot_panel(
    ax: plt.Axes,
    depth: np.ndarray,
    q50: np.ndarray,
    *,
    label: str,
    gridsize: int,
    extent: tuple[float, float, float, float] | None,
) -> None:
    hb = ax.hexbin(depth, q50, gridsize=gridsize, cmap="viridis", mincnt=1, bins="log", extent=extent)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Events / bin")

    ax.set_title(label)
    ax.set_xlabel("Longitudinal shower depth [mm]")
    ax.set_ylabel(r"Reconstructed timing $q_{50}$ [ns]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()
    cbar.ax.tick_params(direction="in")


def _shared_limits(
    series: Sequence[tuple[str, np.ndarray, np.ndarray]],
    percentile: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    depths = [depth for _, depth, _ in series if depth.size]
    q50s = [q50 for _, _, q50 in series if q50.size]
    if not depths or not q50s:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))

    all_depth = np.concatenate(depths)
    all_q50 = np.concatenate(q50s)
    pct = float(np.clip(percentile, 0.0, 100.0))
    lo = 0.5 * (100.0 - pct)
    hi = 100.0 - lo
    xlim = tuple(np.quantile(all_depth, [lo / 100.0, hi / 100.0]).tolist())
    ylim = tuple(np.quantile(all_q50, [lo / 100.0, hi / 100.0]).tolist())
    return xlim, ylim


def _sample_summary(label: str, depth: np.ndarray, q50: np.ndarray) -> dict[str, float | int | str | None]:
    valid = np.isfinite(depth) & np.isfinite(q50)
    depth = depth[valid]
    q50 = q50[valid]
    if depth.size == 0:
        return {
            "label": label,
            "n_events": 0,
            "depth_mean_mm": None,
            "depth_rms_mm": None,
            "q50_mean_ns": None,
            "q50_rms_ns": None,
            "depth_q50_correlation": None,
        }
    corr = None
    if depth.size >= 2 and np.std(depth) > 0.0 and np.std(q50) > 0.0:
        corr = float(np.corrcoef(depth, q50)[0, 1])
    return {
        "label": label,
        "n_events": int(depth.size),
        "depth_mean_mm": float(np.mean(depth)),
        "depth_rms_mm": float(np.std(depth, ddof=0)),
        "q50_mean_ns": float(np.mean(q50)),
        "q50_rms_ns": float(np.std(q50, ddof=0)),
        "depth_q50_correlation": corr,
    }


def _output_stem(out_name: str) -> str:
    path = Path(out_name)
    if path.suffix.lower() in {".png", ".pdf", ".json"}:
        return path.with_suffix("").name
    return path.name


def main() -> None:
    args = parse_args()
    set_publication_style()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    files = resolve_paths(args.files)
    if not files:
        raise FileNotFoundError("No input files found.")
    if args.labels is not None and len(args.labels) != len(files):
        raise ValueError("--labels must match --files count.")

    if args.labels is not None:
        labels = list(args.labels)
    else:
        labels = [default_sample_label(Path(str(p))) for p in files]

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, file_path in zip(labels, files):
        depth, q50 = _collect_series(
            Path(file_path),
            max_events=int(args.max_events),
            bin_width=float(args.bin_width),
            min_theta=float(args.min_theta),
            min_e_gen=float(args.min_e_gen),
        )
        series.append((label, depth, q50))

    ncols = 2 if len(series) > 1 else 1
    nrows = int(math.ceil(len(series) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.8 * nrows), constrained_layout=True)
    if len(series) == 1:
        axes = np.asarray([axes])
    else:
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(nrows, ncols)
        if len(series) < nrows * ncols:
            for ax in np.ravel(axes)[len(series):]:
                ax.set_visible(False)

    xlim, ylim = _shared_limits(series, float(args.axis_percentile))
    extent = None
    if np.all(np.isfinite(xlim)) and np.all(np.isfinite(ylim)):
        extent = (float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
        for ax in np.ravel(axes):
            if ax.get_visible():
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

    for ax, (label, depth, q50) in zip(np.ravel(axes), series):
        _plot_panel(ax, depth, q50, label=label, gridsize=int(args.hexbin_gridsize), extent=extent)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = _output_stem(str(args.out_name))
    pdf_path = out_dir / f"{out_stem}.pdf"
    png_path = out_dir / f"{out_stem}.png"
    json_path = out_dir / f"{out_stem}.json"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    summary = {
        "samples": {label: _sample_summary(label, depth, q50) for label, depth, q50 in series},
        "configuration": {
            "files": [str(path) for path in files],
            "max_events": int(args.max_events),
            "bin_width_ns": float(args.bin_width),
            "min_theta_rad": float(args.min_theta),
            "min_e_gen_GeV": float(args.min_e_gen),
            "axis_percentile": float(args.axis_percentile),
            "hexbin_gridsize": int(args.hexbin_gridsize),
        },
        "outputs": {"pdf": str(pdf_path), "png": str(png_path)},
    }
    if args.summary_json:
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {json_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
