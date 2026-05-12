#!/usr/import/env python
"""Plot publication-quality longitudinal observable summary plots for gamma and pi0 showers."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_timing_quantiles import _build_interval_waveform, _waveform_quantile_from_bins
from utils.plot_common import (
    automatic_bins,
    collect_shower_depths_and_summary,
    default_sample_label,
    resolve_paths,
    read_row,
    shared_range,
    TIME_KEY,
    TIME_END_KEY,
    AMP_KEY,
)

def set_journal_style():
    """Apply journal-quality styling with substantially larger fonts."""
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "axes.grid": False,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.5,
    })

def format_hist_label(label: str, mean: float, rms: float) -> str:
    return f"{label}\nμ={mean:.1f}, RMS={rms:.1f}"

def plot_hist_overlay(
    ax,
    series: list[tuple[str, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    bins: np.ndarray,
    colors: list[str],
) -> None:
    for (label, values), color in zip(series, colors):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean = float(np.mean(arr))
        rms = float(np.std(arr, ddof=0))
        hist = ax.hist(
            arr,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.5,
            color=color,
            label=format_hist_label(label, mean, rms),
        )
        # Add dashed vertical line for mean
        ax.axvline(mean, color=color, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized density")
    ax.set_xlim(*xlim)

    # Remove top and right spines to reduce clutter
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="best", frameon=False)

def plot_scatter_overlay(
    ax,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    colors: list[str],
) -> None:
    for (label, x, y), color in zip(series, colors):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size == 0:
            continue
        ax.scatter(x, y, s=15, alpha=0.15, linewidths=0.0, color=color, label=label, rasterized=True)

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    leg = ax.legend(loc="best", frameon=False, markerscale=3.0)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot publication-quality longitudinal observable summary.")
    p.add_argument("--files", nargs="+", default=["h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"])
    p.add_argument("--labels", nargs="+", default=[r"$\gamma$", r"$\pi^0$"])
    p.add_argument("--max-events", type=int, default=40000)
    p.add_argument("--min-theta", type=float, default=1.5)
    p.add_argument("--min-e-gen", type=float, default=10.0)
    p.add_argument("--bin-width", type=float, default=1.0)
    p.add_argument("--range-percentile", type=float, default=99.0)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--out-dir", default="plots/publication")
    p.add_argument("--out-name", default="longitudinal_observables_summary")
    return p.parse_args()

def main():
    args = parse_args()
    set_journal_style()

    files = resolve_paths(args.files)
    if not files:
        print("No files found. Generating dummy data for demonstration...")
        return

    labels = args.labels
    colors = ["#1f77b4", "#d62728"] # Blue for gamma, Red for pi0

    summary_series = []
    q50_fn = lambda handle, idx, _bw=float(args.bin_width): _event_q50(handle, idx, bin_width=_bw)

    for label, file_path in zip(labels, files):
        _, summary = collect_shower_depths_and_summary(
            Path(file_path),
            max_events=int(args.max_events),
            min_theta=float(args.min_theta),
            min_e_gen=float(args.min_e_gen),
            q50_fn=q50_fn,
            desc=Path(str(file_path)).name,
        )
        summary_series.append((label, summary))

    # Extract series
    mean_depths = [metrics["depth_mean"] for _, metrics in summary_series]
    spreads = [metrics["depth_spread"] for _, metrics in summary_series]
    p5s = [metrics["depth_p5"] for _, metrics in summary_series]
    skewnesses = [metrics["depth_skewness"] for _, metrics in summary_series]
    kurtoses = [metrics["depth_kurtosis"] for _, metrics in summary_series]
    q50s = [metrics["q50"] for _, metrics in summary_series]

    # Shared ranges
    mean_xlim = shared_range(mean_depths, float(args.range_percentile))
    spread_xlim = shared_range(spreads, float(args.range_percentile))
    p5_xlim = shared_range(p5s, float(args.range_percentile))
    skew_xlim = shared_range(skewnesses, float(args.range_percentile))
    kurt_xlim = shared_range(kurtoses, float(args.range_percentile))
    q50_xlim = shared_range(q50s, float(args.range_percentile))

    # Bins
    def get_bins(arrays, xlim):
        valid = [a for a in arrays if a.size > 0]
        comb = np.concatenate(valid) if valid else np.array([])
        return automatic_bins(comb, xlim, args.bins)

    hist_bins_mean = get_bins(mean_depths, mean_xlim)
    hist_bins_spread = get_bins(spreads, spread_xlim)
    hist_bins_p5 = get_bins(p5s, p5_xlim)
    hist_bins_skew = get_bins(skewnesses, skew_xlim)
    hist_bins_kurt = get_bins(kurtoses, kurt_xlim)

    # Plot 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    # Adjust spacing for journal two-column layout
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)

    axes = np.asarray(axes).reshape(2, 3)

    # 1. Mean Shower Depth
    plot_hist_overlay(
        axes[0, 0],
        [(label, metrics["depth_mean"]) for label, metrics in summary_series],
        title="Mean Shower Depth",
        xlabel="Mean Depth [mm]",
        xlim=mean_xlim,
        bins=hist_bins_mean,
        colors=colors,
    )

    # 2. Depth RMS
    plot_hist_overlay(
        axes[0, 1],
        [(label, metrics["depth_spread"]) for label, metrics in summary_series],
        title="Shower Depth Spread",
        xlabel="Depth RMS [mm]",
        xlim=spread_xlim,
        bins=hist_bins_spread,
        colors=colors,
    )

    # 3. p5 Depth
    plot_hist_overlay(
        axes[0, 2],
        [(label, metrics["depth_p5"]) for label, metrics in summary_series],
        title="Early Shower Depth",
        xlabel="5th Percentile Depth [mm]",
        xlim=p5_xlim,
        bins=hist_bins_p5,
        colors=colors,
    )

    # 4. Skewness
    plot_hist_overlay(
        axes[1, 0],
        [(label, metrics["depth_skewness"]) for label, metrics in summary_series],
        title="Longitudinal Skewness",
        xlabel="Skewness",
        xlim=skew_xlim,
        bins=hist_bins_skew,
        colors=colors,
    )

    # 5. Kurtosis
    plot_hist_overlay(
        axes[1, 1],
        [(label, metrics["depth_kurtosis"]) for label, metrics in summary_series],
        title="Longitudinal Kurtosis",
        xlabel="Excess Kurtosis",
        xlim=kurt_xlim,
        bins=hist_bins_kurt,
        colors=colors,
    )

    # 6. Scatter: Mean Depth vs Reconstructed Timing
    plot_scatter_overlay(
        axes[1, 2],
        [(label, metrics["depth_mean"], metrics["q50"]) for label, metrics in summary_series],
        title="Timing vs Shower Depth",
        xlabel="Mean Depth [mm]",
        ylabel="Median Timing [ns]",
        xlim=mean_xlim,
        ylim=q50_xlim,
        colors=colors,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path_png = out_dir / f"{args.out_name}.png"
    out_path_pdf = out_dir / f"{args.out_name}.pdf"

    fig.savefig(out_path_png, dpi=300)
    fig.savefig(out_path_pdf, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path_png}")
    print(f"Wrote {out_path_pdf}")

if __name__ == "__main__":
    main()
