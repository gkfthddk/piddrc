#!/usr/bin/env python
"""Plot normalized shower-depth distributions for gamma and pi0 samples."""

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
    read_row,
    resolve_paths,
    plot_hist_overlay,
    plot_scatter_overlay,
    shared_range,
    TIME_KEY,
    TIME_END_KEY,
    AMP_KEY,
)
from utils.plot_helpers import set_publication_style

set_publication_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot shower-depth distributions for gamma and pi0.")
    p.add_argument(
        "--files",
        nargs="+",
        default=["h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"],
        help="Input HDF5 files. Default compares e-, gamma, and pi0 samples.",
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
        "--bin-width",
        type=float,
        default=1.0,
        help="Time bin width used for waveform-based q50 extraction in the summary figure.",
    )
    p.add_argument(
        "--range-percentile",
        type=float,
        default=99.0,
        help="Central percentile span used to define the shared histogram range.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Fallback number of histogram bins if automatic binning is degenerate.",
    )
    p.add_argument(
        "--out-dir",
        default="plots/shower_depth_distribution",
        help="Output directory.",
    )
    p.add_argument(
        "--out-name",
        default="gamma_vs_pi0_shower_depth.png",
        help="Output filename.",
    )
    p.add_argument(
        "--summary-out-name",
        default="e_gamma_pi0_longitudinal_summary.png",
        help="Output filename for the combined longitudinal summary figure.",
    )
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


def main() -> None:
    args = parse_args()
    files = resolve_paths(args.files)
    if not files:
        raise FileNotFoundError("No input files found.")
    if args.labels is not None and len(args.labels) != len(files):
        raise ValueError("--labels must match --files count.")

    if args.labels is not None:
        labels = list(args.labels)
    else:
        labels = [default_sample_label(Path(str(p))) for p in files]

    summary_series: list[tuple[str, dict[str, np.ndarray]]] = []
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

    all_depths = [metrics["depth_mean"] for _, metrics in summary_series if metrics["depth_mean"].size > 0]
    xlim = shared_range(all_depths, float(args.range_percentile))
    if all_depths:
        combined = np.concatenate(all_depths)
    else:
        combined = np.array([], dtype=np.float64)
    bins = automatic_bins(combined, xlim, int(args.bins))

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    plot_hist_overlay(
        ax,
        [(label, metrics["depth_mean"]) for label, metrics in summary_series],
        title=r"Shower-depth distribution: $e^-$ / $\gamma$ / $\pi^0$",
        xlabel="Shower depth (mm)",
        xlim=xlim,
        bins=bins,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")

    mean_depths = [metrics["depth_mean"] for _, metrics in summary_series]
    spreads = [metrics["depth_spread"] for _, metrics in summary_series]
    p5s = [metrics["depth_p5"] for _, metrics in summary_series]
    skewnesses = [metrics["depth_skewness"] for _, metrics in summary_series]
    kurtoses = [metrics["depth_kurtosis"] for _, metrics in summary_series]
    tr_means = [metrics["tr_mean"] for _, metrics in summary_series]
    tr_spreads = [metrics["tr_spread"] for _, metrics in summary_series]
    tr_p5s = [metrics["tr_p5"] for _, metrics in summary_series]
    tr_skewnesses = [metrics["tr_skewness"] for _, metrics in summary_series]
    tr_kurtoses = [metrics["tr_kurtosis"] for _, metrics in summary_series]
    q50s = [metrics["q50"] for _, metrics in summary_series]

    mean_xlim = shared_range(mean_depths, float(args.range_percentile))
    spread_xlim = shared_range(spreads, float(args.range_percentile))
    p5_xlim = shared_range(p5s, float(args.range_percentile))
    skew_xlim = shared_range(skewnesses, float(args.range_percentile))
    kurt_xlim = shared_range(kurtoses, float(args.range_percentile))
    tr_mean_xlim = shared_range(tr_means, float(args.range_percentile))
    tr_spread_xlim = shared_range(tr_spreads, float(args.range_percentile))
    tr_p5_xlim = shared_range(tr_p5s, float(args.range_percentile))
    tr_skew_xlim = shared_range(tr_skewnesses, float(args.range_percentile))
    tr_kurt_xlim = shared_range(tr_kurtoses, float(args.range_percentile))
    q50_xlim = shared_range(q50s, float(args.range_percentile))

    hist_bins_mean = automatic_bins(
        np.concatenate([a for a in mean_depths if a.size > 0]) if any(a.size > 0 for a in mean_depths) else np.array([]),
        mean_xlim,
        int(args.bins),
    )
    hist_bins_spread = automatic_bins(
        np.concatenate([a for a in spreads if a.size > 0]) if any(a.size > 0 for a in spreads) else np.array([]),
        spread_xlim,
        int(args.bins),
    )
    hist_bins_p5 = automatic_bins(
        np.concatenate([a for a in p5s if a.size > 0]) if any(a.size > 0 for a in p5s) else np.array([]),
        p5_xlim,
        int(args.bins),
    )
    hist_bins_skew = automatic_bins(
        np.concatenate([a for a in skewnesses if a.size > 0]) if any(a.size > 0 for a in skewnesses) else np.array([]),
        skew_xlim,
        int(args.bins),
    )
    hist_bins_kurt = automatic_bins(
        np.concatenate([a for a in kurtoses if a.size > 0]) if any(a.size > 0 for a in kurtoses) else np.array([]),
        kurt_xlim,
        int(args.bins),
    )
    hist_bins_tr_mean = automatic_bins(
        np.concatenate([a for a in tr_means if a.size > 0]) if any(a.size > 0 for a in tr_means) else np.array([]),
        tr_mean_xlim,
        int(args.bins),
    )
    hist_bins_tr_spread = automatic_bins(
        np.concatenate([a for a in tr_spreads if a.size > 0]) if any(a.size > 0 for a in tr_spreads) else np.array([]),
        tr_spread_xlim,
        int(args.bins),
    )
    hist_bins_tr_p5 = automatic_bins(
        np.concatenate([a for a in tr_p5s if a.size > 0]) if any(a.size > 0 for a in tr_p5s) else np.array([]),
        tr_p5_xlim,
        int(args.bins),
    )
    hist_bins_tr_skew = automatic_bins(
        np.concatenate([a for a in tr_skewnesses if a.size > 0]) if any(a.size > 0 for a in tr_skewnesses) else np.array([]),
        tr_skew_xlim,
        int(args.bins),
    )
    hist_bins_tr_kurt = automatic_bins(
        np.concatenate([a for a in tr_kurtoses if a.size > 0]) if any(a.size > 0 for a in tr_kurtoses) else np.array([]),
        tr_kurt_xlim,
        int(args.bins),
    )

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    axes = np.asarray(axes).reshape(2, 3)
    plot_hist_overlay(
        axes[0, 0],
        [(label, metrics["depth_mean"]) for label, metrics in summary_series],
        title="Event-level longitudinal profile",
        xlabel="Shower depth (mm)",
        xlim=mean_xlim,
        bins=hist_bins_mean,
    )
    plot_hist_overlay(
        axes[0, 1],
        [(label, metrics["depth_spread"]) for label, metrics in summary_series],
        title="Depth spread per event",
        xlabel="Weighted depth RMS (mm)",
        xlim=spread_xlim,
        bins=hist_bins_spread,
    )
    plot_hist_overlay(
        axes[0, 2],
        [(label, metrics["depth_p5"]) for label, metrics in summary_series],
        title="p5 depth distribution",
        xlabel="p5 depth (mm)",
        xlim=p5_xlim,
        bins=hist_bins_p5,
    )
    plot_scatter_overlay(
        axes[1, 0],
        [(label, metrics["depth_mean"], metrics["q50"]) for label, metrics in summary_series],
        title="Depth vs median timing",
        xlabel="Shower depth (mm)",
        ylabel="Median timing (ns)",
        xlim=mean_xlim,
        ylim=q50_xlim,
    )
    plot_hist_overlay(
        axes[1, 1],
        [(label, metrics["depth_skewness"]) for label, metrics in summary_series],
        title="Skewness of depth",
        xlabel="Weighted skewness",
        xlim=skew_xlim,
        bins=hist_bins_skew,
    )
    plot_hist_overlay(
        axes[1, 2],
        [(label, metrics["depth_kurtosis"]) for label, metrics in summary_series],
        title="Kurtosis of depth",
        xlabel="Weighted excess kurtosis",
        xlim=kurt_xlim,
        bins=hist_bins_kurt,
    )
    fig.suptitle(r"$e^-$ / $\gamma$ / $\pi^0$ longitudinal summary", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    summary_out = out_dir / args.summary_out_name
    fig.savefig(summary_out, dpi=300)
    plt.close(fig)
    print(f"Wrote {summary_out}")

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    axes = np.asarray(axes).reshape(2, 3)
    plot_hist_overlay(
        axes[0, 0],
        [(label, metrics["tr_mean"]) for label, metrics in summary_series],
        title="Transverse mean radius",
        xlabel="Mean transverse radius (mm)",
        xlim=tr_mean_xlim,
        bins=hist_bins_tr_mean,
    )
    plot_hist_overlay(
        axes[0, 1],
        [(label, metrics["tr_spread"]) for label, metrics in summary_series],
        title="Transverse spread",
        xlabel="Weighted transverse RMS (mm)",
        xlim=tr_spread_xlim,
        bins=hist_bins_tr_spread,
    )
    plot_hist_overlay(
        axes[0, 2],
        [(label, metrics["tr_p5"]) for label, metrics in summary_series],
        title="Transverse p5 radius",
        xlabel="p5 transverse radius (mm)",
        xlim=tr_p5_xlim,
        bins=hist_bins_tr_p5,
    )
    plot_hist_overlay(
        axes[1, 0],
        [(label, metrics["tr_skewness"]) for label, metrics in summary_series],
        title="Skewness of transverse shape",
        xlabel="Weighted skewness",
        xlim=tr_skew_xlim,
        bins=hist_bins_tr_skew,
    )
    plot_hist_overlay(
        axes[1, 1],
        [(label, metrics["tr_kurtosis"]) for label, metrics in summary_series],
        title="Kurtosis of transverse shape",
        xlabel="Weighted excess kurtosis",
        xlim=tr_kurt_xlim,
        bins=hist_bins_tr_kurt,
    )
    plot_scatter_overlay(
        axes[1, 2],
        [(label, metrics["depth_mean"], metrics["tr_mean"]) for label, metrics in summary_series],
        title="Transverse vs longitudinal mean depth",
        xlabel="Longitudinal mean depth (mm)",
        ylabel="Transverse mean radius (mm)",
        xlim=mean_xlim,
        ylim=tr_mean_xlim,
    )
    fig.suptitle(r"$e^-$ / $\gamma$ / $\pi^0$ transverse summary", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    transverse_out = out_dir / "e_gamma_pi0_transverse_summary.png"
    fig.savefig(transverse_out, dpi=300)
    plt.close(fig)
    print(f"Wrote {transverse_out}")


if __name__ == "__main__":
    main()
