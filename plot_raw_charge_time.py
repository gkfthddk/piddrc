#!/usr/bin/env python
"""Plot RawCalorimeterHits charge distributions as a function of time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RawCalorimeterHits amplitude vs time.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/hdfs/ml/dualreadout/v001/kaon+_10GeV"),
        help="Directory containing kaon+ digi ROOT files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="digi_kaon+_10GeV_*.root",
        help="Glob pattern for primary sample files.",
    )
    parser.add_argument("--label", type=str, default="kaon+", help="Primary sample label.")
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=Path("/hdfs/ml/dualreadout/v001/pi+_10GeV"),
        help="Comparison sample directory (default: pi+).",
    )
    parser.add_argument(
        "--compare-pattern",
        type=str,
        default="digi_pi+_10GeV_*.root",
        help="Glob pattern for comparison sample files.",
    )
    parser.add_argument("--compare-label", type=str, default="pi+", help="Comparison sample label.")
    parser.add_argument("--no-compare", action="store_true", help="Disable comparison sample.")
    parser.add_argument("--tree", type=str, default="events", help="TTree name in ROOT files.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files per sample (<=0 means all).")
    parser.add_argument(
        "--shuffle-files",
        action="store_true",
        help="Shuffle file order each run before applying max-file/event caps.",
    )
    parser.add_argument(
        "--no-shuffle-files",
        dest="shuffle_files",
        action="store_false",
        help="Disable file-order shuffling.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible shuffling (used only with --shuffle-files).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=10000,
        help="Global cap on processed events per sample (<=0 means all).",
    )
    parser.add_argument(
        "--range-max-files",
        type=int,
        default=100,
        help="Phase-1 range estimation: max files to sample (<=0 means all candidate files).",
    )
    parser.add_argument(
        "--range-max-events",
        type=int,
        default=2000,
        help="Phase-1 range estimation: max events to sample (<=0 means use --max-events).",
    )
    parser.add_argument("--time-bins", type=int, default=160, help="Histogram bins for time axis.")
    parser.add_argument("--charge-bins", type=int, default=160, help="Histogram bins for charge axis.")
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Shared matplotlib colormap for 2D density plots.",
    )
    parser.add_argument(
        "--charge-quantile-low",
        type=float,
        default=0.0005,
        help="Lower quantile used to set the charge axis before widening.",
    )
    parser.add_argument(
        "--charge-quantile-high",
        type=float,
        default=0.9999,
        help="Upper quantile used to set the charge axis before widening.",
    )
    parser.add_argument(
        "--charge-range-scale",
        type=float,
        default=1.15,
        help="Multiply charge span by this factor to widen the plotted charge range.",
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=300000,
        help="Reservoir sample size used to estimate robust axis ranges.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/raw_charge_time"),
        help="Output directory.",
    )
    parser.set_defaults(shuffle_files=True)
    return parser.parse_args()


def _iter_files(
    input_dir: Path,
    pattern: str,
    max_files: int,
    *,
    rng: np.random.Generator | None = None,
) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    if rng is not None and files:
        order = rng.permutation(len(files))
        files = [files[i] for i in order]
    if max_files > 0:
        files = files[:max_files]
    return files


def _reservoir_update(
    rng: np.random.Generator,
    reservoir: List[float],
    values: np.ndarray,
    seen: int,
    max_size: int,
) -> int:
    if max_size <= 0:
        return seen
    for value in values:
        seen += 1
        if len(reservoir) < max_size:
            reservoir.append(float(value))
        else:
            j = int(rng.integers(0, seen))
            if j < max_size:
                reservoir[j] = float(value)
    return seen


def _phase1_ranges(
    *,
    sample_label: str,
    files: List[Path],
    tree_name: str,
    max_events: int,
    reservoir_size: int,
    amp_branch: str,
    time_branch: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(12345)
    amp_res: List[float] = []
    time_res: List[float] = []
    seen_amp = 0
    seen_time = 0
    events_processed = 0

    file_iter = tqdm(files, desc=f"Phase1 files [{sample_label}]", unit="file")
    evt_target = max_events if max_events > 0 else None
    evt_bar = (
        tqdm(total=evt_target, desc=f"Phase1 events [{sample_label}]", unit="evt")
        if evt_target is not None
        else None
    )
    for path in file_iter:
        if max_events > 0 and events_processed >= max_events:
            break
        with uproot.open(path) as f:
            if tree_name not in f:
                continue
            tree = f[tree_name]
            arr = tree.arrays([amp_branch, time_branch], library="ak")
        n_events = len(arr[amp_branch])
        if max_events > 0:
            remaining = max_events - events_processed
            if remaining <= 0:
                break
            if n_events > remaining:
                arr = arr[:remaining]
                n_events = remaining
        events_processed += n_events
        if evt_bar is not None:
            evt_bar.update(n_events)
        file_iter.set_postfix(events=events_processed)

        amp_flat = np.asarray(ak.flatten(arr[amp_branch], axis=None), dtype=np.float64)
        time_flat = np.asarray(ak.flatten(arr[time_branch], axis=None), dtype=np.float64)
        valid = np.isfinite(amp_flat) & np.isfinite(time_flat) & (amp_flat > 0.0)
        if not np.any(valid):
            continue
        amp_vals = amp_flat[valid]
        time_vals = time_flat[valid]
        seen_amp = _reservoir_update(rng, amp_res, amp_vals, seen_amp, reservoir_size)
        seen_time = _reservoir_update(rng, time_res, time_vals, seen_time, reservoir_size)
    file_iter.close()
    if evt_bar is not None:
        evt_bar.close()

    if not amp_res or not time_res:
        raise RuntimeError(f"No valid RawCalorimeterHits points found for {sample_label}.")
    return np.asarray(time_res, dtype=np.float64), np.asarray(amp_res, dtype=np.float64), events_processed


def _phase2_histograms(
    *,
    sample_label: str,
    files: List[Path],
    tree_name: str,
    max_events: int,
    amp_branch: str,
    time_branch: str,
    time_edges: np.ndarray,
    charge_edges: np.ndarray,
) -> Dict[str, np.ndarray | int | float]:
    hist2d = np.zeros((len(time_edges) - 1, len(charge_edges) - 1), dtype=np.float64)
    time_charge_sum = np.zeros(len(time_edges) - 1, dtype=np.float64)
    time_hit_count = np.zeros(len(time_edges) - 1, dtype=np.float64)
    events_processed = 0
    hit_count = 0
    charge_sum = 0.0

    file_iter = tqdm(files, desc=f"Phase2 files [{sample_label}]", unit="file")
    evt_target = max_events if max_events > 0 else None
    evt_bar = (
        tqdm(total=evt_target, desc=f"Phase2 events [{sample_label}]", unit="evt")
        if evt_target is not None
        else None
    )

    for path in file_iter:
        if max_events > 0 and events_processed >= max_events:
            break
        with uproot.open(path) as f:
            if tree_name not in f:
                continue
            tree = f[tree_name]
            arr = tree.arrays([amp_branch, time_branch], library="ak")
        n_events = len(arr[amp_branch])
        if max_events > 0:
            remaining = max_events - events_processed
            if remaining <= 0:
                break
            if n_events > remaining:
                arr = arr[:remaining]
                n_events = remaining
        events_processed += n_events
        if evt_bar is not None:
            evt_bar.update(n_events)
        file_iter.set_postfix(events=events_processed)

        amp_flat = np.asarray(ak.flatten(arr[amp_branch], axis=None), dtype=np.float64)
        time_flat = np.asarray(ak.flatten(arr[time_branch], axis=None), dtype=np.float64)
        valid = np.isfinite(amp_flat) & np.isfinite(time_flat) & (amp_flat > 0.0)
        if not np.any(valid):
            continue
        amp_vals = amp_flat[valid]
        time_vals = time_flat[valid]
        hit_count += int(amp_vals.size)
        charge_sum += float(np.sum(amp_vals))

        h2, _, _ = np.histogram2d(time_vals, amp_vals, bins=[time_edges, charge_edges])
        hist2d += h2
        time_charge_sum += np.histogram(time_vals, bins=time_edges, weights=amp_vals)[0]
        time_hit_count += np.histogram(time_vals, bins=time_edges)[0]

    file_iter.close()
    if evt_bar is not None:
        evt_bar.close()

    with np.errstate(divide="ignore", invalid="ignore"):
        charge_density_by_time = np.divide(
            time_charge_sum,
            np.sum(time_charge_sum),
            out=np.zeros_like(time_charge_sum),
            where=np.sum(time_charge_sum) > 0,
        )
        mean_charge_by_time = np.divide(
            time_charge_sum,
            time_hit_count,
            out=np.zeros_like(time_charge_sum),
            where=time_hit_count > 0,
        )

    return {
        "hist2d": hist2d,
        "time_charge_sum": time_charge_sum,
        "time_hit_count": time_hit_count,
        "charge_density_by_time": charge_density_by_time,
        "mean_charge_by_time": mean_charge_by_time,
        "events_processed": events_processed,
        "hits_processed": hit_count,
        "charge_sum": charge_sum,
    }


def _analyze_sample(
    *,
    sample_label: str,
    files: List[Path],
    tree_name: str,
    max_events: int,
    time_edges: np.ndarray,
    charge_edges: np.ndarray,
    amp_branch: str,
    time_branch: str,
) -> Dict[str, np.ndarray | int | float]:
    phase2 = _phase2_histograms(
        sample_label=sample_label,
        files=files,
        tree_name=tree_name,
        max_events=max_events,
        amp_branch=amp_branch,
        time_branch=time_branch,
        time_edges=time_edges,
        charge_edges=charge_edges,
    )
    return phase2


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    amp_branch = "RawCalorimeterHits/RawCalorimeterHits.amplitude"
    time_branch = "RawCalorimeterHits/RawCalorimeterHits.timeStamp"

    shuffle_rng = np.random.default_rng(args.shuffle_seed) if args.shuffle_files else None

    primary_files = _iter_files(args.input_dir, args.pattern, args.max_files, rng=shuffle_rng)
    if not primary_files:
        raise FileNotFoundError(f"No files found for primary sample: {args.input_dir / args.pattern}")

    compare_files: List[Path] = []
    if not args.no_compare:
        compare_files = _iter_files(args.compare_dir, args.compare_pattern, args.max_files, rng=shuffle_rng)
        if not compare_files:
            raise FileNotFoundError(f"No files found for comparison sample: {args.compare_dir / args.compare_pattern}")

    # Determine robust global axis ranges from a smaller phase-1 sample.
    range_max_events = args.range_max_events if args.range_max_events > 0 else args.max_events
    if args.range_max_files > 0:
        range_primary_files = primary_files[: args.range_max_files]
        range_compare_files = compare_files[: args.range_max_files]
    else:
        range_primary_files = primary_files
        range_compare_files = compare_files

    p_time_res, p_amp_res, _ = _phase1_ranges(
        sample_label=args.label,
        files=range_primary_files,
        tree_name=args.tree,
        max_events=range_max_events,
        reservoir_size=args.reservoir_size,
        amp_branch=amp_branch,
        time_branch=time_branch,
    )
    all_time_res = [p_time_res]
    all_amp_res = [p_amp_res]
    if compare_files:
        c_time_res, c_amp_res, _ = _phase1_ranges(
            sample_label=args.compare_label,
            files=range_compare_files,
            tree_name=args.tree,
            max_events=range_max_events,
            reservoir_size=args.reservoir_size,
            amp_branch=amp_branch,
            time_branch=time_branch,
        )
        all_time_res.append(c_time_res)
        all_amp_res.append(c_amp_res)

    time_res = np.concatenate(all_time_res)
    amp_res = np.concatenate(all_amp_res)
    time_lo, time_hi = np.quantile(time_res, [0.001, 0.999])
    if not (0.0 <= args.charge_quantile_low < args.charge_quantile_high <= 1.0):
        raise ValueError("charge quantiles must satisfy 0 <= low < high <= 1.")
    if args.charge_range_scale <= 0:
        raise ValueError("charge_range_scale must be positive.")

    amp_lo, amp_hi = np.quantile(amp_res, [args.charge_quantile_low, args.charge_quantile_high])
    if not np.isfinite(time_lo) or not np.isfinite(time_hi) or time_hi <= time_lo:
        raise RuntimeError("Invalid time range inferred from data.")
    if not np.isfinite(amp_lo) or not np.isfinite(amp_hi) or amp_hi <= amp_lo:
        raise RuntimeError("Invalid charge range inferred from data.")

    amp_mid = 0.5 * (amp_lo + amp_hi)
    amp_half = 0.5 * (amp_hi - amp_lo) * args.charge_range_scale
    amp_lo = max(0.0, amp_mid - amp_half)
    amp_hi = amp_mid + amp_half

    time_edges = np.linspace(float(time_lo), float(time_hi), args.time_bins + 1)
    charge_edges = np.linspace(float(amp_lo), float(amp_hi), args.charge_bins + 1)
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

    primary = _analyze_sample(
        sample_label=args.label,
        files=primary_files,
        tree_name=args.tree,
        max_events=args.max_events,
        time_edges=time_edges,
        charge_edges=charge_edges,
        amp_branch=amp_branch,
        time_branch=time_branch,
    )

    comparison = None
    if compare_files:
        comparison = _analyze_sample(
            sample_label=args.compare_label,
            files=compare_files,
            tree_name=args.tree,
            max_events=args.max_events,
            time_edges=time_edges,
            charge_edges=charge_edges,
            amp_branch=amp_branch,
            time_branch=time_branch,
        )

    summary: Dict[str, object] = {
        "time_range": [float(time_edges[0]), float(time_edges[-1])],
        "charge_range": [float(charge_edges[0]), float(charge_edges[-1])],
        "primary": {
            "label": args.label,
            "events_processed": int(primary["events_processed"]),
            "hits_processed": int(primary["hits_processed"]),
            "charge_sum": float(primary["charge_sum"]),
        },
    }
    if comparison is not None:
        summary["comparison"] = {
            "label": args.compare_label,
            "events_processed": int(comparison["events_processed"]),
            "hits_processed": int(comparison["hits_processed"]),
            "charge_sum": float(comparison["charge_sum"]),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 2D density (primary)
    fig, ax = plt.subplots(figsize=(8, 6))
    h2 = np.asarray(primary["hist2d"], dtype=np.float64).T
    pcm = ax.pcolormesh(time_edges, charge_edges, np.log10(h2 + 1.0), shading="auto", cmap=args.cmap)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("log10(count + 1)")
    ax.set_title(f"Raw Charge-Time Density ({args.label})")
    ax.set_xlabel("timeStamp")
    ax.set_ylabel("amplitude (charge)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.label.replace('+','plus')}_charge_time_density_2d.png", dpi=160)
    plt.close(fig)

    # 2D density (comparison)
    if comparison is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ch2 = np.asarray(comparison["hist2d"], dtype=np.float64).T
        pcm = ax.pcolormesh(time_edges, charge_edges, np.log10(ch2 + 1.0), shading="auto", cmap=args.cmap)
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("log10(count + 1)")
        ax.set_title(f"Raw Charge-Time Density ({args.compare_label})")
        ax.set_xlabel("timeStamp")
        ax.set_ylabel("amplitude (charge)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.compare_label.replace('+','plus')}_charge_time_density_2d.png", dpi=160)
        plt.close(fig)

    # Charge density by time (overlay)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_centers, np.asarray(primary["charge_density_by_time"]), label=f"{args.label} normalized charge density")
    if comparison is not None:
        ax.plot(time_centers, np.asarray(comparison["charge_density_by_time"]), label=f"{args.compare_label} normalized charge density")
    ax.set_title("Raw Charge Distribution Density by Time")
    ax.set_xlabel("timeStamp")
    ax.set_ylabel("normalized charge density")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "charge_density_by_time.png", dpi=160)
    plt.close(fig)

    # Mean charge by time (overlay)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_centers, np.asarray(primary["mean_charge_by_time"]), label=f"{args.label} mean charge")
    if comparison is not None:
        ax.plot(time_centers, np.asarray(comparison["mean_charge_by_time"]), label=f"{args.compare_label} mean charge")
    ax.set_title("Mean Raw Charge by Time")
    ax.set_xlabel("timeStamp")
    ax.set_ylabel("mean amplitude")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_charge_by_time.png", dpi=160)
    plt.close(fig)

    # Stacked (cumulative absolute) charge vs time (overlay)
    fig, ax = plt.subplots(figsize=(8, 5))
    primary_charge = np.asarray(primary["time_charge_sum"], dtype=np.float64)
    primary_cum = np.cumsum(primary_charge)
    ax.plot(time_centers, primary_cum, label=f"{args.label} cumulative charge")

    if comparison is not None:
        comp_charge = np.asarray(comparison["time_charge_sum"], dtype=np.float64)
        comp_cum = np.cumsum(comp_charge)
        ax.plot(time_centers, comp_cum, label=f"{args.compare_label} cumulative charge")

    ax.set_title("Stacked Absolute Charge vs Time")
    ax.set_xlabel("timeStamp")
    ax.set_ylabel("cumulative charge")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "stacked_charge_vs_time.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
