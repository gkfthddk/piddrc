"""Analyze readout validity and waveform split behavior at point cutoffs.

The script inspects point-cloud ordering and answers:
1) How many valid readouts/hits are kept at each cutoff (e.g. 1000, 2000)?
2) Whether the cutoff lands exactly on a waveform boundary or splits a waveform.
3) Which cellID is most frequently split at the cutoff.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze waveform boundary behavior at point cutoffs.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["h5s/*_1-100GeV_10.h5py"],
        help="Input .h5py files (paths or globs).",
    )
    parser.add_argument(
        "--mask-feature",
        default="DRcalo3dHits.amplitude_sum",
        help="Feature used to define valid hits (finite and non-zero).",
    )
    parser.add_argument(
        "--cellid-feature",
        default="DRcalo3dHits.cellID",
        help="CellID feature that defines readout channels/waveforms.",
    )
    parser.add_argument(
        "--amplitude-feature",
        default="DRcalo3dHits.amplitude_sum",
        help="Amplitude-like feature used for included/remnant fraction analysis.",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[1000, 2000],
        help="Point count cutoffs to evaluate.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=5000,
        help="Max events per file (<=0 means all).",
    )
    parser.add_argument(
        "--output",
        default="plots/cutoff_analysis.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--details-output",
        default="plots/cutoff_event_details.csv",
        help="Per-event CSV details path.",
    )
    return parser.parse_args()


def expand_paths(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        matches = sorted(Path(".").glob(pattern))
        if matches:
            out.extend(matches)
        else:
            out.append(Path(pattern))
    dedup: List[Path] = []
    seen = set()
    for path in out:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup


def valid_mask(handle: h5py.File, mask_feature: str, event_idx: int, row_shape: tuple[int, ...]) -> np.ndarray:
    if mask_feature in handle:
        base = np.asarray(handle[mask_feature][event_idx]).reshape(-1)
        if base.shape == row_shape:
            return np.isfinite(base) & (base != 0)
    return np.ones(row_shape, dtype=bool)


def run_length_count(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    return int(np.count_nonzero(values[1:] != values[:-1]) + 1)


def as_float(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def main() -> None:
    args = parse_args()
    files = [p for p in expand_paths(args.files) if p.exists()]
    if not files:
        raise SystemExit("No input files found.")

    cutoffs = sorted({c for c in args.cutoffs if c > 0})
    if not cutoffs:
        raise SystemExit("No positive cutoffs provided.")

    agg: Dict[int, Dict[str, float]] = {
        c: defaultdict(float) for c in cutoffs
    }
    split_cell_counter: Dict[int, Counter] = {c: Counter() for c in cutoffs}
    partial_cell_counter: Dict[int, Counter] = {c: Counter() for c in cutoffs}
    event_rows: List[str] = [
        "file,event_index,valid_points,cutoff,kept_points,unique_readouts,unique_readouts_in_kept,readout_runs,wholly_included_readouts,partial_readouts_in_kept,included_amplitude_sum,remnant_amplitude_sum,included_amplitude_fraction,remnant_amplitude_fraction,exact_boundary,split_cellid"
    ]

    file_iter = tqdm(files, desc="files")
    for path in file_iter:
        with h5py.File(path, "r") as handle:
            if args.cellid_feature not in handle:
                print(f"Skipping {path}: missing {args.cellid_feature}")
                continue
            if args.amplitude_feature not in handle:
                print(f"Skipping {path}: missing {args.amplitude_feature}")
                continue

            n_events = handle[args.cellid_feature].shape[0]
            if args.max_events > 0:
                n_events = min(n_events, args.max_events)

            event_iter = tqdm(range(n_events), desc=f"events {path.stem}", leave=False)
            for event_idx in event_iter:
                cells_raw = np.asarray(handle[args.cellid_feature][event_idx]).reshape(-1)
                amp_raw = np.asarray(handle[args.amplitude_feature][event_idx]).reshape(-1)
                if cells_raw.size == 0:
                    continue
                if amp_raw.shape != cells_raw.shape:
                    continue
                mask = valid_mask(handle, args.mask_feature, event_idx, cells_raw.shape)
                cells = cells_raw[mask]
                amps = amp_raw[mask]
                finite_pair = np.isfinite(cells) & np.isfinite(amps)
                cells = cells[finite_pair]
                amps = amps[finite_pair]
                cells = cells[np.isfinite(cells)].astype(np.int64, copy=False)
                amps = amps.astype(np.float64, copy=False)
                valid_points = int(cells.size)
                if valid_points == 0:
                    continue
                total_amp = float(np.sum(amps))
                full_unique, full_counts = np.unique(cells, return_counts=True)
                full_unique_count = int(full_unique.size)
                for cutoff in cutoffs:
                    agg[cutoff]["sum_unique_readouts_full"] += full_unique_count

                for cutoff in cutoffs:
                    stats = agg[cutoff]
                    kept = int(min(valid_points, cutoff))
                    kept_cells = cells[:kept]
                    kept_amps = amps[:kept]
                    kept_unique, kept_counts = np.unique(kept_cells, return_counts=True)
                    unique_readouts_in_kept = int(kept_unique.size)
                    runs = run_length_count(kept_cells)
                    idx = np.searchsorted(full_unique, kept_unique)
                    full_counts_for_kept = full_counts[idx]
                    wholly = int(np.sum(kept_counts == full_counts_for_kept))
                    partial = int(unique_readouts_in_kept - wholly)
                    included_amp = float(np.sum(kept_amps))
                    remnant_amp = float(total_amp - included_amp)
                    if total_amp > 0.0:
                        included_frac = included_amp / total_amp
                        remnant_frac = remnant_amp / total_amp
                    else:
                        included_frac = float("nan")
                        remnant_frac = float("nan")

                    stats["events_seen"] += 1
                    stats["sum_valid_points"] += valid_points
                    stats["sum_kept_points"] += kept
                    stats["sum_unique_readouts_in_kept"] += unique_readouts_in_kept
                    stats["sum_readout_runs"] += runs
                    stats["sum_wholly_readouts"] += wholly
                    stats["sum_partial_readouts"] += partial
                    stats["sum_total_amp"] += total_amp
                    stats["sum_included_amp"] += included_amp
                    stats["sum_remnant_amp"] += remnant_amp
                    if np.isfinite(included_frac):
                        stats["sum_included_amp_frac"] += included_frac
                        stats["sum_remnant_amp_frac"] += remnant_frac
                        stats["events_with_finite_amp_frac"] += 1
                    if partial > 0:
                        partial_cells = kept_unique[kept_counts < full_counts_for_kept]
                        partial_cell_counter[cutoff].update(partial_cells.tolist())

                    exact_boundary = None
                    split_cellid = None
                    if valid_points > cutoff:
                        stats["events_exceed_cutoff"] += 1
                        left = int(cells[cutoff - 1])
                        right = int(cells[cutoff])
                        exact_boundary = int(left != right)
                        split_cellid = left if left == right else None
                        if exact_boundary:
                            stats["events_exact_boundary"] += 1
                        else:
                            stats["events_split_waveform"] += 1
                            split_cell_counter[cutoff][left] += 1

                    event_rows.append(
                        f"{path.name},{event_idx},{valid_points},{cutoff},{kept},{full_unique_count},{unique_readouts_in_kept},{runs},{wholly},{partial},"
                        f"{included_amp:.10g},{remnant_amp:.10g},{'' if not np.isfinite(included_frac) else f'{included_frac:.10g}'},{'' if not np.isfinite(remnant_frac) else f'{remnant_frac:.10g}'},"
                        f"{'' if exact_boundary is None else exact_boundary},{'' if split_cellid is None else split_cellid}"
                    )

    summary: Dict[str, Dict[str, object]] = {}
    for cutoff in cutoffs:
        s = agg[cutoff]
        seen = max(int(s["events_seen"]), 1)
        exceed = int(s["events_exceed_cutoff"])
        exact = int(s["events_exact_boundary"])
        split = int(s["events_split_waveform"])
        finite_amp = max(int(s["events_with_finite_amp_frac"]), 1)
        summary[str(cutoff)] = {
            "events_seen": int(s["events_seen"]),
            "events_exceed_cutoff": exceed,
            "avg_valid_points": as_float(s["sum_valid_points"] / seen),
            "avg_kept_points": as_float(s["sum_kept_points"] / seen),
            "avg_unique_readouts": as_float(s["sum_unique_readouts_full"] / seen),
            "avg_unique_readouts_in_kept": as_float(s["sum_unique_readouts_in_kept"] / seen),
            "avg_readout_runs_in_kept": as_float(s["sum_readout_runs"] / seen),
            "avg_wholly_included_readouts_in_kept": as_float(s["sum_wholly_readouts"] / seen),
            "avg_partial_readouts_in_kept": as_float(s["sum_partial_readouts"] / seen),
            "avg_included_amplitude_sum": as_float(s["sum_included_amp"] / seen),
            "avg_remnant_amplitude_sum": as_float(s["sum_remnant_amp"] / seen),
            "avg_included_amplitude_fraction": as_float(s["sum_included_amp_frac"] / finite_amp),
            "avg_remnant_amplitude_fraction": as_float(s["sum_remnant_amp_frac"] / finite_amp),
            "global_included_amplitude_fraction": as_float(s["sum_included_amp"] / s["sum_total_amp"]) if s["sum_total_amp"] > 0 else None,
            "global_remnant_amplitude_fraction": as_float(s["sum_remnant_amp"] / s["sum_total_amp"]) if s["sum_total_amp"] > 0 else None,
            "frac_exact_boundary_given_exceed": as_float(exact / exceed) if exceed > 0 else None,
            "frac_split_waveform_given_exceed": as_float(split / exceed) if exceed > 0 else None,
            "top_split_cellids": split_cell_counter[cutoff].most_common(20),
            "top_partial_cellids": partial_cell_counter[cutoff].most_common(20),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    details_path = Path(args.details_output)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text("\n".join(event_rows) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary: {output_path}")
    print(f"Wrote event details: {details_path}")


if __name__ == "__main__":
    main()
