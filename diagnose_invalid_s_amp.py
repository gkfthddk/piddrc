#!/usr/bin/env python
"""Diagnose zero S_amp/C_amp using digi RawCalorimeterHits channel sums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import awkward as ak
import h5py
import matplotlib.pyplot as plt
import numpy as np
import uproot
from tqdm.auto import tqdm

RAW_AMP = "RawCalorimeterHits/RawCalorimeterHits.amplitude"
RAW_CID = "RawCalorimeterHits/RawCalorimeterHits.cellID"
DIGI_CID = "DigiCalorimeterHits/DigiCalorimeterHits.cellID"
DIGI_TYPE = "DigiCalorimeterHits/DigiCalorimeterHits.type"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose S_amp/C_amp zeros vs RawCalorimeterHits sums.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("h5s/validation_reports"),
        help="Directory containing validation report txt files.",
    )
    parser.add_argument(
        "--list-file",
        action="append",
        type=Path,
        default=None,
        help="Explicit reco list txt file(s). Repeatable.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Restrict default-discovered lists to sample name(s). Repeatable.",
    )
    parser.add_argument("--max-files", type=int, default=100, help="Max reco files to process (<=0 means all).")
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=0,
        help="Max events per file (<=0 means all).",
    )
    parser.add_argument(
        "--digi-base",
        type=Path,
        default=Path("/hdfs/ml/dualreadout/v001"),
        help="Base directory for digi ROOT files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/invalid_s_amp_diagnosis"),
        help="Output directory for plots and summary JSON.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Zero threshold used for stored/raw sums.",
    )
    parser.add_argument(
        "--use-abs-raw",
        action="store_true",
        help="Use absolute RawCalorimeterHits amplitude when summing channels.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plotting and only write summary.json.",
    )
    return parser.parse_args()


def _discover_default_lists(reports_dir: Path, selected_samples: set[str] | None) -> list[Path]:
    lists = sorted(reports_dir.glob("*_readable_but_invalid_reco.txt"))
    lists += sorted(reports_dir.glob("*_readable_mixed_reco.txt"))
    if selected_samples is None:
        return lists
    out: list[Path] = []
    for p in lists:
        n = p.name
        for suffix in ("_readable_but_invalid_reco.txt", "_readable_mixed_reco.txt"):
            if n.endswith(suffix):
                sample = n[: -len(suffix)]
                if sample in selected_samples:
                    out.append(p)
                break
    return out


def _load_reco_paths(list_files: list[Path]) -> list[Path]:
    reco_set: set[Path] = set()
    for lp in list_files:
        if not lp.is_file():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                reco_set.add(Path(s))
    return sorted(reco_set)


def _reco_to_digi(reco_path: Path, digi_base: Path) -> Path:
    sample = reco_path.parent.name
    return digi_base / sample / f"digi_{reco_path.stem}.root"


def _safe_hist_log(values: np.ndarray, title: str, xlabel: str, out_path: Path) -> bool:
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return False
    vmin = np.min(finite)
    vmax = np.max(finite)
    if vmax <= vmin:
        return False
    bins = np.logspace(np.log10(vmin), np.log10(vmax), 120)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(finite, bins=bins, histtype="step", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log", nonpositive="clip")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected_samples = set(args.sample) if args.sample else None
    list_files = args.list_file or _discover_default_lists(args.reports_dir, selected_samples)
    if not list_files:
        raise RuntimeError("No reco list files found. Pass --list-file or generate validation reports first.")

    reco_files = _load_reco_paths(list_files)
    if args.max_files > 0:
        reco_files = reco_files[: args.max_files]
    if not reco_files:
        raise RuntimeError("No reco files found in selected list files.")

    s_amp_sel: list[float] = []
    c_amp_sel: list[float] = []
    raw_s_sel: list[float] = []
    raw_c_sel: list[float] = []

    skipped: list[dict[str, str]] = []
    total_events_seen = 0
    total_selected = 0
    total_raw_hits = 0
    total_raw_hits_unmatched = 0

    for reco in tqdm(reco_files, desc="Scanning reco+digi", unit="file"):
        digi = _reco_to_digi(reco, args.digi_base)
        if not digi.is_file():
            skipped.append({"file": str(reco), "reason": f"missing_digi:{digi}"})
            continue

        try:
            with h5py.File(reco, "r") as h:
                if "S_amp" not in h or "C_amp" not in h:
                    skipped.append({"file": str(reco), "reason": "missing_S_amp_or_C_amp"})
                    continue
                s_amp = np.asarray(h["S_amp"][:], dtype=np.float64).reshape(-1)
                c_amp = np.asarray(h["C_amp"][:], dtype=np.float64).reshape(-1)

            with uproot.open(digi) as dr:
                if "events" not in dr:
                    skipped.append({"file": str(reco), "reason": "digi_missing_events_tree"})
                    continue
                tree = dr["events"]
                for b in (RAW_AMP, RAW_CID, DIGI_CID, DIGI_TYPE):
                    if b not in tree:
                        skipped.append({"file": str(reco), "reason": f"digi_missing_branch:{b}"})
                        tree = None
                        break
                if tree is None:
                    continue

                entry_stop = None if args.max_events_per_file <= 0 else args.max_events_per_file
                arr = tree.arrays([RAW_AMP, RAW_CID, DIGI_CID, DIGI_TYPE], entry_stop=entry_stop, library="ak")

            n_evt = min(len(s_amp), len(c_amp), len(arr[RAW_AMP]), len(arr[DIGI_CID]))
            if n_evt <= 0:
                skipped.append({"file": str(reco), "reason": "zero_events"})
                continue

            for i in range(n_evt):
                total_events_seen += 1
                s0 = float(s_amp[i])
                c0 = float(c_amp[i])
                if not ((s0 <= args.eps) or (c0 <= args.eps)):
                    continue

                raw_amp = np.asarray(arr[RAW_AMP][i], dtype=np.float64)
                raw_cid = np.asarray(arr[RAW_CID][i], dtype=np.int64)
                digi_cid = np.asarray(arr[DIGI_CID][i], dtype=np.int64)
                digi_type = np.asarray(arr[DIGI_TYPE][i], dtype=np.int64)

                if args.use_abs_raw:
                    raw_amp = np.abs(raw_amp)

                total_raw_hits += int(raw_amp.size)
                if raw_amp.size == 0:
                    raw_s_sum = 0.0
                    raw_c_sum = 0.0
                else:
                    cids_s = digi_cid[digi_type == 0]
                    cids_c = digi_cid[digi_type != 0]

                    m_s = np.isin(raw_cid, cids_s)
                    m_c = np.isin(raw_cid, cids_c)
                    matched = m_s | m_c
                    total_raw_hits_unmatched += int(np.sum(~matched))

                    raw_s_sum = float(np.sum(raw_amp[m_s], dtype=np.float64))
                    raw_c_sum = float(np.sum(raw_amp[m_c], dtype=np.float64))

                s_amp_sel.append(s0)
                c_amp_sel.append(c0)
                raw_s_sel.append(raw_s_sum)
                raw_c_sel.append(raw_c_sum)
                total_selected += 1

        except Exception as exc:
            skipped.append({"file": str(reco), "reason": f"open_error:{exc}"})

    if total_selected == 0:
        raise RuntimeError("No selected events (S_amp==0 or C_amp==0) found in processed files.")

    s_amp_np = np.asarray(s_amp_sel, dtype=np.float64)
    c_amp_np = np.asarray(c_amp_sel, dtype=np.float64)
    raw_s_np = np.asarray(raw_s_sel, dtype=np.float64)
    raw_c_np = np.asarray(raw_c_sel, dtype=np.float64)

    eps = float(args.eps)
    s_zero = s_amp_np <= eps
    c_zero = c_amp_np <= eps
    raw_s_zero = raw_s_np <= eps
    raw_c_zero = raw_c_np <= eps

    def frac(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if mask.size else float("nan")

    summary: dict[str, Any] = {
        "num_files_input": len(reco_files),
        "num_files_skipped": len(skipped),
        "num_events_total_seen": int(total_events_seen),
        "num_events_selected": int(total_selected),
        "eps": eps,
        "use_abs_raw": bool(args.use_abs_raw),
        "raw_hit_matching": {
            "total_raw_hits": int(total_raw_hits),
            "total_raw_hits_unmatched": int(total_raw_hits_unmatched),
            "unmatched_fraction": (
                float(total_raw_hits_unmatched / total_raw_hits) if total_raw_hits > 0 else None
            ),
        },
        "fractions": {
            "S_amp_zero": frac(s_zero),
            "C_amp_zero": frac(c_zero),
            "raw_s_sum_zero": frac(raw_s_zero),
            "raw_c_sum_zero": frac(raw_c_zero),
            "S_amp_zero_and_raw_s_zero": frac(s_zero & raw_s_zero),
            "S_amp_zero_but_raw_s_positive": frac(s_zero & (~raw_s_zero)),
            "C_amp_zero_and_raw_c_zero": frac(c_zero & raw_c_zero),
            "C_amp_zero_but_raw_c_positive": frac(c_zero & (~raw_c_zero)),
        },
        "skipped_files": skipped,
        "notes": [
            "Only events with S_amp==0 or C_amp==0 are included.",
            "RawCalorimeterHits are split into S/C channels by matching raw cellID to DigiCalorimeterHits cellID/type within each event.",
            "Digi type convention used: type==0 -> scintillation, type!=0 -> Cherenkov.",
        ],
    }

    if not args.no_plots:
        _safe_hist_log(raw_s_np[s_zero], "Raw scintillation sum for events with S_amp==0", "raw_s_sum", args.out_dir / "raw_s_for_s_zero.png")
        _safe_hist_log(raw_c_np[c_zero], "Raw Cherenkov sum for events with C_amp==0", "raw_c_sum", args.out_dir / "raw_c_for_c_zero.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        mask = np.isfinite(s_amp_np) & np.isfinite(raw_s_np)
        if np.any(mask):
            ax.scatter(raw_s_np[mask], s_amp_np[mask], s=3, alpha=0.2)
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("raw scintillation sum (from RawCalorimeterHits)")
        ax.set_ylabel("stored S_amp (reco)")
        ax.set_title("S_amp vs RawCalorimeterHits scintillation sum")
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.out_dir / "s_amp_vs_raw_s.png", dpi=160)
        plt.close(fig)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnosis outputs to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
