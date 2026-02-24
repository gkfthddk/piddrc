#!/usr/bin/env python
"""Debug one event/cell bridge between RawCalorimeterHits and DigiWaveforms."""

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

RAW_AMP = "RawCalorimeterHits/RawCalorimeterHits.amplitude"
RAW_CID = "RawCalorimeterHits/RawCalorimeterHits.cellID"
DIGI_CID = "DigiCalorimeterHits/DigiCalorimeterHits.cellID"
DIGI_TYPE = "DigiCalorimeterHits/DigiCalorimeterHits.type"
WF_CID = "DigiWaveforms/DigiWaveforms.cellID"
WF_T0 = "DigiWaveforms/DigiWaveforms.time"
WF_DT = "DigiWaveforms/DigiWaveforms.interval"
WF_BEG = "DigiWaveforms/DigiWaveforms.amplitude_begin"
WF_END = "DigiWaveforms/DigiWaveforms.amplitude_end"
WF_AMP = "_DigiWaveforms_amplitude"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot one scintillation waveform for an S_amp==0 event.")
    p.add_argument("--reco-file", type=Path, default=None, help="Reco .h5py file path.")
    p.add_argument(
        "--list-file",
        type=Path,
        default=Path("h5s/validation_reports/e-_1-120GeV_readable_mixed_reco.txt"),
        help="Fallback list file if --reco-file is not given.",
    )
    p.add_argument("--digi-base", type=Path, default=Path("/hdfs/ml/dualreadout/v001"), help="Base digi directory.")
    p.add_argument("--event-index", type=int, default=-1, help="Specific event index (>=0). If <0, random S_amp==0 event.")
    p.add_argument("--max-search-events", type=int, default=500, help="Search cap when --event-index<0.")
    p.add_argument("--eps", type=float, default=1e-12, help="Zero threshold for S_amp.")
    p.add_argument("--use-abs-raw", action="store_true", help="Use abs(raw amplitude) for raw sums.")
    p.add_argument("--seed", type=int, default=12345, help="Random seed for reproducible selection.")
    p.add_argument("--out-dir", type=Path, default=Path("analysis/one_waveform_debug"), help="Output dir.")
    return p.parse_args()


def _pick_reco(args: argparse.Namespace) -> Path:
    if args.reco_file is not None:
        return args.reco_file
    lines = [ln.strip() for ln in args.list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"No reco files in list: {args.list_file}")
    return Path(lines[0])


def _to_digi(reco: Path, digi_base: Path) -> Path:
    return digi_base / reco.parent.name / f"digi_{reco.stem}.root"


def _raw_channel_sums_for_event(raw_amp: np.ndarray, raw_cid: np.ndarray, digi_cid: np.ndarray, digi_type: np.ndarray) -> tuple[float, float, dict[int, float]]:
    cids_s = digi_cid[digi_type == 0]
    cids_c = digi_cid[digi_type != 0]
    m_s = np.isin(raw_cid, cids_s)
    m_c = np.isin(raw_cid, cids_c)

    raw_s_sum = float(np.sum(raw_amp[m_s], dtype=np.float64))
    raw_c_sum = float(np.sum(raw_amp[m_c], dtype=np.float64))

    per_cell: dict[int, float] = {}
    if np.any(m_s):
        s_cid = raw_cid[m_s]
        s_amp = raw_amp[m_s]
        for cid, amp in zip(s_cid, s_amp):
            k = int(cid)
            per_cell[k] = per_cell.get(k, 0.0) + float(amp)
    return raw_s_sum, raw_c_sum, per_cell


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    reco = _pick_reco(args)
    digi = _to_digi(reco, args.digi_base)
    if not reco.is_file():
        raise FileNotFoundError(f"Reco file not found: {reco}")
    if not digi.is_file():
        raise FileNotFoundError(f"Digi file not found: {digi}")

    with h5py.File(reco, "r") as h:
        for k in ("S_amp", "C_amp"):
            if k not in h:
                raise KeyError(f"Missing key in reco: {k}")
        s_amp = np.asarray(h["S_amp"][:], dtype=np.float64).reshape(-1)
        c_amp = np.asarray(h["C_amp"][:], dtype=np.float64).reshape(-1)

    with uproot.open(digi) as f:
        if "events" not in f:
            raise KeyError(f"Missing events tree in {digi}")
        t = f["events"]
        needed = [RAW_AMP, RAW_CID, DIGI_CID, DIGI_TYPE, WF_CID, WF_T0, WF_DT, WF_BEG, WF_END, WF_AMP]
        for b in needed:
            if b not in t:
                raise KeyError(f"Missing branch in digi: {b}")

        n_evt = min(len(s_amp), int(t.num_entries))
        if args.event_index >= 0:
            evt = args.event_index
            if evt >= n_evt:
                raise IndexError(f"event-index {evt} >= n_events {n_evt}")
        else:
            evt = -1
            candidates: list[int] = []
            stop = min(n_evt, args.max_search_events if args.max_search_events > 0 else n_evt)
            for i in range(stop):
                if s_amp[i] <= args.eps:
                    candidates.append(i)
            if not candidates:
                raise RuntimeError("No event found with S_amp==0 in search range.")
            evt = int(rng.choice(np.asarray(candidates, dtype=np.int64)))

        arr = t.arrays(needed, entry_start=evt, entry_stop=evt + 1, library="ak")

    raw_amp = np.asarray(arr[RAW_AMP][0], dtype=np.float64)
    raw_cid = np.asarray(arr[RAW_CID][0], dtype=np.int64)
    digi_cid = np.asarray(arr[DIGI_CID][0], dtype=np.int64)
    digi_type = np.asarray(arr[DIGI_TYPE][0], dtype=np.int64)
    if args.use_abs_raw:
        raw_amp = np.abs(raw_amp)

    raw_s_sum, raw_c_sum, raw_s_per_cell = _raw_channel_sums_for_event(raw_amp, raw_cid, digi_cid, digi_type)

    wf_cell = np.asarray(arr[WF_CID][0], dtype=np.int64)
    wf_t0 = np.asarray(arr[WF_T0][0], dtype=np.float64)
    wf_dt = np.asarray(arr[WF_DT][0], dtype=np.float64)
    wf_beg = np.asarray(arr[WF_BEG][0], dtype=np.int64)
    wf_end = np.asarray(arr[WF_END][0], dtype=np.int64)
    wf_flat = np.asarray(arr[WF_AMP][0], dtype=np.float64)

    # Pick random scintillation waveform/cell in this event.
    s_cells = np.unique(digi_cid[digi_type == 0])
    wf_scint_mask = np.isin(wf_cell, s_cells)
    wf_scint_indices = np.where(wf_scint_mask)[0]
    if wf_scint_indices.size > 0:
        pick_wf = int(rng.choice(wf_scint_indices))
        target_cell = int(wf_cell[pick_wf])
    elif raw_s_per_cell:
        target_cell = int(rng.choice(np.asarray(list(raw_s_per_cell.keys()), dtype=np.int64)))
    else:
        raise RuntimeError("Selected S_amp==0 event has no scintillation waveform/cell to draw.")
    wf_indices = np.where(wf_cell == target_cell)[0]

    fig, ax0 = plt.subplots(1, 1, figsize=(10, 5), sharex=False)

    if wf_cell.size == 0:
        ax0.text(0.05, 0.5, "No DigiWaveforms in selected event", transform=ax0.transAxes)
    elif wf_indices.size == 0:
        ax0.text(0.05, 0.5, "No DigiWaveforms for target scintillation cell", transform=ax0.transAxes)
    else:
        # Draw full event waveform range first (all cells), then highlight target cell.
        t_min = np.inf
        t_max = -np.inf
        for wi in range(wf_cell.size):
            b = int(wf_beg[wi])
            e = int(wf_end[wi])
            if e <= b or b < 0 or e > wf_flat.size:
                continue
            y = wf_flat[b:e]
            t = wf_t0[wi] + wf_dt[wi] * np.arange(y.size, dtype=np.float64)
            if t.size > 0:
                t_min = min(t_min, float(t[0]))
                t_max = max(t_max, float(t[-1]))
            ax0.plot(t, y, linewidth=0.8, alpha=0.25, color="gray")

        for wi in wf_indices:
            b = int(wf_beg[wi])
            e = int(wf_end[wi])
            if e <= b or b < 0 or e > wf_flat.size:
                continue
            y = wf_flat[b:e]
            t = wf_t0[wi] + wf_dt[wi] * np.arange(y.size, dtype=np.float64)
            ax0.plot(t, y, linewidth=1.5, alpha=0.95, color="tab:blue")
        if np.isfinite(t_min) and np.isfinite(t_max) and t_max > t_min:
            ax0.set_xlim(t_min, t_max)
    ax0.set_title(f"DigiWaveforms full range (event {evt}), highlight cellID={target_cell}")
    ax0.set_xlabel("time")
    ax0.set_ylabel("waveform amplitude")
    ax0.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    plot_path = args.out_dir / "one_waveform_bridge.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary: dict[str, Any] = {
        "reco_file": str(reco),
        "digi_file": str(digi),
        "event_index": int(evt),
        "S_amp_event": float(s_amp[evt]) if evt < len(s_amp) else None,
        "C_amp_event": float(c_amp[evt]) if evt < len(c_amp) else None,
        "raw_s_sum_event": float(raw_s_sum),
        "raw_c_sum_event": float(raw_c_sum),
        "target_cellid": int(target_cell),
        "target_cell_raw_s_sum": float(raw_s_per_cell.get(target_cell, 0.0)),
        "num_waveforms_target_cell": int(wf_indices.size),
        "plot": str(plot_path),
    }
    (args.out_dir / "one_waveform_bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
