#!/usr/bin/env python
"""Compare shower timing observables using event-level timing quantiles."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ALIASES: Dict[str, str] = {
    "time": "DRcalo3dHits.time",
    "time_end": "DRcalo3dHits.time_end",
    "weight": "DRcalo3dHits.amplitude_sum",
    "amp": "DRcalo3dHits.amplitude_sum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare shower timing quantiles across HDF5 samples.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "h5s/e-_1-120GeV.h5py",
            "h5s/gamma_1-120GeV.h5py",
            "h5s/pi0_1-120GeV.h5py",
            "h5s/pi+_1-120GeV.h5py",
            "h5s/kaon+_1-120GeV.h5py",
        ],
        help="Input HDF5 files or globs (for example: h5s/pi+_1-120GeV*.h5py h5s/kaon+_1-120GeV*.h5py).",
    )
    parser.add_argument(
        "--time-key",
        default="time",
        help="Timing dataset key or alias. Default uses DRcalo3dHits.time.",
    )
    parser.add_argument(
        "--time-end-key",
        default="time_end",
        help="Timing end dataset key or alias. Used together with --time-key to build hit-center and hit-duration observables.",
    )
    parser.add_argument(
        "--weight-key",
        default="weight",
        help="Optional weight dataset key or alias. Use '' or 'none' for unweighted quantiles.",
    )
    parser.add_argument(
        "--energy-key",
        default="E_gen",
        help="Per-event energy dataset used for E_gen range filtering.",
    )
    parser.add_argument(
        "--edep-key",
        default="E_dep",
        help="Per-event deposited-energy dataset used for timing vs E_dep scatters.",
    )
    parser.add_argument(
        "--eleak-key",
        default="E_leak",
        help="Per-event leakage-energy dataset used for timing vs E_leak scatters.",
    )
    parser.add_argument(
        "--c-amp-key",
        default="C_amp",
        help="Per-event Cherenkov amplitude dataset used for timing vs C_amp/S_amp.",
    )
    parser.add_argument(
        "--s-amp-key",
        default="S_amp",
        help="Per-event scintillation amplitude dataset used for timing vs C_amp/S_amp.",
    )
    parser.add_argument(
        "--theta-key",
        default="GenParticles.momentum.theta",
        help="Per-event theta dataset used for q50 vs theta scatter.",
    )
    parser.add_argument(
        "--phi-key",
        default="GenParticles.momentum.phi",
        help="Per-event phi dataset used for timing-width vs phi scatter.",
    )
    parser.add_argument("--e-min", type=float, default=None, help="Minimum E_gen to keep (inclusive).")
    parser.add_argument("--e-max", type=float, default=None, help="Maximum E_gen to keep (inclusive).")
    parser.add_argument("--theta-min", type=float, default=None, help="Minimum theta to keep (inclusive).")
    parser.add_argument("--theta-max", type=float, default=None, help="Maximum theta to keep (inclusive).")
    parser.add_argument(
        "--late-time-threshold",
        type=float,
        default=100.0,
        help="Time threshold used to define late-energy fraction.",
    )
    parser.add_argument(
        "--band-split-threshold",
        type=float,
        default=100.0,
        help="Preferred spread threshold used to split low-band and high-band events.",
    )
    parser.add_argument(
        "--q50-band-center",
        type=float,
        default=475.0,
        help="Center of the suspicious q50 band used for amplitude-split diagnostics.",
    )
    parser.add_argument(
        "--q50-band-halfwidth",
        type=float,
        default=10.0,
        help="Half-width around --q50-band-center used to define the q50 band.",
    )
    parser.add_argument(
        "--plot-q50-band-waveform",
        action="store_true",
        help="Draw example event waveform proxies for events with q50 near --q50-band-center.",
    )
    parser.add_argument(
        "--waveform-only",
        action="store_true",
        help="Only scan and draw q50-band waveform examples. Skip the full timing-summary workflow.",
    )
    parser.add_argument(
        "--waveform-bin-width",
        type=float,
        default=1.0,
        help="Time-bin width used for the interval-smeared waveform proxy and waveform-based timing quantiles.",
    )
    parser.add_argument(
        "--waveform-max-intervals",
        type=int,
        default=48,
        help="Maximum number of highest-amplitude hit intervals to show in the interval panel.",
    )
    parser.add_argument(
        "--waveform-max-examples",
        type=int,
        default=1,
        help="Maximum number of q50-band waveform examples to draw per sample label.",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.5, 0.8, 0.9],
        help="Timing quantiles to compute per event.",
    )
    parser.add_argument("--bins", type=int, default=50, help="Histogram bin count.")
    parser.add_argument("--max-events", type=int, default=5000, help="Per-file event cap (<=0 means all).")
    parser.add_argument(
        "--read-chunk-events",
        type=int,
        default=256,
        help="Number of events to read per chunk from large timing datasets.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional per-event point cap applied before quantile computation (<=0 means all points).",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Normalize histograms to density instead of event count.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/timing_quantiles",
        help="Output directory for plots and summary JSON.",
    )
    return parser.parse_args()


def _canonical_key(name: str) -> str | None:
    lowered = name.strip().lower()
    if lowered in {"", "none", "null", "false"}:
        return None
    return ALIASES.get(lowered, name)


def _resolve_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    roots = [Path("."), Path("h5s"), Path("/store/ml/dual-readout/h5s")]
    for pattern in patterns:
        found: List[Path] = []
        for root in roots:
            found.extend(sorted(root.glob(pattern)))
        if found:
            paths.extend(found)
            continue
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)
            continue
        for root in roots[1:]:
            alt = root / pattern
            if alt.exists():
                paths.append(alt)
                break

    dedup: List[Path] = []
    seen = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except FileNotFoundError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup


def _sample_label(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1]) or stem
    return stem


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return math.nan
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    total_weight = float(cumulative[-1])
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        return math.nan
    target = float(np.clip(quantile, 0.0, 1.0)) * total_weight
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 0), sorted_values.size - 1)
    return float(sorted_values[idx])


def _safe_summary(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "p10": math.nan,
            "p50": math.nan,
            "p90": math.nan,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def _safe_label_slug(label: str) -> str:
    chars = []
    for ch in label:
        if ch.isalnum() or ch in {"-", "_", "+"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def _read_event_vector(handle: h5py.File, key: str | None, take: int, ref_key: str) -> np.ndarray | None:
    if key is None:
        return None
    if key not in handle:
        raise KeyError(f"{key} is missing in {handle.filename}")
    ds = handle[key]
    if ds.ndim == 0:
        values = np.asarray([ds[()]], dtype=np.float64)
    else:
        if ds.shape[0] < take:
            raise ValueError(f"{key} in {handle.filename} has fewer events than {ref_key}.")
        values = np.asarray(ds[:take], dtype=np.float64).reshape(-1)
    if values.shape[0] != take:
        raise ValueError(f"{key} in {handle.filename} produced {values.shape[0]} entries for {take} events.")
    return values


def _iter_contiguous_runs(indices: np.ndarray, max_run: int) -> Iterable[tuple[int, int]]:
    if indices.size == 0:
        return
    max_run = max(1, int(max_run))
    run_start = int(indices[0])
    prev = int(indices[0])
    run_len = 1
    for idx_value in indices[1:]:
        idx = int(idx_value)
        contiguous = idx == prev + 1
        if contiguous and run_len < max_run:
            prev = idx
            run_len += 1
            continue
        yield run_start, prev + 1
        run_start = idx
        prev = idx
        run_len = 1
    yield run_start, prev + 1


def _quantile_key(quantile: float) -> str:
    return f"q{int(round(float(quantile) * 100.0)):02d}"


def _median_like_key(quantiles: Sequence[float]) -> str:
    if not quantiles:
        return "q50"
    return _quantile_key(min(quantiles, key=lambda q: abs(float(q) - 0.5)))


def _preferred_spread_key(quantile_keys: Sequence[str]) -> str | None:
    if "q10" in quantile_keys and "q90" in quantile_keys:
        return "q90_minus_q10"
    if "q20" in quantile_keys and "q90" in quantile_keys:
        return "q90_minus_q20"
    if "q30" in quantile_keys and "q80" in quantile_keys:
        return "q80_minus_q30"
    if len(quantile_keys) >= 2:
        return f"{quantile_keys[-1]}_minus_{quantile_keys[0]}"
    return None


def _add_quantile_differences(target: Dict[str, np.ndarray], *, prefix: str = "") -> None:
    def _has(*keys: str) -> bool:
        return all(f"{prefix}{key}" in target for key in keys)

    def _assign(name: str, upper: str, lower: str) -> None:
        values = target[f"{prefix}{upper}"] - target[f"{prefix}{lower}"]
        target[f"{prefix}{name}"] = np.asarray(values, dtype=np.float64)

    if _has("q30", "q80"):
        _assign("q80_minus_q30", "q80", "q30")
    if _has("q10", "q90"):
        _assign("q90_minus_q10", "q90", "q10")
    if _has("q20", "q90"):
        _assign("q90_minus_q20", "q90", "q20")
    if _has("q10", "q30"):
        _assign("q30_minus_q10", "q30", "q10")
    if _has("q30", "q50"):
        _assign("q50_minus_q30", "q50", "q30")
    if _has("q50", "q80"):
        _assign("q80_minus_q50", "q80", "q50")
    if _has("q50", "q90"):
        _assign("q90_minus_q50", "q90", "q50")
    if _has("q20", "q50"):
        _assign("q50_minus_q20", "q50", "q20")


def _extract_prefixed_view(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    prefix: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    view: Dict[str, Dict[str, np.ndarray]] = {}
    for label, values in samples.items():
        mapped: Dict[str, np.ndarray] = {}
        for key, array in values.items():
            if key.startswith(prefix):
                mapped[key[len(prefix):]] = array
            elif not key.startswith("dur_"):
                mapped[key] = array
        if mapped:
            view[label] = mapped
    return view


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180)
    print(f"Saved plot: {path}")


def _paired_series(
    values: Mapping[str, np.ndarray],
    x_key: str,
    y_key: str,
    *,
    positive_x: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values.get(x_key, []), dtype=np.float64).reshape(-1)
    y = np.asarray(values.get(y_key, []), dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"Mismatched series lengths for {x_key} ({x.size}) and {y_key} ({y.size})")
    valid = np.isfinite(x) & np.isfinite(y)
    if positive_x:
        valid &= x > 0.0
    return x[valid], y[valid]


def _collect_file_quantiles(
    path: Path,
    *,
    time_key: str,
    time_end_key: str,
    weight_key: str | None,
    energy_key: str | None,
    edep_key: str | None,
    eleak_key: str | None,
    c_amp_key: str | None,
    s_amp_key: str | None,
    theta_key: str | None,
    phi_key: str | None,
    e_min: float | None,
    e_max: float | None,
    theta_min: float | None,
    theta_max: float | None,
    quantiles: Sequence[float],
    max_events: int,
    read_chunk_events: int,
    waveform_bin_width: float,
    max_points: int,
    late_time_threshold: float,
) -> Tuple[Dict[str, np.ndarray], int]:
    out = {f"q{int(round(q * 100.0)):02d}": [] for q in quantiles}
    for q in quantiles:
        out[f"dur_q{int(round(q * 100.0)):02d}"] = []
    selected_energies: List[float] = []
    selected_edep: List[float] = []
    selected_eleak: List[float] = []
    selected_c_amp: List[float] = []
    selected_s_amp: List[float] = []
    selected_thetas: List[float] = []
    selected_phis: List[float] = []
    selected_late_fraction: List[float] = []
    selected_max_valid_point: List[float] = []
    with h5py.File(path, "r") as handle:
        if time_key not in handle:
            raise KeyError(f"{time_key} is missing in {path}")
        if time_end_key not in handle:
            raise KeyError(f"{time_end_key} is missing in {path}")
        time_ds = handle[time_key]
        time_end_ds = handle[time_end_key]
        if time_ds.ndim < 2:
            raise ValueError(f"{time_key} in {path} must be at least 2D (event, hit).")
        if time_end_ds.shape != time_ds.shape:
            raise ValueError(f"{time_end_key} shape {time_end_ds.shape} does not match {time_key} shape {time_ds.shape} in {path}.")

        n_events = int(time_ds.shape[0])
        take = n_events if max_events <= 0 else min(n_events, max_events)
        event_mask = np.ones(take, dtype=bool)
        energies = _read_event_vector(handle, energy_key, take, time_key)
        if energies is not None:
            event_mask &= np.isfinite(energies)
            if e_min is not None:
                event_mask &= energies >= e_min
            if e_max is not None:
                event_mask &= energies <= e_max
        edeps = _read_event_vector(handle, edep_key, take, time_key)
        if edeps is not None:
            event_mask &= np.isfinite(edeps)
        eleaks = _read_event_vector(handle, eleak_key, take, time_key)
        if eleaks is not None:
            event_mask &= np.isfinite(eleaks)
        c_amps = _read_event_vector(handle, c_amp_key, take, time_key)
        if c_amps is not None:
            event_mask &= np.isfinite(c_amps)
        s_amps = _read_event_vector(handle, s_amp_key, take, time_key)
        if s_amps is not None:
            event_mask &= np.isfinite(s_amps)
        thetas = _read_event_vector(handle, theta_key, take, time_key)
        if thetas is not None:
            event_mask &= np.isfinite(thetas)
            if theta_min is not None:
                event_mask &= thetas >= theta_min
            if theta_max is not None:
                event_mask &= thetas <= theta_max
        phis = _read_event_vector(handle, phi_key, take, time_key)
        if phis is not None:
            event_mask &= np.isfinite(phis)

        weights = None
        if weight_key is not None:
            if weight_key not in handle:
                raise KeyError(f"{weight_key} is missing in {path}")
            weight_ds = handle[weight_key]
            if weight_ds.shape[0] < take:
                raise ValueError(f"{weight_key} in {path} has fewer events than {time_key}.")
            expected_tail = tuple(time_ds.shape[1:])
            if tuple(weight_ds.shape[1:]) != expected_tail:
                raise ValueError(
                    f"{weight_key} tail shape {weight_ds.shape[1:]} does not match {time_key} tail shape {expected_tail} in {path}."
                )

        chunk_size = max(1, int(read_chunk_events))
        selected_indices = np.flatnonzero(event_mask)
        event_iter = tqdm(total=int(selected_indices.size), desc=f"{path.name} events", unit="evt", leave=False)
        for run_start, run_stop in _iter_contiguous_runs(selected_indices, chunk_size):
            times = np.asarray(time_ds[run_start:run_stop], dtype=np.float64)
            time_ends = np.asarray(time_end_ds[run_start:run_stop], dtype=np.float64)
            if weight_key is not None:
                weight_chunk = np.asarray(weight_ds[run_start:run_stop], dtype=np.float64)
            else:
                weight_chunk = None
            if weight_chunk is not None and weight_chunk.shape != times.shape:
                raise ValueError(
                    f"{weight_key} chunk shape {weight_chunk.shape} does not match {time_key} chunk shape {times.shape} in {path}."
                )
            for local_idx, idx in enumerate(range(run_start, run_stop)):
                row_time = np.asarray(times[local_idx]).reshape(-1)
                row_time_end = np.asarray(time_ends[local_idx]).reshape(-1)
                if max_points > 0:
                    row_time = row_time[:max_points]
                    row_time_end = row_time_end[:max_points]
                valid = np.isfinite(row_time) & np.isfinite(row_time_end)
                row_weight = None
                if weight_chunk is not None:
                    row_weight = np.asarray(weight_chunk[local_idx]).reshape(-1)
                    if max_points > 0:
                        row_weight = row_weight[:max_points]
                    valid &= np.isfinite(row_weight) & (row_weight > 0.0)
                if not np.any(valid):
                    continue
                start_valid = row_time[valid]
                end_valid = row_time_end[valid]
                duration_valid = end_valid - start_valid
                duration_mask = np.isfinite(duration_valid) & (duration_valid >= 0.0)
                duration_valid = duration_valid[duration_mask]
                weights_valid = np.ones_like(start_valid, dtype=np.float64) if row_weight is None else row_weight[valid]
                edges, binned = _build_interval_waveform(
                    start_valid,
                    end_valid,
                    weights_valid,
                    bin_width=waveform_bin_width,
                )
                bin_centers = 0.5 * (edges[:-1] + edges[1:]) if edges.size >= 2 else np.asarray([], dtype=np.float64)
                if bin_centers.size > 0 and binned.size == bin_centers.size and np.any(np.isfinite(binned) & (binned > 0.0)):
                    total_binned = float(np.sum(binned))
                    if total_binned > 0.0:
                        late_fraction = float(np.sum(binned[bin_centers > late_time_threshold]) / total_binned)
                    else:
                        late_fraction = math.nan
                    waveform_quantiles: Dict[str, float] = {}
                    for q in quantiles:
                        key = f"q{int(round(q * 100.0)):02d}"
                        waveform_quantiles[key] = _waveform_quantile_from_bins(edges, binned, q)
                    if duration_valid.size > 0:
                        if row_weight is None:
                            duration_quantiles = {
                                f"dur_q{int(round(q * 100.0)):02d}": float(np.quantile(duration_valid, q))
                                for q in quantiles
                            }
                        else:
                            duration_weights = weights_valid[duration_mask]
                            duration_quantiles = {}
                            for q in quantiles:
                                dur_key = f"dur_q{int(round(q * 100.0)):02d}"
                                if duration_weights.size > 0:
                                    duration_quantiles[dur_key] = _weighted_quantile(duration_valid, duration_weights, q)
                                else:
                                    duration_quantiles[dur_key] = math.nan
                    else:
                        duration_quantiles = {
                            f"dur_q{int(round(q * 100.0)):02d}": math.nan
                            for q in quantiles
                        }
                else:
                    continue
                for key, value in waveform_quantiles.items():
                    out[key].append(float(value))
                for key, value in duration_quantiles.items():
                    out[key].append(float(value))
                if energies is not None:
                    selected_energies.append(float(energies[idx]))
                if edeps is not None:
                    selected_edep.append(float(edeps[idx]))
                if eleaks is not None:
                    selected_eleak.append(float(eleaks[idx]))
                if c_amps is not None:
                    selected_c_amp.append(float(c_amps[idx]))
                if s_amps is not None:
                    selected_s_amp.append(float(s_amps[idx]))
                if thetas is not None:
                    selected_thetas.append(float(thetas[idx]))
                if phis is not None:
                    selected_phis.append(float(phis[idx]))
                selected_max_valid_point.append(float(np.count_nonzero(valid)))
                selected_late_fraction.append(float(late_fraction))
                event_iter.update(1)
        event_iter.close()

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in out.items()}
    if selected_energies:
        arrays["E_gen"] = np.asarray(selected_energies, dtype=np.float64)
    if selected_edep:
        arrays["E_dep"] = np.asarray(selected_edep, dtype=np.float64)
    if selected_eleak:
        arrays["E_leak"] = np.asarray(selected_eleak, dtype=np.float64)
    if selected_c_amp:
        arrays["C_amp"] = np.asarray(selected_c_amp, dtype=np.float64)
    if selected_s_amp:
        arrays["S_amp"] = np.asarray(selected_s_amp, dtype=np.float64)
    if selected_energies and selected_edep:
        egen = np.asarray(selected_energies, dtype=np.float64)
        edep = np.asarray(selected_edep, dtype=np.float64)
        ratio = np.full_like(edep, np.nan, dtype=np.float64)
        valid_ratio = np.isfinite(egen) & np.isfinite(edep) & (egen != 0.0)
        ratio[valid_ratio] = edep[valid_ratio] / egen[valid_ratio]
        arrays["E_dep_over_E_gen"] = ratio
    if selected_c_amp and selected_s_amp:
        c_amp = np.asarray(selected_c_amp, dtype=np.float64)
        s_amp = np.asarray(selected_s_amp, dtype=np.float64)
        ratio = np.full_like(c_amp, np.nan, dtype=np.float64)
        valid_ratio = np.isfinite(c_amp) & np.isfinite(s_amp) & (s_amp != 0.0)
        ratio[valid_ratio] = c_amp[valid_ratio] / s_amp[valid_ratio]
        arrays["C_amp_over_S_amp"] = ratio
    if selected_thetas:
        arrays["GenParticles.momentum.theta"] = np.asarray(selected_thetas, dtype=np.float64)
    if selected_phis:
        arrays["GenParticles.momentum.phi"] = np.asarray(selected_phis, dtype=np.float64)
    if selected_late_fraction:
        arrays["late_energy_fraction"] = np.asarray(selected_late_fraction, dtype=np.float64)
    if selected_max_valid_point:
        arrays["max_valid_point"] = np.asarray(selected_max_valid_point, dtype=np.float64)
    return arrays, take


def _merge_by_sample(
    files: Sequence[Path],
    *,
    time_key: str,
    time_end_key: str,
    weight_key: str | None,
    energy_key: str | None,
    edep_key: str | None,
    eleak_key: str | None,
    c_amp_key: str | None,
    s_amp_key: str | None,
    theta_key: str | None,
    phi_key: str | None,
    e_min: float | None,
    e_max: float | None,
    theta_min: float | None,
    theta_max: float | None,
    quantiles: Sequence[float],
    max_events: int,
    read_chunk_events: int,
    waveform_bin_width: float,
    max_points: int,
    late_time_threshold: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    file_iter = tqdm(files, desc="Files", unit="file")
    for path in file_iter:
        label = _sample_label(path)
        file_iter.set_postfix_str(label)
        qvals, _ = _collect_file_quantiles(
            path,
            time_key=time_key,
            time_end_key=time_end_key,
            weight_key=weight_key,
            energy_key=energy_key,
            edep_key=edep_key,
            eleak_key=eleak_key,
            c_amp_key=c_amp_key,
            s_amp_key=s_amp_key,
            theta_key=theta_key,
            phi_key=phi_key,
            e_min=e_min,
            e_max=e_max,
            theta_min=theta_min,
            theta_max=theta_max,
            quantiles=quantiles,
            max_events=max_events,
            read_chunk_events=read_chunk_events,
            waveform_bin_width=waveform_bin_width,
            max_points=max_points,
            late_time_threshold=late_time_threshold,
        )
        bucket = grouped.setdefault(label, {key: [] for key in qvals})
        for key, values in qvals.items():
            if values.size > 0:
                bucket[key].append(values)

    merged: Dict[str, Dict[str, np.ndarray]] = {}
    for label, parts in grouped.items():
        merged[label] = {}
        for key, arrays in parts.items():
            merged[label][key] = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float64)
        _add_quantile_differences(merged[label])
        _add_quantile_differences(merged[label], prefix="dur_")
    return merged


def _plot_histograms(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    quantile_keys: Sequence[str],
    spread_key: str | None,
    bins: int,
    density: bool,
) -> None:
    plot_keys = [key for key in quantile_keys if any(key in v for v in samples.values())]
    if spread_key is not None and any(spread_key in v for v in samples.values()):
        plot_keys.append(spread_key)
    for extra_key in ("q30_minus_q10", "q50_minus_q30", "q80_minus_q50", "q90_minus_q50", "q50_minus_q20"):
        if any(extra_key in v for v in samples.values()) and extra_key not in plot_keys:
            plot_keys.append(extra_key)
    if not plot_keys:
        return

    fig, axes = plt.subplots(len(plot_keys), 1, figsize=(10, 3.5 * len(plot_keys)), constrained_layout=True)
    if len(plot_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, plot_keys):
        any_drawn = False
        for label, series in samples.items():
            values = np.asarray(series.get(key, []), dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            any_drawn = True
            ax.hist(values, bins=bins, histtype="step", linewidth=2.0, density=density, label=label)
        ax.set_xlabel(key)
        ax.set_ylabel("density" if density else "count")
        ax.set_title(f"{key} distribution")
        ax.grid(True, alpha=0.25)
        if any_drawn:
            ax.legend()

    _save_figure(fig, out_dir / "timing_quantiles_hist.png")
    plt.close(fig)


def _plot_boxplots(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    quantile_keys: Sequence[str],
    spread_key: str | None,
) -> None:
    plot_keys = [key for key in quantile_keys if any(key in v for v in samples.values())]
    if spread_key is not None and any(spread_key in v for v in samples.values()):
        plot_keys.append(spread_key)
    for extra_key in ("q30_minus_q10", "q50_minus_q30", "q80_minus_q50", "q90_minus_q50", "q50_minus_q20"):
        if any(extra_key in v for v in samples.values()) and extra_key not in plot_keys:
            plot_keys.append(extra_key)
    if not plot_keys:
        return

    fig, axes = plt.subplots(1, len(plot_keys), figsize=(4.8 * len(plot_keys), 5.0), constrained_layout=True)
    if len(plot_keys) == 1:
        axes = [axes]

    labels = list(samples.keys())
    for ax, key in zip(axes, plot_keys):
        series = []
        valid_labels = []
        for label in labels:
            values = np.asarray(samples[label].get(key, []), dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            series.append(values)
            valid_labels.append(label)
        if not series:
            ax.axis("off")
            continue
        ax.boxplot(series, tick_labels=valid_labels, showfliers=False)
        ax.set_title(key)
        ax.set_ylabel("time")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)

    _save_figure(fig, out_dir / "timing_quantiles_boxplot.png")
    plt.close(fig)


def _plot_scatter(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    median_key: str,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, median_key, spread_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel(median_key)
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing median vs width")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_quantiles_scatter.png")
    plt.close(fig)


def _plot_q50_vs_energy(samples: Mapping[str, Mapping[str, np.ndarray]], *, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_gen", "q50")
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_gen")
    ax.set_ylabel("q50")
    ax.set_title("Timing median vs E_gen")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_q50_vs_egen.png")
    plt.close(fig)


def _plot_spread_vs_energy(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_gen", spread_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_gen")
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing width vs E_gen")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"{spread_key}_vs_Egen.png")
    plt.close(fig)


def _plot_q50_vs_edep(samples: Mapping[str, Mapping[str, np.ndarray]], *, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_dep", "q50")
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_dep")
    ax.set_ylabel("q50")
    ax.set_title("Timing median vs E_dep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_q50_vs_edep.png")
    plt.close(fig)


def _plot_q50_vs_eleak(samples: Mapping[str, Mapping[str, np.ndarray]], *, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_leak", "q50", positive_x=True)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_leak")
    ax.set_ylabel("q50")
    ax.set_title("Timing median vs E_leak")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_q50_vs_eleak.png")
    plt.close(fig)


def _plot_q50_vs_edep_over_egen(samples: Mapping[str, Mapping[str, np.ndarray]], *, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_dep_over_E_gen", "q50")
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_dep / E_gen")
    ax.set_ylabel("q50")
    ax.set_title("Timing median vs E_dep / E_gen")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_q50_vs_edep_over_egen.png")
    plt.close(fig)


def _plot_spread_vs_edep(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_dep", spread_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_dep")
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing width vs E_dep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"{spread_key}_vs_Edep.png")
    plt.close(fig)


def _plot_spread_vs_eleak(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_leak", spread_key, positive_x=True)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_leak")
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing width vs E_leak")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"{spread_key}_vs_Eleak.png")
    plt.close(fig)


def _plot_specific_width_vs_x(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    width_key: str,
    x_key: str,
    x_label: str,
    filename: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, x_key, width_key, positive_x=(x_key == "E_leak"))
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel(x_label)
    ax.set_ylabel(width_key.replace("_minus_", " - "))
    ax.set_title(title)
    if x_key == "E_leak":
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / filename)
    plt.close(fig)


def _plot_spread_vs_edep_over_egen(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "E_dep_over_E_gen", spread_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("E_dep / E_gen")
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing width vs E_dep / E_gen")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"{spread_key}_vs_Edep_over_Egen.png")
    plt.close(fig)


def _plot_late_fraction_vs_spread(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, spread_key, "late_energy_fraction")
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel(spread_key.replace("_minus_", " - "))
    ax.set_ylabel("late_energy_fraction")
    ax.set_title("Late energy fraction vs timing width")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"late_energy_fraction_vs_{spread_key}.png")
    plt.close(fig)


def _plot_band_split_diagnostics(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
    split_threshold: float,
) -> None:
    if spread_key is None:
        return
    metrics = [
        ("q50", "q50"),
        ("E_leak", "E_leak"),
        ("E_dep_over_E_gen", "E_dep / E_gen"),
        ("late_energy_fraction", "late_energy_fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    drew_any = False
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for label, values in samples.items():
            spread = np.asarray(values.get(spread_key, []), dtype=np.float64)
            metric = np.asarray(values.get(metric_key, []), dtype=np.float64)
            if spread.size == 0 or metric.size == 0:
                continue
            if spread.size != metric.size:
                n = min(spread.size, metric.size)
                spread = spread[:n]
                metric = metric[:n]
            valid = np.isfinite(spread) & np.isfinite(metric)
            if not np.any(valid):
                continue
            low = metric[valid & (spread < split_threshold)]
            high = metric[valid & (spread >= split_threshold)]
            if low.size > 0:
                drew_any = True
                ax.hist(low, bins=50, histtype="step", linewidth=1.8, alpha=0.85, label=f"{label} low")
            if high.size > 0:
                drew_any = True
                ax.hist(high, bins=50, histtype="stepfilled", alpha=0.18, label=f"{label} high")
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("count")
    if not drew_any:
        plt.close(fig)
        return
    axes[2].set_xlabel("value")
    axes[3].set_xlabel("value")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(f"Low/high band split at {spread_key.replace('_minus_', ' - ')} = {split_threshold:g}")
    _save_figure(fig, out_dir / f"{spread_key}_band_split.png")
    plt.close(fig)


def _plot_q50_amp_split(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    center: float,
    halfwidth: float,
) -> None:
    metrics = [
        ("S_amp", "S_amp"),
        ("C_amp", "C_amp"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    drew_any = False
    low_edge = center - halfwidth
    high_edge = center + halfwidth
    preferred_labels = [label for label in samples if ("pi+" in label or "kaon+" in label)]
    labels_to_draw = preferred_labels or list(samples.keys())
    for ax, (metric_key, label_key) in zip(axes, metrics):
        for label in labels_to_draw:
            values = samples[label]
            q50, metric = _paired_series(values, "q50", metric_key)
            if q50.size == 0:
                continue
            valid_metric = np.isfinite(metric) & (metric > 0.0)
            q50 = q50[valid_metric]
            metric = metric[valid_metric]
            if q50.size == 0:
                continue
            near = metric[np.abs(q50 - center) <= halfwidth]
            other = metric[np.abs(q50 - center) > halfwidth]
            if near.size > 0:
                drew_any = True
                ax.hist(near, bins=60, histtype="step", linewidth=1.8, alpha=0.9, label=f"{label} near")
            if other.size > 0:
                drew_any = True
                ax.hist(other, bins=60, histtype="stepfilled", alpha=0.16, label=f"{label} other")
        ax.set_title(label_key)
        ax.set_xlabel(label_key)
        ax.set_ylabel("count")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
    if not drew_any:
        plt.close(fig)
        return
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(f"{low_edge:g} <= q50 <= {high_edge:g} vs others")
    _save_figure(fig, out_dir / "q50_band_amp_split.png")
    plt.close(fig)


def _plot_q50_vs_camp_over_samp(samples: Mapping[str, Mapping[str, np.ndarray]], *, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "C_amp_over_S_amp", "q50")
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("C_amp / S_amp")
    ax.set_ylabel("q50")
    ax.set_title("Timing median vs C_amp / S_amp")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_q50_vs_camp_over_samp.png")
    plt.close(fig)


def _plot_spread_vs_camp_over_samp(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "C_amp_over_S_amp", spread_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("C_amp / S_amp")
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing width vs C_amp / S_amp")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"{spread_key}_vs_Camp_over_Samp.png")
    plt.close(fig)


def _plot_q50_vs_theta(samples: Mapping[str, Mapping[str, np.ndarray]], *, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "GenParticles.momentum.theta", "q50")
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("GenParticles.momentum.theta")
    ax.set_ylabel("q50")
    ax.set_title("Timing median vs theta")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_q50_vs_theta.png")
    plt.close(fig)


def _plot_q50_vs_max_valid_point(samples: Mapping[str, Mapping[str, np.ndarray]], *, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "max_valid_point", "q50")
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("max_valid_point")
    ax.set_ylabel("q50")
    ax.set_title("Timing median vs maximum valid point")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / "timing_q50_vs_max_valid_point.png")
    plt.close(fig)


def _plot_spread_vs_theta(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "GenParticles.momentum.theta", spread_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("GenParticles.momentum.theta")
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing width vs theta")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"{spread_key}_vs_theta.png")
    plt.close(fig)


def _plot_spread_vs_phi(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    spread_key: str | None,
) -> None:
    if spread_key is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, "GenParticles.momentum.phi", spread_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("GenParticles.momentum.phi")
    ax.set_ylabel(spread_key.replace("_minus_", " - "))
    ax.set_title("Timing width vs phi")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / f"{spread_key}_vs_phi.png")
    plt.close(fig)


def _plot_width_vs_width(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    out_dir: Path,
    x_key: str,
    y_key: str,
    filename: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    drew = False
    for label, values in samples.items():
        x, y = _paired_series(values, x_key, y_key)
        if x.size == 0:
            continue
        drew = True
        ax.scatter(x, y, s=10, alpha=0.35, label=label)
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel(x_key.replace("_minus_", " - "))
    ax.set_ylabel(y_key.replace("_minus_", " - "))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, out_dir / filename)
    plt.close(fig)


def _build_summary(
    samples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    time_key: str,
    time_end_key: str,
    weight_key: str | None,
    energy_key: str | None,
    edep_key: str | None,
    eleak_key: str | None,
    c_amp_key: str | None,
    s_amp_key: str | None,
    theta_key: str | None,
    e_min: float | None,
    e_max: float | None,
    theta_min: float | None,
    theta_max: float | None,
    quantiles: Sequence[float],
    late_time_threshold: float,
    band_split_threshold: float,
) -> Dict[str, object]:
    per_sample: Dict[str, Dict[str, object]] = {}
    for label, values in samples.items():
        per_sample[label] = {
            key: _safe_summary(np.asarray(arr, dtype=np.float64))
            for key, arr in values.items()
        }
    return {
        "time_key": time_key,
        "time_end_key": time_end_key,
        "weight_key": weight_key,
        "energy_key": energy_key,
        "edep_key": edep_key,
        "eleak_key": eleak_key,
        "c_amp_key": c_amp_key,
        "s_amp_key": s_amp_key,
        "theta_key": theta_key,
        "e_min": e_min,
        "e_max": e_max,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "quantiles": list(quantiles),
        "late_time_threshold": late_time_threshold,
        "band_split_threshold": band_split_threshold,
        "samples": per_sample,
    }


def _build_interval_waveform(
    starts: np.ndarray,
    ends: np.ndarray,
    amplitudes: np.ndarray,
    *,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(amplitudes)
    finite &= amplitudes > 0.0
    if not np.any(finite):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    starts = np.asarray(starts[finite], dtype=np.float64)
    ends = np.asarray(ends[finite], dtype=np.float64)
    amplitudes = np.asarray(amplitudes[finite], dtype=np.float64)
    swap = ends < starts
    if np.any(swap):
        starts[swap], ends[swap] = ends[swap], starts[swap]
    t_min = float(np.min(starts))
    t_max = float(np.max(ends))
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if t_max <= t_min:
        t_max = t_min + max(bin_width, 1.0)
    width = max(float(bin_width), 1e-6)
    lo = math.floor(t_min / width) * width
    hi = math.ceil(t_max / width) * width
    if hi <= lo:
        hi = lo + width
    edges = np.arange(lo, hi + width, width, dtype=np.float64)
    if edges.size < 2:
        edges = np.asarray([lo, lo + width], dtype=np.float64)
    n_bins = edges.size - 1
    binned = np.zeros(n_bins, dtype=np.float64)
    full_bin_diff = np.zeros(n_bins + 1, dtype=np.float64)
    end_adj = np.nextafter(ends, starts)
    left_bins = np.floor((starts - lo) / width).astype(np.int64)
    right_bins = np.floor((end_adj - lo) / width).astype(np.int64)
    left_bins = np.clip(left_bins, 0, n_bins - 1)
    right_bins = np.clip(right_bins, 0, n_bins - 1)
    durations = ends - starts
    positive = durations > 0.0
    point_like = ~positive
    if np.any(point_like):
        np.add.at(binned, left_bins[point_like], amplitudes[point_like])
    if np.any(positive):
        p_starts = starts[positive]
        p_ends = ends[positive]
        p_amps = amplitudes[positive]
        p_left = left_bins[positive]
        p_right = right_bins[positive]
        p_durations = durations[positive]
        density = p_amps / p_durations
        same_bin = p_left == p_right
        if np.any(same_bin):
            np.add.at(binned, p_left[same_bin], p_amps[same_bin])
        span = ~same_bin
        if np.any(span):
            s_left = p_left[span]
            s_right = p_right[span]
            s_density = density[span]
            s_starts = p_starts[span]
            s_ends = p_ends[span]
            left_overlap = edges[s_left + 1] - s_starts
            right_overlap = s_ends - edges[s_right]
            np.add.at(binned, s_left, s_density * left_overlap)
            np.add.at(binned, s_right, s_density * right_overlap)
            full_mask = s_right > (s_left + 1)
            if np.any(full_mask):
                full_left = s_left[full_mask] + 1
                full_right = s_right[full_mask]
                full_value = s_density[full_mask] * width
                np.add.at(full_bin_diff, full_left, full_value)
                np.add.at(full_bin_diff, full_right, -full_value)
    if np.any(full_bin_diff[:-1] != 0.0):
        binned += np.cumsum(full_bin_diff[:-1], dtype=np.float64)
    return edges, binned


def _waveform_quantile_from_bins(edges: np.ndarray, binned: np.ndarray, quantile: float) -> float:
    if edges.size < 2 or binned.size == 0:
        return math.nan
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.asarray(binned, dtype=np.float64)
    valid = np.isfinite(centers) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return math.nan
    return _weighted_quantile(centers[valid], weights[valid], quantile)


def _plot_q50_band_waveforms(
    files: Sequence[Path],
    *,
    out_dir: Path,
    time_key: str,
    time_end_key: str,
    weight_key: str | None,
    energy_key: str | None,
    theta_key: str | None,
    e_min: float | None,
    e_max: float | None,
    theta_min: float | None,
    theta_max: float | None,
    max_events: int,
    read_chunk_events: int,
    max_points: int,
    q50_center: float,
    q50_halfwidth: float,
    bin_width: float,
    max_intervals: int,
    max_examples: int,
) -> None:
    if max_examples <= 0:
        return
    selections: Dict[str, List[dict[str, object]]] = {}
    file_iter = tqdm(files, desc="Waveform scan", unit="file", leave=False)
    for path in file_iter:
        label = _sample_label(path)
        picked = selections.setdefault(label, [])
        if len(picked) >= max_examples:
            continue
        file_iter.set_postfix_str(label)
        with h5py.File(path, "r") as handle:
            if time_key not in handle or time_end_key not in handle:
                continue
            time_ds = handle[time_key]
            time_end_ds = handle[time_end_key]
            if time_ds.ndim < 2 or time_ds.shape != time_end_ds.shape:
                continue
            n_events = int(time_ds.shape[0])
            take = n_events if max_events <= 0 else min(n_events, max_events)
            event_mask = np.ones(take, dtype=bool)
            energies = _read_event_vector(handle, energy_key, take, time_key) if energy_key is not None and energy_key in handle else None
            if energy_key is not None and energies is None:
                event_mask[:] = False
            if energies is not None:
                event_mask &= np.isfinite(energies)
                if e_min is not None:
                    event_mask &= energies >= e_min
                if e_max is not None:
                    event_mask &= energies <= e_max
            thetas = _read_event_vector(handle, theta_key, take, time_key) if theta_key is not None and theta_key in handle else None
            if theta_key is not None and thetas is None:
                event_mask[:] = False
            if thetas is not None:
                event_mask &= np.isfinite(thetas)
                if theta_min is not None:
                    event_mask &= thetas >= theta_min
                if theta_max is not None:
                    event_mask &= thetas <= theta_max

            chunk_size = max(1, int(read_chunk_events))
            best: dict[str, object] | None = None
            selected_indices = np.flatnonzero(event_mask)
            for run_start, run_stop in _iter_contiguous_runs(selected_indices, chunk_size):
                times = np.asarray(time_ds[run_start:run_stop], dtype=np.float64)
                time_ends = np.asarray(time_end_ds[run_start:run_stop], dtype=np.float64)
                weight_chunk = None
                if weight_key is not None:
                    if weight_key not in handle:
                        best = None
                        break
                    weight_chunk = np.asarray(handle[weight_key][run_start:run_stop], dtype=np.float64)
                    if weight_chunk.shape != times.shape:
                        best = None
                        break

                for local_idx, idx in enumerate(range(run_start, run_stop)):
                    row_time = np.asarray(times[local_idx]).reshape(-1)
                    row_time_end = np.asarray(time_ends[local_idx]).reshape(-1)
                    if max_points > 0:
                        row_time = row_time[:max_points]
                        row_time_end = row_time_end[:max_points]
                    valid = np.isfinite(row_time) & np.isfinite(row_time_end)
                    row_weight = None
                    if weight_chunk is not None:
                        row_weight = np.asarray(weight_chunk[local_idx]).reshape(-1)
                        if max_points > 0:
                            row_weight = row_weight[:max_points]
                        valid &= np.isfinite(row_weight) & (row_weight > 0.0)
                    if not np.any(valid):
                        continue
                    starts = row_time[valid]
                    ends = row_time_end[valid]
                    centers = 0.5 * (starts + ends)
                    amps = np.ones_like(centers, dtype=np.float64) if row_weight is None else row_weight[valid]
                    hit_center_q50 = _weighted_quantile(centers, amps, 0.5) if row_weight is not None else float(np.quantile(centers, 0.5))
                    edges, binned = _build_interval_waveform(starts, ends, amps, bin_width=bin_width)
                    q50 = _waveform_quantile_from_bins(edges, binned, 0.5)
                    if not np.isfinite(q50):
                        continue
                    delta = abs(q50 - q50_center)
                    if delta > q50_halfwidth:
                        continue
                    candidate = {
                        "file": str(path),
                        "label": label,
                        "event_index": int(idx),
                        "q50": float(q50),
                        "hit_center_q50": float(hit_center_q50) if np.isfinite(hit_center_q50) else math.nan,
                        "delta_q50": float(delta),
                        "starts": starts.copy(),
                        "ends": ends.copy(),
                        "amps": amps.copy(),
                    }
                    if best is None or float(candidate["delta_q50"]) < float(best["delta_q50"]):
                        best = candidate
        if best is not None:
            picked.append(best)

    saved: List[dict[str, object]] = []
    for label, examples in selections.items():
        for iex, example in enumerate(examples):
            starts = np.asarray(example["starts"], dtype=np.float64)
            ends = np.asarray(example["ends"], dtype=np.float64)
            amps = np.asarray(example["amps"], dtype=np.float64)
            edges, binned = _build_interval_waveform(starts, ends, amps, bin_width=bin_width)
            if edges.size < 2 or binned.size == 0:
                continue
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            waveform_q50 = _waveform_quantile_from_bins(edges, binned, 0.5)
            peak_time = float(bin_centers[int(np.nanargmax(binned))]) if np.any(np.isfinite(binned)) else math.nan
            fig, (ax0, ax1) = plt.subplots(
                2,
                1,
                figsize=(10, 7),
                constrained_layout=True,
                gridspec_kw={"height_ratios": [2.2, 1.5]},
            )
            ax0.step(edges[:-1], binned, where="post", linewidth=1.8, color="tab:blue")
            ax0.fill_between(edges[:-1], binned, step="post", alpha=0.18, color="tab:blue")
            ax0.axvline(
                float(example["q50"]),
                color="tab:red",
                linestyle="--",
                linewidth=1.4,
                label=f"waveform q50={float(example['q50']):.2f}",
            )
            hit_center_q50 = float(example.get("hit_center_q50", math.nan))
            if np.isfinite(hit_center_q50):
                ax0.axvline(
                    float(hit_center_q50),
                    color="tab:green",
                    linestyle="-.",
                    linewidth=1.4,
                    label=f"hit-center q50={float(hit_center_q50):.2f}",
                )
            if np.isfinite(waveform_q50):
                ax0.axvline(
                    float(waveform_q50),
                    color="tab:olive",
                    linestyle=(0, (1, 2)),
                    linewidth=1.2,
                    label=f"recomputed waveform q50={float(waveform_q50):.2f}",
                )
            if np.isfinite(peak_time):
                ax0.axvline(
                    float(peak_time),
                    color="tab:purple",
                    linestyle=":",
                    linewidth=1.3,
                    label=f"waveform peak={float(peak_time):.2f}",
                )
            ax0.axvspan(q50_center - q50_halfwidth, q50_center + q50_halfwidth, color="tab:red", alpha=0.08, label="target band")
            ax0.set_ylabel("binned amplitude")
            ax0.set_title(f"{label} event {int(example['event_index'])} from {Path(str(example['file'])).name}")
            ax0.grid(True, alpha=0.25)
            ax0.legend()

            order = np.argsort(amps)[::-1]
            top = order[: max(1, min(int(max_intervals), order.size))]
            for rank, hit_idx in enumerate(top):
                y = float(amps[hit_idx])
                ax1.hlines(y, float(starts[hit_idx]), float(ends[hit_idx]), color="tab:gray", linewidth=1.2, alpha=0.75)
                ax1.plot(0.5 * (starts[hit_idx] + ends[hit_idx]), y, marker="o", markersize=2.6, color="tab:orange", alpha=0.85)
            ax1.set_xlabel("time")
            ax1.set_ylabel("hit amplitude")
            ax1.set_title(f"Top {top.size} hit intervals")
            if np.all(amps[top] > 0.0):
                ax1.set_yscale("log")
            ax1.grid(True, alpha=0.25)

            filename = f"q50_band_waveform_{_safe_label_slug(label)}_{iex:02d}.png"
            out_path = out_dir / filename
            _save_figure(fig, out_path)
            plt.close(fig)
            saved.append(
                {
                    "label": label,
                    "file": str(example["file"]),
                    "event_index": int(example["event_index"]),
                    "q50": float(example["q50"]),
                    "hit_center_q50": float(hit_center_q50) if np.isfinite(hit_center_q50) else None,
                    "waveform_q50": float(waveform_q50) if np.isfinite(waveform_q50) else None,
                    "waveform_peak_time": float(peak_time) if np.isfinite(peak_time) else None,
                    "delta_q50": float(example["delta_q50"]),
                    "plot": str(out_path),
                }
            )
    if saved:
        summary_path = out_dir / "q50_band_waveform_examples.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(saved, handle, indent=2)
        print(f"Saved plot summary: {summary_path}")


def main() -> int:
    args = parse_args()
    time_key = _canonical_key(args.time_key)
    time_end_key = _canonical_key(args.time_end_key)
    weight_key = _canonical_key(args.weight_key)
    quantiles = [float(q) for q in args.quantiles]
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantiles must be in [0, 1], got {q}.")

    files = _resolve_paths(args.files)
    if not files:
        raise FileNotFoundError("No input HDF5 files matched --files.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quantile_keys = [_quantile_key(q) for q in quantiles]
    spread_key = _preferred_spread_key(quantile_keys)
    median_key = _median_like_key(quantiles)

    if args.waveform_only:
        _plot_q50_band_waveforms(
            files,
            out_dir=out_dir,
            time_key=time_key or "DRcalo3dHits.time",
            time_end_key=time_end_key or "DRcalo3dHits.time_end",
            weight_key=weight_key,
            energy_key=args.energy_key,
            theta_key=args.theta_key,
            e_min=args.e_min,
            e_max=args.e_max,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            max_events=args.max_events,
            read_chunk_events=args.read_chunk_events,
            max_points=args.max_points,
            q50_center=args.q50_band_center,
            q50_halfwidth=args.q50_band_halfwidth,
            bin_width=args.waveform_bin_width,
            max_intervals=args.waveform_max_intervals,
            max_examples=args.waveform_max_examples,
        )
        return 0

    samples = _merge_by_sample(
        files,
        time_key=time_key or "DRcalo3dHits.time",
        time_end_key=time_end_key or "DRcalo3dHits.time_end",
        weight_key=weight_key,
        energy_key=args.energy_key,
        edep_key=args.edep_key,
        eleak_key=args.eleak_key,
        c_amp_key=args.c_amp_key,
        s_amp_key=args.s_amp_key,
        theta_key=args.theta_key,
        phi_key=args.phi_key,
        e_min=args.e_min,
        e_max=args.e_max,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        quantiles=quantiles,
        max_events=args.max_events,
        read_chunk_events=args.read_chunk_events,
        waveform_bin_width=args.waveform_bin_width,
        max_points=args.max_points,
        late_time_threshold=args.late_time_threshold,
    )
    if not samples:
        raise RuntimeError("No timing quantiles were computed from the selected files.")

    duration_dir = out_dir / "duration"
    duration_dir.mkdir(parents=True, exist_ok=True)
    duration_samples = _extract_prefixed_view(samples, "dur_")

    _plot_histograms(
        samples,
        out_dir=out_dir,
        quantile_keys=quantile_keys,
        spread_key=spread_key,
        bins=args.bins,
        density=args.density,
    )
    _plot_boxplots(samples, out_dir=out_dir, quantile_keys=quantile_keys, spread_key=spread_key)
    _plot_scatter(samples, out_dir=out_dir, median_key=median_key, spread_key=spread_key)
    _plot_q50_vs_energy(samples, out_dir=out_dir)
    _plot_spread_vs_energy(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_q50_vs_edep(samples, out_dir=out_dir)
    _plot_spread_vs_edep(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_q50_vs_eleak(samples, out_dir=out_dir)
    _plot_spread_vs_eleak(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q90_minus_q50",
        x_key="E_gen",
        x_label="E_gen",
        filename="q90_minus_q50_vs_Egen.png",
        title="q90 - q50 vs E_gen",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q90_minus_q50",
        x_key="E_leak",
        x_label="E_leak",
        filename="q90_minus_q50_vs_Eleak.png",
        title="q90 - q50 vs E_leak",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q90_minus_q50",
        x_key="E_dep",
        x_label="E_dep",
        filename="q90_minus_q50_vs_Edep.png",
        title="q90 - q50 vs E_dep",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q90_minus_q50",
        x_key="E_dep_over_E_gen",
        x_label="E_dep / E_gen",
        filename="q90_minus_q50_vs_Edep_over_Egen.png",
        title="q90 - q50 vs E_dep / E_gen",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q90_minus_q50",
        x_key="C_amp_over_S_amp",
        x_label="C_amp / S_amp",
        filename="q90_minus_q50_vs_Camp_over_Samp.png",
        title="q90 - q50 vs C_amp / S_amp",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q90_minus_q50",
        x_key="GenParticles.momentum.theta",
        x_label="GenParticles.momentum.theta",
        filename="q90_minus_q50_vs_theta.png",
        title="q90 - q50 vs theta",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q90_minus_q50",
        x_key="GenParticles.momentum.phi",
        x_label="GenParticles.momentum.phi",
        filename="q90_minus_q50_vs_phi.png",
        title="q90 - q50 vs phi",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q50_minus_q30",
        x_key="E_dep_over_E_gen",
        x_label="E_dep / E_gen",
        filename="q50_minus_q30_vs_Edep_over_Egen.png",
        title="q50 - q30 vs E_dep / E_gen",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q80_minus_q50",
        x_key="E_dep_over_E_gen",
        x_label="E_dep / E_gen",
        filename="q80_minus_q50_vs_Edep_over_Egen.png",
        title="q80 - q50 vs E_dep / E_gen",
    )
    _plot_q50_vs_edep_over_egen(samples, out_dir=out_dir)
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q30",
        x_key="E_dep_over_E_gen",
        x_label="E_dep / E_gen",
        filename="q30_vs_Edep_over_Egen.png",
        title="q30 vs E_dep / E_gen",
    )
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="q80",
        x_key="E_dep_over_E_gen",
        x_label="E_dep / E_gen",
        filename="q80_vs_Edep_over_Egen.png",
        title="q80 vs E_dep / E_gen",
    )
    _plot_spread_vs_edep_over_egen(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_q50_vs_camp_over_samp(samples, out_dir=out_dir)
    _plot_spread_vs_camp_over_samp(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_q50_vs_theta(samples, out_dir=out_dir)
    _plot_q50_vs_max_valid_point(samples, out_dir=out_dir)
    _plot_spread_vs_theta(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_spread_vs_phi(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_q50_amp_split(
        samples,
        out_dir=out_dir,
        center=args.q50_band_center,
        halfwidth=args.q50_band_halfwidth,
    )
    if args.plot_q50_band_waveform:
        _plot_q50_band_waveforms(
            files,
            out_dir=out_dir,
            time_key=time_key or "DRcalo3dHits.time",
            time_end_key=time_end_key or "DRcalo3dHits.time_end",
            weight_key=weight_key,
            energy_key=args.energy_key,
            theta_key=args.theta_key,
            e_min=args.e_min,
            e_max=args.e_max,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            max_events=args.max_events,
            read_chunk_events=args.read_chunk_events,
            max_points=args.max_points,
            q50_center=args.q50_band_center,
            q50_halfwidth=args.q50_band_halfwidth,
            bin_width=args.waveform_bin_width,
            max_intervals=args.waveform_max_intervals,
            max_examples=args.waveform_max_examples,
        )
    _plot_width_vs_width(
        samples,
        out_dir=out_dir,
        x_key="q90_minus_q10",
        y_key="q90_minus_q50",
        filename="q90_minus_q10_vs_q90_minus_q50.png",
        title="q90 - q10 vs q90 - q50",
    )
    _plot_late_fraction_vs_spread(samples, out_dir=out_dir, spread_key=spread_key)
    _plot_specific_width_vs_x(
        samples,
        out_dir=out_dir,
        width_key="late_energy_fraction",
        x_key="q90_minus_q50",
        x_label="q90 - q50",
        filename="late_energy_fraction_vs_q90_minus_q50.png",
        title="Late energy fraction vs q90 - q50",
    )
    _plot_band_split_diagnostics(
        samples,
        out_dir=out_dir,
        spread_key=spread_key,
        split_threshold=args.band_split_threshold,
    )

    if duration_samples:
        _plot_histograms(
            duration_samples,
            out_dir=duration_dir,
            quantile_keys=quantile_keys,
            spread_key=spread_key,
            bins=args.bins,
            density=args.density,
        )
        _plot_boxplots(duration_samples, out_dir=duration_dir, quantile_keys=quantile_keys, spread_key=spread_key)
        _plot_scatter(duration_samples, out_dir=duration_dir, median_key=median_key, spread_key=spread_key)
        _plot_q50_vs_energy(duration_samples, out_dir=duration_dir)
        _plot_spread_vs_energy(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_q50_vs_edep(duration_samples, out_dir=duration_dir)
        _plot_spread_vs_edep(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_q50_vs_eleak(duration_samples, out_dir=duration_dir)
        _plot_spread_vs_eleak(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q90_minus_q50",
            x_key="E_gen",
            x_label="E_gen",
            filename="q90_minus_q50_vs_Egen.png",
            title="duration q90 - q50 vs E_gen",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q90_minus_q50",
            x_key="E_leak",
            x_label="E_leak",
            filename="q90_minus_q50_vs_Eleak.png",
            title="duration q90 - q50 vs E_leak",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q90_minus_q50",
            x_key="E_dep",
            x_label="E_dep",
            filename="q90_minus_q50_vs_Edep.png",
            title="duration q90 - q50 vs E_dep",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q90_minus_q50",
            x_key="E_dep_over_E_gen",
            x_label="E_dep / E_gen",
            filename="q90_minus_q50_vs_Edep_over_Egen.png",
            title="duration q90 - q50 vs E_dep / E_gen",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q90_minus_q50",
            x_key="C_amp_over_S_amp",
            x_label="C_amp / S_amp",
            filename="q90_minus_q50_vs_Camp_over_Samp.png",
            title="duration q90 - q50 vs C_amp / S_amp",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q90_minus_q50",
            x_key="GenParticles.momentum.theta",
            x_label="GenParticles.momentum.theta",
            filename="q90_minus_q50_vs_theta.png",
            title="duration q90 - q50 vs theta",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q90_minus_q50",
            x_key="GenParticles.momentum.phi",
            x_label="GenParticles.momentum.phi",
            filename="q90_minus_q50_vs_phi.png",
            title="duration q90 - q50 vs phi",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q50_minus_q30",
            x_key="E_dep_over_E_gen",
            x_label="E_dep / E_gen",
            filename="q50_minus_q30_vs_Edep_over_Egen.png",
            title="duration q50 - q30 vs E_dep / E_gen",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q80_minus_q50",
            x_key="E_dep_over_E_gen",
            x_label="E_dep / E_gen",
            filename="q80_minus_q50_vs_Edep_over_Egen.png",
            title="duration q80 - q50 vs E_dep / E_gen",
        )
        _plot_q50_vs_edep_over_egen(duration_samples, out_dir=duration_dir)
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q30",
            x_key="E_dep_over_E_gen",
            x_label="E_dep / E_gen",
            filename="q30_vs_Edep_over_Egen.png",
            title="duration q30 vs E_dep / E_gen",
        )
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="q80",
            x_key="E_dep_over_E_gen",
            x_label="E_dep / E_gen",
            filename="q80_vs_Edep_over_Egen.png",
            title="duration q80 vs E_dep / E_gen",
        )
        _plot_spread_vs_edep_over_egen(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_q50_vs_camp_over_samp(duration_samples, out_dir=duration_dir)
        _plot_spread_vs_camp_over_samp(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_q50_vs_theta(duration_samples, out_dir=duration_dir)
        _plot_q50_vs_max_valid_point(duration_samples, out_dir=duration_dir)
        _plot_spread_vs_theta(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_spread_vs_phi(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_q50_amp_split(
            duration_samples,
            out_dir=duration_dir,
            center=args.q50_band_center,
            halfwidth=args.q50_band_halfwidth,
        )
        _plot_width_vs_width(
            duration_samples,
            out_dir=duration_dir,
            x_key="q90_minus_q10",
            y_key="q90_minus_q50",
            filename="q90_minus_q10_vs_q90_minus_q50.png",
            title="duration q90 - q10 vs q90 - q50",
        )
        _plot_late_fraction_vs_spread(duration_samples, out_dir=duration_dir, spread_key=spread_key)
        _plot_specific_width_vs_x(
            duration_samples,
            out_dir=duration_dir,
            width_key="late_energy_fraction",
            x_key="q90_minus_q50",
            x_label="q90 - q50",
            filename="late_energy_fraction_vs_q90_minus_q50.png",
            title="Duration late energy fraction vs q90 - q50",
        )
        _plot_band_split_diagnostics(
            duration_samples,
            out_dir=duration_dir,
            spread_key=spread_key,
            split_threshold=args.band_split_threshold,
        )

    summary = _build_summary(
        samples,
        time_key=time_key or "DRcalo3dHits.time",
        time_end_key=time_end_key or "DRcalo3dHits.time_end",
        weight_key=weight_key,
        energy_key=args.energy_key,
        edep_key=args.edep_key,
        eleak_key=args.eleak_key,
        c_amp_key=args.c_amp_key,
        s_amp_key=args.s_amp_key,
        theta_key=args.theta_key,
        e_min=args.e_min,
        e_max=args.e_max,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        quantiles=quantiles,
        late_time_threshold=args.late_time_threshold,
        band_split_threshold=args.band_split_threshold,
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
