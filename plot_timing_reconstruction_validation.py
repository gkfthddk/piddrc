#!/usr/bin/env python
"""Publication-style timing reconstruction validation plots.

The figure is intentionally detector/timing focused: event-level timing residuals,
resolution versus photon count, and a representative waveform before/after Wiener
deconvolution. It does not consume model outputs or report ML metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import matplotlib
import numpy as np
import uproot
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_common import AMP_KEY, TIME_END_KEY, TIME_KEY, default_sample_label, resolve_paths
from utils.plot_helpers import set_publication_style

RAW_AMP_BRANCH = "RawCalorimeterHits/RawCalorimeterHits.amplitude"
RAW_TIME_BRANCH = "RawCalorimeterHits/RawCalorimeterHits.timeStamp"
GEN_PX_BRANCH = "GenParticles/GenParticles.momentum.x"
GEN_PY_BRANCH = "GenParticles/GenParticles.momentum.y"
GEN_PZ_BRANCH = "GenParticles/GenParticles.momentum.z"


LABEL_MAP = {
    "gamma": r"$\gamma$",
    "pi0": r"$\pi^{0}$",
    "pi+": r"$\pi^{+}$",
    "e-": r"$e^{-}$",
    "electron": r"$e^{-}$",
}

COLORS = {
    "gamma": "#0072B2",
    "pi0": "#D55E00",
    "pi+": "#009E73",
    "e-": "#CC79A7",
    "electron": "#CC79A7",
}


@dataclass
class TimingSample:
    label: str
    reco_time: np.ndarray
    truth_time: np.ndarray
    photon_count: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        return self.reco_time - self.truth_time


@dataclass
class WaveformExample:
    label: str
    time: np.ndarray
    raw: np.ndarray
    deconvolved: np.ndarray
    truth_time: np.ndarray
    truth_photons: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a publication-quality timing reconstruction validation figure "
            "from event-level arrays, or derive timing observables from merged DR HDF5 files."
        )
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=[
            "h5s/gamma_1-120GeV.h5py",
            "h5s/pi0_1-120GeV.h5py",
            "h5s/pi+_1-120GeV.h5py",
            "h5s/e-_1-120GeV.h5py",
        ],
        help="Input HDF5/NPZ files, one per particle species.",
    )
    p.add_argument("--labels", nargs="+", default=["gamma", "pi0", "pi+", "e-"], help="Labels matching --files.")
    p.add_argument("--max-events", type=int, default=20000, help="Maximum events per file; <=0 scans all.")
    p.add_argument(
        "--sampling",
        choices=["head", "random"],
        default="head",
        help="Event sampling mode for large HDF5 files. 'head' uses contiguous reads; 'random' gives a random subset.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed used for deterministic subsampling.")
    p.add_argument("--min-photons", type=float, default=10.0, help="Minimum photon/count proxy for event inclusion.")
    p.add_argument("--time-key", default=TIME_KEY, help="Per-hit reconstructed timing key for HDF5 derivation.")
    p.add_argument("--time-end-key", default=TIME_END_KEY, help="Per-hit timing end key for HDF5 derivation.")
    p.add_argument("--amplitude-key", default=AMP_KEY, help="Per-hit photon-count/amplitude proxy key.")
    p.add_argument("--reco-event-key", default="reco_time", help="Event-level reconstructed time key if present.")
    p.add_argument("--truth-event-key", default="truth_time", help="Event-level truth photon-arrival time key if present.")
    p.add_argument("--photon-count-key", default="photon_count", help="Event-level photon-count key if present.")
    p.add_argument(
        "--truth-source",
        choices=["digi", "event", "h5-fallback"],
        default="digi",
        help="Source for truth timing/photon count. 'digi' matches ROOT digi files by seed and E_gen.",
    )
    p.add_argument(
        "--digi-base",
        type=Path,
        default=Path("/users/yulee/dream/tools/hdfs"),
        help="Base directory containing per-sample digi ROOT directories.",
    )
    p.add_argument(
        "--digi-sample-dirs",
        nargs="+",
        default=None,
        help="Optional digi directory names matching --files order. Defaults to each HDF5 stem.",
    )
    p.add_argument("--seed-key", default="seed", help="Merged HDF5 seed key used to locate the digi ROOT file.")
    p.add_argument("--energy-key", default="E_gen", help="Merged HDF5 generated-energy key used for event matching.")
    p.add_argument("--digi-time-branch", default=RAW_TIME_BRANCH, help="Digi ROOT MC truth photon-arrival time branch.")
    p.add_argument("--digi-photon-branch", default=RAW_AMP_BRANCH, help="Digi ROOT raw photon/count branch.")
    p.add_argument(
        "--photon-count-mode",
        choices=["raw-hit-count", "raw-amplitude-sum"],
        default="raw-amplitude-sum",
        help="Photon count observable derived from the digi raw branch.",
    )
    p.add_argument(
        "--energy-match-tolerance",
        type=float,
        default=1e-3,
        help="Maximum |E_gen(HDF5)-E_gen(digi)| in GeV for matching events with the same seed.",
    )
    p.add_argument(
        "--truth-mode",
        choices=["earliest", "weighted-mean"],
        default="weighted-mean",
        help=(
            "Truth-time estimator. For digi truth, 'weighted-mean' is the amplitude-weighted mean of raw photon "
            "arrival timestamps; 'earliest' uses the first positive raw photon timestamp."
        ),
    )
    p.add_argument("--bins", type=int, default=56, help="Residual histogram bins.")
    p.add_argument("--count-bins", type=int, default=9, help="Photon-count bins for resolution curve.")
    p.add_argument("--range-percentile", type=float, default=99.0, help="Central residual percentile shown.")
    p.add_argument("--waveform-bin-width", type=float, default=1.0, help="Waveform bin width in ns.")
    p.add_argument("--waveform-xmin", type=float, default=100.0, help="Lower x-limit for the waveform panel [ns].")
    p.add_argument("--waveform-xmax", type=float, default=300.0, help="Upper x-limit for the waveform panel [ns].")
    p.add_argument("--response-tau", type=float, default=5.0, help="Exponential response time constant for Wiener deconvolution [ns].")
    p.add_argument("--wiener-k", type=float, default=0.015, help="Wiener regularization noise-to-signal parameter.")
    p.add_argument(
        "--center-residuals",
        action="store_true",
        help="Subtract each sample's mean residual before plotting. Disabled by default to preserve absolute bias.",
    )
    p.add_argument(
        "--show-truth-photon-overlay",
        action="store_true",
        help="Overlay matched digi truth photon timing on the waveform panel.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("plots/timing_reconstruction_validation"), help="Output directory.")
    p.add_argument("--out-name", default="timing_reconstruction_validation", help="Output basename without extension.")
    p.add_argument("--summary-json", action="store_true", help="Also write numeric summary JSON.")
    return p.parse_args()


def _plain_label(label: str) -> str:
    return LABEL_MAP.get(label, label)


def _color(label: str) -> str | None:
    return COLORS.get(label)


def _finite_event_arrays(reco: np.ndarray, truth: np.ndarray, counts: np.ndarray, min_photons: float) -> TimingSample:
    n = min(reco.size, truth.size, counts.size)
    reco = np.asarray(reco[:n], dtype=np.float64)
    truth = np.asarray(truth[:n], dtype=np.float64)
    counts = np.asarray(counts[:n], dtype=np.float64)
    valid = np.isfinite(reco) & np.isfinite(truth) & np.isfinite(counts) & (counts >= float(min_photons))
    return TimingSample("", reco[valid], truth[valid], counts[valid])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return math.nan
    return float(np.average(values[good], weights=weights[good]))


def _read_event_level_h5(handle: h5py.File, args: argparse.Namespace, label: str) -> TimingSample | None:
    keys = [args.reco_event_key, args.truth_event_key, args.photon_count_key]
    if not all(key in handle for key in keys):
        return None
    sample = _finite_event_arrays(
        np.asarray(handle[args.reco_event_key][:]).reshape(-1),
        np.asarray(handle[args.truth_event_key][:]).reshape(-1),
        np.asarray(handle[args.photon_count_key][:]).reshape(-1),
        args.min_photons,
    )
    sample.label = label
    return sample


def _choose_indices(n_events: int, max_events: int, sampling: str, rng: np.random.Generator) -> np.ndarray:
    if max_events > 0 and n_events > max_events:
        if sampling == "random":
            return np.sort(rng.choice(n_events, size=max_events, replace=False))
        return np.arange(max_events, dtype=np.int64)
    return np.arange(n_events, dtype=np.int64)


def _derive_event_level_h5(handle: h5py.File, args: argparse.Namespace, label: str, rng: np.random.Generator) -> TimingSample:
    required = [args.time_key, args.time_end_key, args.amplitude_key]
    missing = [key for key in required if key not in handle]
    if missing:
        raise KeyError(f"{label}: missing required dataset(s): {', '.join(missing)}")

    n_events = int(handle[args.time_key].shape[0])
    indices = _choose_indices(n_events, int(args.max_events), str(args.sampling), rng)
    reco: list[float] = []
    truth: list[float] = []
    counts: list[float] = []

    for idx in tqdm(indices, desc=f"{label}", unit="event", leave=False):
        start = np.asarray(handle[args.time_key][idx], dtype=np.float64).reshape(-1)
        end = np.asarray(handle[args.time_end_key][idx], dtype=np.float64).reshape(-1)
        amp = np.asarray(handle[args.amplitude_key][idx], dtype=np.float64).reshape(-1)
        n = min(start.size, end.size, amp.size)
        if n == 0:
            continue
        start = start[:n]
        end = end[:n]
        amp = amp[:n]
        mid = 0.5 * (start + end)
        valid = np.isfinite(start) & np.isfinite(end) & np.isfinite(mid) & np.isfinite(amp) & (amp > 0.0)
        if not np.any(valid):
            continue
        start = start[valid]
        mid = mid[valid]
        amp = amp[valid]
        photon_proxy = float(np.sum(amp, dtype=np.float64))
        if not np.isfinite(photon_proxy) or photon_proxy < float(args.min_photons):
            continue
        reco_t = _weighted_mean(mid, amp)
        if args.truth_mode == "weighted-mean":
            truth_t = _weighted_mean(start, amp)
        else:
            truth_t = float(np.min(start))
        if np.isfinite(reco_t) and np.isfinite(truth_t):
            reco.append(reco_t)
            truth.append(truth_t)
            counts.append(photon_proxy)

    return TimingSample(label, np.asarray(reco), np.asarray(truth), np.asarray(counts))


def _event_reco_time_from_h5(handle: h5py.File, idx: int, args: argparse.Namespace) -> float:
    if args.reco_event_key in handle:
        value = np.asarray(handle[args.reco_event_key][idx], dtype=np.float64).reshape(-1)
        return float(value[0]) if value.size and np.isfinite(value[0]) else math.nan

    start = np.asarray(handle[args.time_key][idx], dtype=np.float64).reshape(-1)
    end = np.asarray(handle[args.time_end_key][idx], dtype=np.float64).reshape(-1)
    amp = np.asarray(handle[args.amplitude_key][idx], dtype=np.float64).reshape(-1)
    n = min(start.size, end.size, amp.size)
    if n == 0:
        return math.nan
    mid = 0.5 * (start[:n] + end[:n])
    return _weighted_mean(mid, amp[:n])


def _digi_path_for_seed(digi_base: Path, sample_dir: str, seed: int) -> Path | None:
    directory = digi_base.expanduser() / sample_dir
    direct = directory / f"digi_{sample_dir}_{seed}.root"
    if direct.is_file():
        return direct
    matches = sorted(directory.glob(f"digi_*_{seed}.root"))
    return matches[0] if matches else None


def _root_event_energies(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    px = arrays[GEN_PX_BRANCH]
    py = arrays[GEN_PY_BRANCH]
    pz = arrays[GEN_PZ_BRANCH]
    out = np.full(len(px), np.nan, dtype=np.float64)
    for i in range(len(px)):
        x = np.asarray(px[i], dtype=np.float64).reshape(-1)
        y = np.asarray(py[i], dtype=np.float64).reshape(-1)
        z = np.asarray(pz[i], dtype=np.float64).reshape(-1)
        if x.size and y.size and z.size:
            out[i] = float(np.sqrt(x[0] * x[0] + y[0] * y[0] + z[0] * z[0]))
    return out


def _digi_truth_time_and_count(raw_time: np.ndarray, raw_photon: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    times = np.asarray(raw_time, dtype=np.float64).reshape(-1)
    photons = np.asarray(raw_photon, dtype=np.float64).reshape(-1)
    n = min(times.size, photons.size)
    if n == 0:
        return math.nan, math.nan
    times = times[:n]
    photons = photons[:n]
    valid = np.isfinite(times) & np.isfinite(photons)
    if not np.any(valid):
        return math.nan, math.nan
    times = times[valid]
    photons = photons[valid]

    positive = photons > 0.0
    photon_count = float(np.sum(positive)) if args.photon_count_mode == "raw-hit-count" else float(np.sum(photons[positive]))
    if not np.isfinite(photon_count) or photon_count <= 0.0:
        return math.nan, math.nan

    if args.truth_mode == "weighted-mean" and np.any(positive):
        truth_time = float(np.average(times[positive], weights=photons[positive]))
    else:
        truth_time = float(np.min(times[positive])) if np.any(positive) else float(np.min(times))
    return truth_time, photon_count


def _load_digi_arrays(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branches = [
        GEN_PX_BRANCH,
        GEN_PY_BRANCH,
        GEN_PZ_BRANCH,
        args.digi_time_branch,
        args.digi_photon_branch,
    ]
    with uproot.open(path) as root_file:
        if "events" not in root_file:
            raise KeyError(f"Missing events tree in {path}")
        tree = root_file["events"]
        missing = [branch for branch in branches if branch not in tree]
        if missing:
            raise KeyError(f"{path}: missing branch(es): {', '.join(missing)}")
        arrays = tree.arrays(branches, library="np")
    energies = _root_event_energies(arrays)
    return energies, arrays[args.digi_time_branch], arrays[args.digi_photon_branch]


def _match_digi_entry(energies: np.ndarray, target_energy: float, used: set[int], tolerance: float) -> int | None:
    diff = np.abs(energies - float(target_energy))
    order = np.argsort(diff)
    for idx in order:
        i = int(idx)
        if i in used or not np.isfinite(diff[i]):
            continue
        if float(diff[i]) <= float(tolerance):
            used.add(i)
            return i
        break
    return None


def _derive_event_level_with_digi(
    h5_path: Path,
    args: argparse.Namespace,
    label: str,
    sample_dir: str,
    rng: np.random.Generator,
) -> TimingSample:
    with h5py.File(h5_path, "r") as handle:
        required = [args.seed_key, args.energy_key, args.time_key, args.time_end_key, args.amplitude_key]
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{label}: missing required HDF5 dataset(s): {', '.join(missing)}")

        n_events = int(handle[args.seed_key].shape[0])
        indices = _choose_indices(n_events, int(args.max_events), str(args.sampling), rng)
        seeds = np.asarray(handle[args.seed_key][indices], dtype=np.int64).reshape(-1)
        energies = np.asarray(handle[args.energy_key][indices], dtype=np.float64).reshape(-1)

        by_seed: dict[int, list[tuple[int, float]]] = {}
        for local_pos, (seed, energy) in enumerate(zip(seeds, energies)):
            if np.isfinite(energy):
                by_seed.setdefault(int(seed), []).append((int(indices[local_pos]), float(energy)))

        reco: list[float] = []
        truth: list[float] = []
        counts: list[float] = []
        missing_files = 0
        unmatched = 0

        for seed, h5_events in tqdm(by_seed.items(), desc=f"{label} digi", unit="seed", leave=False):
            digi_path = _digi_path_for_seed(args.digi_base, sample_dir, seed)
            if digi_path is None:
                missing_files += len(h5_events)
                continue
            digi_energies, digi_times, digi_photons = _load_digi_arrays(digi_path, args)
            used_entries: set[int] = set()
            for h5_idx, energy in h5_events:
                digi_idx = _match_digi_entry(digi_energies, energy, used_entries, float(args.energy_match_tolerance))
                if digi_idx is None:
                    unmatched += 1
                    continue
                reco_t = _event_reco_time_from_h5(handle, h5_idx, args)
                truth_t, photon_count = _digi_truth_time_and_count(digi_times[digi_idx], digi_photons[digi_idx], args)
                if (
                    np.isfinite(reco_t)
                    and np.isfinite(truth_t)
                    and np.isfinite(photon_count)
                    and photon_count >= float(args.min_photons)
                ):
                    reco.append(reco_t)
                    truth.append(truth_t)
                    counts.append(photon_count)

    if missing_files or unmatched:
        tqdm.write(f"{label}: skipped {missing_files} events with missing digi files and {unmatched} unmatched E_gen entries.")
    return TimingSample(label, np.asarray(reco), np.asarray(truth), np.asarray(counts))


def _load_npz(path: Path, args: argparse.Namespace, label: str) -> TimingSample:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in (args.reco_event_key, args.truth_event_key, args.photon_count_key) if key not in data]
        if missing:
            raise KeyError(f"{path}: missing required NPZ array(s): {', '.join(missing)}")
        sample = _finite_event_arrays(
            np.asarray(data[args.reco_event_key]).reshape(-1),
            np.asarray(data[args.truth_event_key]).reshape(-1),
            np.asarray(data[args.photon_count_key]).reshape(-1),
            args.min_photons,
        )
    sample.label = label
    return sample


def load_sample(path: Path, args: argparse.Namespace, label: str, sample_dir: str, rng: np.random.Generator) -> TimingSample:
    if path.suffix == ".npz":
        return _load_npz(path, args, label)
    with h5py.File(path, "r") as handle:
        if args.truth_source == "event":
            explicit = _read_event_level_h5(handle, args, label)
            if explicit is not None:
                return explicit
            raise KeyError(f"{label}: requested --truth-source event but event-level timing keys are missing.")
        if args.truth_source == "h5-fallback":
            return _derive_event_level_h5(handle, args, label, rng)
    return _derive_event_level_with_digi(path, args, label, sample_dir, rng)


def _shared_residual_range(samples: Sequence[TimingSample], percentile: float) -> tuple[float, float]:
    residuals = [s.residual[np.isfinite(s.residual)] for s in samples if s.residual.size]
    if not residuals:
        return -1.0, 1.0
    values = np.concatenate(residuals)
    pct = float(np.clip(percentile, 0.0, 100.0))
    lo_q = 0.5 * (100.0 - pct)
    hi_q = 100.0 - lo_q
    lo, hi = np.percentile(values, [lo_q, hi_q])
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        center = float(np.nanmean(values)) if values.size else 0.0
        return center - 1.0, center + 1.0
    pad = 0.08 * span
    return float(lo - pad), float(hi + pad)


def _resolution_points(sample: TimingSample, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = sample.residual
    count = sample.photon_count
    valid = np.isfinite(residual) & np.isfinite(count) & (count > 0.0)
    residual = residual[valid]
    count = count[valid]
    if residual.size < 10:
        return np.asarray([]), np.asarray([]), np.asarray([])
    lo, hi = np.percentile(count, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(count))
        hi = float(np.max(count))
    if hi <= lo:
        return np.asarray([]), np.asarray([]), np.asarray([])
    edges = np.geomspace(max(lo, 1e-9), hi, int(n_bins) + 1)
    centers: list[float] = []
    rms: list[float] = []
    err: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (count >= left) & (count < right)
        values = residual[in_bin]
        if values.size < 20:
            continue
        sigma = float(np.std(values, ddof=0))
        centers.append(float(np.sqrt(left * right)))
        rms.append(sigma)
        err.append(sigma / math.sqrt(2.0 * max(values.size - 1, 1)))
    return np.asarray(centers), np.asarray(rms), np.asarray(err)


def _build_interval_waveform(starts: np.ndarray, ends: np.ndarray, amplitudes: np.ndarray, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(amplitudes) & (amplitudes > 0.0)
    if not np.any(finite):
        return np.asarray([]), np.asarray([])
    starts = starts[finite].astype(np.float64)
    ends = ends[finite].astype(np.float64)
    amplitudes = amplitudes[finite].astype(np.float64)
    swap = ends < starts
    if np.any(swap):
        starts[swap], ends[swap] = ends[swap], starts[swap]
    width = max(float(bin_width), 1e-6)
    lo = math.floor(float(np.min(starts)) / width) * width
    hi = math.ceil(float(np.max(ends)) / width) * width
    if hi <= lo:
        hi = lo + width
    edges = np.arange(lo, hi + width, width, dtype=np.float64)
    hist = np.zeros(edges.size - 1, dtype=np.float64)
    for s, e, a in zip(starts, ends, amplitudes):
        if e <= s:
            j = int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, hist.size - 1))
            hist[j] += a
            continue
        left = max(0, int(np.searchsorted(edges, s, side="right") - 1))
        right = min(hist.size - 1, int(np.searchsorted(edges, np.nextafter(e, s), side="right") - 1))
        density = a / (e - s)
        for j in range(left, right + 1):
            overlap = max(0.0, min(e, edges[j + 1]) - max(s, edges[j]))
            hist[j] += density * overlap
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def _wiener_deconvolve(signal: np.ndarray, dt: float, tau: float, k: float) -> np.ndarray:
    y = np.asarray(signal, dtype=np.float64)
    if y.size < 3 or not np.any(np.isfinite(y)):
        return y.copy()
    y = np.nan_to_num(y, copy=False)
    n = int(2 ** math.ceil(math.log2(max(8, 2 * y.size))))
    t = np.arange(n, dtype=np.float64) * max(dt, 1e-9)
    response = np.exp(-t / max(tau, 1e-9))
    response /= np.sum(response)
    y_fft = np.fft.rfft(y - np.min(y), n=n)
    h_fft = np.fft.rfft(response, n=n)
    filt = np.conj(h_fft) / (np.abs(h_fft) ** 2 + max(k, 1e-12))
    out = np.fft.irfft(y_fft * filt, n=n)[: y.size]
    out -= np.min(out)
    if np.max(out) > 0:
        out /= np.max(out)
    return out


def build_waveform_example(path: Path, args: argparse.Namespace, label: str, sample_dir: str, rng: np.random.Generator) -> WaveformExample:
    with h5py.File(path, "r") as handle:
        required = [args.time_key, args.time_end_key, args.amplitude_key]
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{label}: missing waveform dataset(s): {', '.join(missing)}")
        n_events = int(handle[args.time_key].shape[0])
        candidates = _choose_indices(n_events, min(max(int(args.max_events), 200), 5000), str(args.sampling), rng)
        best_idx = int(candidates[0])
        best_sum = -np.inf
        for idx in candidates[: min(candidates.size, 500)]:
            amp = np.asarray(handle[args.amplitude_key][idx], dtype=np.float64).reshape(-1)
            total = float(np.nansum(amp[amp > 0.0]))
            if total > best_sum:
                best_sum = total
                best_idx = int(idx)
        start = np.asarray(handle[args.time_key][best_idx], dtype=np.float64).reshape(-1)
        end = np.asarray(handle[args.time_end_key][best_idx], dtype=np.float64).reshape(-1)
        amp = np.asarray(handle[args.amplitude_key][best_idx], dtype=np.float64).reshape(-1)

        truth_time = np.asarray([])
        truth_photons = np.asarray([])
        if args.seed_key in handle and args.energy_key in handle:
            seed = int(handle[args.seed_key][best_idx])
            energy = float(handle[args.energy_key][best_idx])
            digi_path = _digi_path_for_seed(args.digi_base, sample_dir, seed)
            if digi_path is not None:
                digi_energies, digi_times, digi_photons_array = _load_digi_arrays(digi_path, args)
                digi_idx = _match_digi_entry(digi_energies, energy, set(), float(args.energy_match_tolerance))
                if digi_idx is not None:
                    truth_time = np.asarray(digi_times[digi_idx], dtype=np.float64).reshape(-1)
                    truth_photons = np.asarray(digi_photons_array[digi_idx], dtype=np.float64).reshape(-1)

    n = min(start.size, end.size, amp.size)
    time, raw = _build_interval_waveform(start[:n], end[:n], amp[:n], float(args.waveform_bin_width))
    if raw.size and np.max(raw) > 0:
        raw = raw / np.max(raw)
    deconv = _wiener_deconvolve(raw, float(args.waveform_bin_width), float(args.response_tau), float(args.wiener_k))
    return WaveformExample(label, time, raw, deconv, truth_time, truth_photons)


def _annotate_stats(ax: plt.Axes, samples: Sequence[TimingSample]) -> None:
    lines = []
    for sample in samples:
        r = sample.residual[np.isfinite(sample.residual)]
        if r.size == 0:
            continue
        lines.append(f"{_plain_label(sample.label)}: RMS {np.std(r):.2f} ns")
    ax.text(
        0.98,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.8, "pad": 4.0},
    )


def plot(samples: Sequence[TimingSample], waveform: WaveformExample, args: argparse.Namespace) -> Mapping[str, object]:
    set_publication_style()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    xlim = _shared_residual_range(samples, float(args.range_percentile))
    bins = np.linspace(xlim[0], xlim[1], int(args.bins) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    ax0, ax1, ax2 = axes

    summary: dict[str, object] = {"samples": {}}
    for sample in samples:
        residual = sample.residual[np.isfinite(sample.residual)]
        if residual.size == 0:
            continue
        label = _plain_label(sample.label)
        color = _color(sample.label)
        ax0.hist(residual, bins=bins, density=True, histtype="step", linewidth=1.8, color=color, label=label)
        x, y, yerr = _resolution_points(sample, int(args.count_bins))
        if x.size:
            ax1.errorbar(x, y, yerr=yerr, marker="o", markersize=4.5, linewidth=1.4, capsize=2.5, color=color, label=label)
        summary["samples"][sample.label] = {
            "n_events": int(residual.size),
            "mean_residual_ns": float(np.mean(residual)),
            "rms_residual_ns": float(np.std(residual, ddof=0)),
            "median_photon_count": float(np.median(sample.photon_count)) if sample.photon_count.size else None,
        }

    ax0.axvline(0.0, color="0.25", linewidth=1.0, linestyle=":")
    ax0.set_xlabel(r"$t_{\mathrm{reco}} - t_{\mathrm{truth}}$ [ns]")
    ax0.set_ylabel("Normalized events")
    ax0.set_xlim(*xlim)
    ax0.legend(frameon=False, loc="upper left")
    _annotate_stats(ax0, samples)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Photon count $N_{\gamma}$")
    ax1.set_ylabel(r"Timing resolution RMS [ns]")
    ax1.legend(frameon=False, loc="best")

    if waveform.time.size and waveform.raw.size:
        ax2.step(waveform.time, waveform.raw, where="mid", color="0.25", linewidth=1.3, label="Before deconvolution")
        ax2.step(
            waveform.time,
            waveform.deconvolved,
            where="mid",
            color="#0072B2",
            linewidth=1.7,
            label="After Wiener deconvolution",
        )
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("Normalized amplitude")
    if np.isfinite(args.waveform_xmin) and np.isfinite(args.waveform_xmax) and args.waveform_xmax > args.waveform_xmin:
        ax2.set_xlim(float(args.waveform_xmin), float(args.waveform_xmax))
    ax2.set_ylim(bottom=-0.05)

    if args.show_truth_photon_overlay and waveform.truth_time.size > 0 and waveform.truth_photons.size > 0:
        tt = waveform.truth_time
        tp = waveform.truth_photons
        valid = np.isfinite(tt) & np.isfinite(tp) & (tp > 0)
        tt = tt[valid]
        tp = tp[valid]
        if tt.size > 0 and waveform.time.size > 0:
            # Align truth peak to deconvolved peak
            t_deconv_peak = waveform.time[np.argmax(waveform.deconvolved)]
            hist, edges = np.histogram(tt, bins=50, weights=tp)
            t_truth_peak = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
            shift = t_deconv_peak - t_truth_peak
            tt_shifted = tt + shift

            ax2_twin = ax2.twinx()
            ax2_twin.hist(tt_shifted, bins=np.arange(tt_shifted.min(), tt_shifted.max()+1.0, 1.0), weights=tp, color="#D55E00", alpha=0.3, label="Geant4 truth photons")
            ax2_twin.set_ylabel("Truth photon count", color="#D55E00")
            ax2_twin.tick_params(axis="y", labelcolor="#D55E00")
            ax2_twin.set_ylim(bottom=0)

            lines_1, labels_1 = ax2.get_legend_handles_labels()
            lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper right")
        else:
            ax2.legend(frameon=False, loc="upper right")
    else:
        ax2.legend(frameon=False, loc="upper right")

    ax2.text(0.04, 0.93, _plain_label(waveform.label), transform=ax2.transAxes, ha="left", va="top", fontsize=13)

    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"{args.out_name}.pdf"
    png_path = args.out_dir / f"{args.out_name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    summary["outputs"] = {"pdf": str(pdf_path), "png": str(png_path)}
    summary["configuration"] = {
        "truth_source": args.truth_source,
        "truth_mode": args.truth_mode,
        "sampling": args.sampling,
        "digi_base": str(args.digi_base),
        "digi_time_branch": args.digi_time_branch,
        "digi_photon_branch": args.digi_photon_branch,
        "photon_count_mode": args.photon_count_mode,
        "center_residuals": bool(args.center_residuals),
        "show_truth_photon_overlay": bool(args.show_truth_photon_overlay),
        "energy_match_tolerance_GeV": float(args.energy_match_tolerance),
        "time_key": args.time_key,
        "amplitude_key": args.amplitude_key,
        "waveform_bin_width_ns": float(args.waveform_bin_width),
        "waveform_xlim_ns": [float(args.waveform_xmin), float(args.waveform_xmax)],
        "response_tau_ns": float(args.response_tau),
        "wiener_k": float(args.wiener_k),
    }
    return summary


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.files)
    if len(paths) != len(args.files):
        raise FileNotFoundError(f"Resolved {len(paths)} input files for {len(args.files)} requested patterns.")
    if args.labels is not None and len(args.labels) != len(paths):
        raise ValueError("--labels must match --files.")
    if args.digi_sample_dirs is not None and len(args.digi_sample_dirs) != len(paths):
        raise ValueError("--digi-sample-dirs must match --files.")
    labels = list(args.labels) if args.labels else [default_sample_label(path) for path in paths]
    sample_dirs = list(args.digi_sample_dirs) if args.digi_sample_dirs is not None else [path.stem for path in paths]
    rng = np.random.default_rng(int(args.seed))

    samples = [load_sample(path, args, label, sample_dir, rng) for path, label, sample_dir in zip(paths, labels, sample_dirs)]
    samples = [sample for sample in samples if sample.residual.size > 0]
    if not samples:
        raise RuntimeError("No valid event-level timing samples were found.")

    if args.center_residuals:
        for sample in samples:
            valid_res = sample.residual[np.isfinite(sample.residual)]
            if valid_res.size > 0:
                sample.reco_time -= float(np.mean(valid_res))

    waveform = build_waveform_example(paths[0], args, labels[0], sample_dirs[0], rng)
    summary = plot(samples, waveform, args)

    if args.summary_json:
        summary_path = args.out_dir / f"{args.out_name}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary: {summary_path}")
    print(f"Saved PDF: {summary['outputs']['pdf']}")
    print(f"Saved PNG: {summary['outputs']['png']}")


if __name__ == "__main__":
    main()
