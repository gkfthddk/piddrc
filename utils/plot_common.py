"""Shared plotting and depth-selection helpers."""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np
from tqdm.auto import tqdm

SIM3D_E = "Sim3dCalorimeterHits.energy"
SIM3D_X = "Sim3dCalorimeterHits.position.x"
SIM3D_Y = "Sim3dCalorimeterHits.position.y"
SIM3D_Z = "Sim3dCalorimeterHits.position.z"
TIME_KEY = "DRcalo3dHits.time"
TIME_END_KEY = "DRcalo3dHits.time_end"
AMP_KEY = "DRcalo3dHits.amplitude_sum"
THETA_KEY = "GenParticles.momentum.theta"
PHI_KEY = "GenParticles.momentum.phi"
ENERGY_KEY = "E_gen"
VERTEX_X = "GenParticles.vertex.x"
VERTEX_Y = "GenParticles.vertex.y"
VERTEX_Z = "GenParticles.vertex.z"


def resolve_paths(patterns: Iterable[str]) -> list[Path]:
    roots = [Path("."), Path("h5s"), Path("/store/ml/dual-readout/h5s")]
    paths: list[Path] = []
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        found: list[Path] = []
        if Path(expanded).is_absolute():
            found.extend(sorted(Path(p) for p in glob.glob(expanded)))
        else:
            for root in roots:
                found.extend(sorted(root.glob(pattern)))
        if found:
            paths.extend(found)
            continue
        candidate = Path(expanded)
        if candidate.exists():
            paths.append(candidate)
            continue
        for root in roots[1:]:
            alt = root / expanded
            if alt.exists():
                paths.append(alt)
                break

    dedup: list[Path] = []
    seen: set[str] = set()
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


def read_row(handle: h5py.File, key: str, idx: int) -> np.ndarray:
    if key not in handle:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(handle[key][idx], dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def read_midpoint_time_row(handle: h5py.File, idx: int) -> np.ndarray:
    starts = read_row(handle, TIME_KEY, idx)
    ends = read_row(handle, TIME_END_KEY, idx)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)
    if ends.size != starts.size:
        return starts
    return 0.5 * (starts + ends)


def read_shifted_time_row(handle: h5py.File, idx: int) -> np.ndarray:
    starts = read_row(handle, TIME_KEY, idx)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)
    return starts - float(np.min(starts))


def unit_vector(theta: float, phi: float) -> np.ndarray:
    st = math.sin(theta)
    return np.asarray([st * math.cos(phi), st * math.sin(phi), math.cos(theta)], dtype=np.float64)


def tower_head_distance_mm(theta: float) -> float:
    s = math.sin(theta)
    c = math.cos(theta)
    if abs(s) < 1e-12 or abs(c) < 1e-12:
        return float("nan")
    return min(1800.0 / abs(s), 2556.0 / abs(c))


def event_passes_selection(
    handle: h5py.File,
    idx: int,
    *,
    min_theta: float,
    min_e_gen: float,
) -> bool:
    theta = read_row(handle, THETA_KEY, idx)
    e_gen = read_row(handle, ENERGY_KEY, idx)
    if theta.size == 0 or not np.isfinite(theta[0]) or float(theta[0]) <= float(min_theta):
        return False
    if e_gen.size == 0 or not np.isfinite(e_gen[0]) or float(e_gen[0]) < float(min_e_gen):
        return False
    return True


def compute_shower_depth_mm(handle: h5py.File, idx: int) -> float:
    depths, weights = projected_shower_depths_mm(handle, idx)
    if depths.size == 0 or weights.size == 0 or not np.isfinite(weights).any():
        return float("nan")
    return float(np.average(depths, weights=weights))


def projected_shower_depths_mm(handle: h5py.File, idx: int) -> tuple[np.ndarray, np.ndarray]:
    required = [SIM3D_E, SIM3D_X, SIM3D_Y, SIM3D_Z, THETA_KEY, PHI_KEY, VERTEX_X, VERTEX_Y, VERTEX_Z]
    if any(key not in handle for key in required):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    e = read_row(handle, SIM3D_E, idx)
    x = read_row(handle, SIM3D_X, idx)
    y = read_row(handle, SIM3D_Y, idx)
    z = read_row(handle, SIM3D_Z, idx)
    n = min(e.size, x.size, y.size, z.size)
    if n == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    e = e[:n]
    x = x[:n]
    y = y[:n]
    z = z[:n]
    valid = np.isfinite(e) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (e > 0.0)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    theta = read_row(handle, THETA_KEY, idx)
    phi = read_row(handle, PHI_KEY, idx)
    vx = read_row(handle, VERTEX_X, idx)
    vy = read_row(handle, VERTEX_Y, idx)
    vz = read_row(handle, VERTEX_Z, idx)
    if theta.size == 0 or phi.size == 0 or vx.size == 0 or vy.size == 0 or vz.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    axis = unit_vector(float(theta[0]), float(phi[0]))
    vertex = np.asarray([float(vx[0]), float(vy[0]), float(vz[0])], dtype=np.float64)
    pos = np.column_stack([x[valid], y[valid], z[valid]])
    proj = (pos - vertex) @ axis
    if proj.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    head = tower_head_distance_mm(float(theta[0]))
    if not np.isfinite(head):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.asarray(proj - head, dtype=np.float64), np.asarray(e[valid], dtype=np.float64)


def projected_shower_transverse_radii_mm(handle: h5py.File, idx: int) -> tuple[np.ndarray, np.ndarray]:
    required = [SIM3D_E, SIM3D_X, SIM3D_Y, SIM3D_Z, THETA_KEY, PHI_KEY, VERTEX_X, VERTEX_Y, VERTEX_Z]
    if any(key not in handle for key in required):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    e = read_row(handle, SIM3D_E, idx)
    x = read_row(handle, SIM3D_X, idx)
    y = read_row(handle, SIM3D_Y, idx)
    z = read_row(handle, SIM3D_Z, idx)
    n = min(e.size, x.size, y.size, z.size)
    if n == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    e = e[:n]
    x = x[:n]
    y = y[:n]
    z = z[:n]
    valid = np.isfinite(e) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (e > 0.0)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    theta = read_row(handle, THETA_KEY, idx)
    phi = read_row(handle, PHI_KEY, idx)
    vx = read_row(handle, VERTEX_X, idx)
    vy = read_row(handle, VERTEX_Y, idx)
    vz = read_row(handle, VERTEX_Z, idx)
    if theta.size == 0 or phi.size == 0 or vx.size == 0 or vy.size == 0 or vz.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    axis = unit_vector(float(theta[0]), float(phi[0]))
    vertex = np.asarray([float(vx[0]), float(vy[0]), float(vz[0])], dtype=np.float64)
    pos = np.column_stack([x[valid], y[valid], z[valid]])
    rel = pos - vertex
    proj = rel @ axis
    perp = rel - np.outer(proj, axis)
    radii = np.linalg.norm(perp, axis=1)
    if radii.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.asarray(radii, dtype=np.float64), np.asarray(e[valid], dtype=np.float64)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if values.size == 0 or weights.size == 0:
        return float("nan")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    q = float(np.clip(quantile, 0.0, 1.0)) * total
    return float(np.interp(q, cum, values))


def compute_event_depth_metrics_mm(handle: h5py.File, idx: int) -> tuple[float, float, float]:
    depths, weights = projected_shower_depths_mm(handle, idx)
    if depths.size == 0 or weights.size == 0 or not np.isfinite(weights).any():
        return float("nan"), float("nan"), float("nan")

    valid = np.isfinite(depths) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")

    depths = depths[valid]
    weights = weights[valid]
    mean = float(np.average(depths, weights=weights))
    spread = float(np.sqrt(np.average((depths - mean) ** 2, weights=weights)))
    start = float(np.min(depths))
    return mean, spread, start


def compute_event_transverse_metrics_mm(handle: h5py.File, idx: int) -> tuple[float, float, float, float, float]:
    radii, weights = projected_shower_transverse_radii_mm(handle, idx)
    if radii.size == 0 or weights.size == 0 or not np.isfinite(weights).any():
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    valid = np.isfinite(radii) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    radii = radii[valid]
    weights = weights[valid]
    mean = float(np.average(radii, weights=weights))
    centered = radii - mean
    m2 = float(np.average(centered**2, weights=weights))
    spread = float(np.sqrt(m2))
    p5 = weighted_quantile(radii, weights, 0.05)
    if m2 > 0.0 and np.isfinite(m2):
        m3 = float(np.average(centered**3, weights=weights))
        m4 = float(np.average(centered**4, weights=weights))
        skewness = float(m3 / (m2 ** 1.5))
        kurtosis = float(m4 / (m2**2) - 3.0)
    else:
        skewness = float("nan")
        kurtosis = float("nan")
    return mean, spread, p5, skewness, kurtosis


def collect_shower_depths(
    path: Path,
    *,
    max_events: int,
    min_theta: float,
    min_e_gen: float,
    desc: str | None = None,
) -> np.ndarray:
    depths: list[float] = []
    with h5py.File(path, "r") as handle:
        if THETA_KEY not in handle:
            raise KeyError(f"{THETA_KEY} missing in {path}")
        if ENERGY_KEY not in handle:
            raise KeyError(f"{ENERGY_KEY} missing in {path}")
        n_events = int(handle[THETA_KEY].shape[0])
        take = n_events if max_events <= 0 else min(n_events, max_events)
        iterator = range(take)
        if desc is not None:
            iterator = tqdm(iterator, desc=desc, unit="evt", leave=False)
        for idx in iterator:
            if not event_passes_selection(
                handle,
                idx,
                min_theta=min_theta,
                min_e_gen=min_e_gen,
            ):
                continue
            depth = compute_shower_depth_mm(handle, idx)
            if np.isfinite(depth):
                depths.append(float(depth))
    return np.asarray(depths, dtype=np.float64)


def collect_shower_depths_and_summary(
    path: Path,
    *,
    max_events: int,
    min_theta: float,
    min_e_gen: float,
    q50_fn: Callable[[h5py.File, int], float],
    desc: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    depths: list[float] = []
    summary: dict[str, list[float]] = {
        "depth_mean": [],
        "depth_spread": [],
        "depth_p5": [],
        "depth_skewness": [],
        "depth_kurtosis": [],
        "tr_mean": [],
        "tr_spread": [],
        "tr_p5": [],
        "tr_skewness": [],
        "tr_kurtosis": [],
        "q50": [],
    }
    with h5py.File(path, "r") as handle:
        if THETA_KEY not in handle:
            raise KeyError(f"{THETA_KEY} missing in {path}")
        if ENERGY_KEY not in handle:
            raise KeyError(f"{ENERGY_KEY} missing in {path}")
        n_events = int(handle[THETA_KEY].shape[0])
        take = n_events if max_events <= 0 else min(n_events, max_events)
        iterator = range(take)
        if desc is not None:
            iterator = tqdm(iterator, desc=desc, unit="evt", leave=False)
        for idx in iterator:
            if not event_passes_selection(
                handle,
                idx,
                min_theta=min_theta,
                min_e_gen=min_e_gen,
            ):
                continue
            depths_arr, weights_arr = projected_shower_depths_mm(handle, idx)
            if depths_arr.size == 0 or weights_arr.size == 0:
                continue
            valid = np.isfinite(depths_arr) & np.isfinite(weights_arr) & (weights_arr > 0.0)
            if not np.any(valid):
                continue
            depths_arr = depths_arr[valid]
            weights_arr = weights_arr[valid]
            mean = float(np.average(depths_arr, weights=weights_arr))
            centered = depths_arr - mean
            m2 = float(np.average(centered**2, weights=weights_arr))
            spread = float(np.sqrt(m2))
            p5 = weighted_quantile(depths_arr, weights_arr, 0.05)
            if m2 > 0.0 and np.isfinite(m2):
                m3 = float(np.average(centered**3, weights=weights_arr))
                m4 = float(np.average(centered**4, weights=weights_arr))
                skewness = float(m3 / (m2 ** 1.5))
                kurtosis = float(m4 / (m2**2) - 3.0)
            else:
                skewness = float("nan")
                kurtosis = float("nan")

            tr_mean, tr_spread, tr_p5, tr_skewness, tr_kurtosis = compute_event_transverse_metrics_mm(handle, idx)
            timing_q50 = float(q50_fn(handle, idx))
            if not np.isfinite(timing_q50) or not np.isfinite(tr_mean):
                continue
            depths.append(mean)
            summary["depth_mean"].append(mean)
            summary["depth_spread"].append(spread)
            summary["depth_p5"].append(p5)
            summary["depth_skewness"].append(skewness)
            summary["depth_kurtosis"].append(kurtosis)
            summary["tr_mean"].append(tr_mean)
            summary["tr_spread"].append(tr_spread)
            summary["tr_p5"].append(tr_p5)
            summary["tr_skewness"].append(tr_skewness)
            summary["tr_kurtosis"].append(tr_kurtosis)
            summary["q50"].append(timing_q50)
    return np.asarray(depths, dtype=np.float64), {
        key: np.asarray(values, dtype=np.float64) for key, values in summary.items()
    }


def shared_range(arrays: list[np.ndarray], percentile: float) -> tuple[float, float]:
    finite = [np.asarray(a, dtype=np.float64).ravel() for a in arrays if np.asarray(a).size > 0]
    finite = [a[np.isfinite(a)] for a in finite if np.isfinite(a).any()]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    if values.size == 0:
        return 0.0, 1.0
    pct = float(np.clip(percentile, 0.0, 100.0))
    lo_q = 0.5 * (100.0 - pct)
    hi_q = 100.0 - lo_q
    lo, hi = np.percentile(values, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.min(values))
        hi = float(np.max(values))
    if lo == hi:
        pad = max(1.0, abs(lo) * 0.05)
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def automatic_bins(values: np.ndarray, xlim: tuple[float, float], fallback_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(xlim[0], xlim[1], fallback_bins + 1)
    clipped = finite[(finite >= xlim[0]) & (finite <= xlim[1])]
    source = clipped if clipped.size >= 2 else finite
    try:
        edges = np.histogram_bin_edges(source, bins="auto", range=xlim)
    except ValueError:
        edges = np.linspace(xlim[0], xlim[1], fallback_bins + 1)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(xlim[0], xlim[1], fallback_bins + 1)
    return edges


def format_hist_label(label: str, mean: float, rms: float) -> str:
    return f"{label}  (μ={mean:.1f} mm, RMS={rms:.1f} mm)"


def plot_hist_overlay(
    ax,
    series: list[tuple[str, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    bins: np.ndarray,
) -> None:
    for label, values in series:
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
            linewidth=1.9,
            label=format_hist_label(label, mean, rms),
        )
        color = tuple(hist[2][0].get_edgecolor())
        ax.axvline(mean, color=color, linestyle="--", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized density")
    ax.set_xlim(*xlim)
    ax.legend(loc="best", fontsize=11)


def plot_scatter_overlay(
    ax,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for label, x, y in series:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size == 0:
            continue
        ax.scatter(x, y, s=8, alpha=0.12, linewidths=0.0, label=label, rasterized=True)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=11)


def default_sample_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("e-"):
        return r"$e^-$"
    if stem.startswith("gamma"):
        return r"$\gamma$"
    if stem.startswith("pi+"):
        return r"$\pi^+$"
    if stem.startswith("pi0"):
        return r"$\pi^0$"
    return stem
