#!/usr/bin/env python
"""Create an intuitive overview figure for point-cloud-style calorimeter inputs.

The script draws one representative event from one or more HDF5 files and shows:
- raw point cloud structure
- the same points colored by timing
- the same points with shuffled timing, keeping geometry fixed

Each point is a hit. Point size is derived from hit amplitude and color encodes time.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from utils.plot_helpers import safe_name


HIT_KEYS = {
    "x": "DRcalo3dHits.position.x",
    "y": "DRcalo3dHits.position.y",
    "z": "DRcalo3dHits.position.z",
    "amp": "DRcalo3dHits.amplitude_sum",
    "time": "DRcalo3dHits.time",
    "time_end": "DRcalo3dHits.time_end",
}


@dataclass
class EventCloud:
    file_path: Path
    event_index: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    amp: np.ndarray
    time: np.ndarray
    time_end: np.ndarray | None
    metadata: dict[str, float | int | str | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a representative point-cloud event from HDF5 inputs."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Input HDF5 files or globs, for example: h5s/pi0_1-120GeV.h5py",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        default=0,
        help="Which resolved file to use when multiple files are provided.",
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=None,
        help="Explicit event index to visualize. If omitted, a representative event is selected.",
    )
    parser.add_argument(
        "--selection",
        choices=["random", "median-hits", "first", "max-hits", "max-amplitude"],
        default="random",
        help="How to choose a representative event when --event-index is not provided.",
    )
    parser.add_argument(
        "--scan-events",
        type=int,
        default=256,
        help="How many events to inspect when selecting a representative event.",
    )
    parser.add_argument(
        "--projection",
        choices=["xy", "xz", "yz"],
        default="yz",
        help="Legacy single-projection selector; the figure now renders the yz plane.",
    )
    parser.add_argument(
        "--min-hit-energy",
        type=float,
        default=0.5,
        help="Drop hits with amplitude_sum below this value before plotting. Recommended default: 0.5.",
    )
    parser.add_argument(
        "--max-hit-duration",
        type=float,
        default=30.0,
        help="Drop hits with time_end-time above this value before plotting. Recommended default: 30.0.",
    )
    parser.add_argument(
        "--min-theta",
        type=float,
        default=1.5,
        help="Only consider events with GenParticles.momentum.theta at or above this value. Recommended default: 1.5.",
    )
    parser.add_argument(
        "--min-e-gen",
        type=float,
        default=10.0,
        help="Only consider events with E_gen at or above this value. Recommended default: 10.0.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1200,
        help="Maximum number of hits to draw from the event (<=0 means all).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=12345,
        help="Seed for timing shuffling.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/point_cloud_overview",
        help="Directory where the figure will be written.",
    )
    parser.add_argument(
        "--out-name",
        default="point_cloud_overview.png",
        help="Output image filename.",
    )
    return parser.parse_args()


def _resolve_paths(patterns: Iterable[str]) -> list[Path]:
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


def _read_1d_row(handle: h5py.File, key: str, event_index: int) -> np.ndarray:
    if key not in handle:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(handle[key][event_index], dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _event_selection_mask(
    theta_vals: np.ndarray,
    e_gen_vals: np.ndarray,
    *,
    min_theta: float,
    min_e_gen: float,
) -> np.ndarray:
    return (
        np.isfinite(theta_vals)
        & np.isfinite(e_gen_vals)
        & (theta_vals >= min_theta)
        & (e_gen_vals >= min_e_gen)
    )


def _select_event_index(
    handle: h5py.File,
    *,
    selection: str,
    scan_events: int,
    min_theta: float,
    min_e_gen: float,
    seed: int,
) -> int:
    if HIT_KEYS["amp"] not in handle:
        raise KeyError(f"Missing required dataset: {HIT_KEYS['amp']}")

    n_events = int(handle[HIT_KEYS["amp"]].shape[0])
    if n_events <= 0:
        raise RuntimeError("Input file contains no events.")

    limit = n_events if scan_events <= 0 else min(n_events, scan_events)
    counts = np.zeros(limit, dtype=np.float64)
    amplitudes = np.zeros(limit, dtype=np.float64)
    theta_vals = np.full(limit, np.nan, dtype=np.float64)
    e_gen_vals = np.full(limit, np.nan, dtype=np.float64)

    for idx in range(limit):
        amp = _read_1d_row(handle, HIT_KEYS["amp"], idx)
        x = _read_1d_row(handle, HIT_KEYS["x"], idx)
        y = _read_1d_row(handle, HIT_KEYS["y"], idx)
        z = _read_1d_row(handle, HIT_KEYS["z"], idx)
        theta = _read_1d_row(handle, "GenParticles.momentum.theta", idx)
        e_gen = _read_1d_row(handle, "E_gen", idx)
        valid = min(amp.size, x.size, y.size, z.size)
        counts[idx] = float(valid)
        amplitudes[idx] = float(np.sum(amp[:valid])) if valid > 0 else 0.0
        theta_vals[idx] = float(theta[0]) if theta.size > 0 else float("nan")
        e_gen_vals[idx] = float(e_gen[0]) if e_gen.size > 0 else float("nan")

    valid = np.isfinite(counts) & _event_selection_mask(
        theta_vals,
        e_gen_vals,
        min_theta=min_theta,
        min_e_gen=min_e_gen,
    )
    if not np.any(valid):
        raise RuntimeError(f"No events found with theta >= {min_theta} and E_gen >= {min_e_gen}.")

    valid_indices = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)

    if selection == "max-hits":
        masked = np.where(valid, counts, -np.inf)
        return int(np.argmax(masked))
    if selection == "max-amplitude":
        masked = np.where(valid, amplitudes, -np.inf)
        return int(np.argmax(masked))
    if selection == "first":
        return int(valid_indices[0])
    if selection == "random":
        return int(rng.choice(valid_indices))

    target = float(np.nanmedian(counts[valid]))
    masked_diff = np.where(valid, np.abs(counts - target), np.inf)
    return int(np.argmin(masked_diff))


def _fallback_event_index(
    handle: h5py.File,
    *,
    min_theta: float,
    min_e_gen: float,
    seed: int,
    preferred_index: int | None = None,
) -> int:
    theta_key = "GenParticles.momentum.theta"
    e_gen_key = "E_gen"
    if theta_key not in handle or e_gen_key not in handle:
        if preferred_index is not None:
            return int(preferred_index)
        raise RuntimeError(f"Missing required dataset: {theta_key} or {e_gen_key}")

    theta_vals = np.asarray(handle[theta_key][:], dtype=np.float64).reshape(-1)
    e_gen_vals = np.asarray(handle[e_gen_key][:], dtype=np.float64).reshape(-1)
    valid = _event_selection_mask(
        theta_vals,
        e_gen_vals,
        min_theta=min_theta,
        min_e_gen=min_e_gen,
    )
    if not np.any(valid):
        raise RuntimeError(f"No events found with theta >= {min_theta} and E_gen >= {min_e_gen}.")

    valid_indices = np.flatnonzero(valid)
    if preferred_index is not None:
        preferred_index = int(np.clip(preferred_index, 0, theta_vals.size - 1))
        later = valid_indices[valid_indices >= preferred_index]
        if later.size > 0:
            valid_indices = later
    rng = np.random.default_rng(seed)
    return int(rng.choice(valid_indices))


def _load_event(file_path: Path, event_index: int) -> EventCloud:
    with h5py.File(file_path, "r") as handle:
        x = _read_1d_row(handle, HIT_KEYS["x"], event_index)
        y = _read_1d_row(handle, HIT_KEYS["y"], event_index)
        z = _read_1d_row(handle, HIT_KEYS["z"], event_index)
        amp = _read_1d_row(handle, HIT_KEYS["amp"], event_index)
        time = _read_1d_row(handle, HIT_KEYS["time"], event_index)
        time_end = _read_1d_row(handle, HIT_KEYS["time_end"], event_index) if HIT_KEYS["time_end"] in handle else None

        sizes = [x.size, y.size, z.size, amp.size, time.size]
        if time_end is not None:
            sizes.append(time_end.size)
        n = int(min(sizes)) if sizes else 0
        if n <= 0:
            raise RuntimeError(f"Event {event_index} in {file_path} has no usable hits.")

        x = x[:n]
        y = y[:n]
        z = z[:n]
        amp = amp[:n]
        time = time[:n]
        if time_end is not None:
            time_end = time_end[:n]

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(amp) & np.isfinite(time)
        mask &= amp > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        amp = amp[mask]
        time = time[mask]
        if time_end is not None:
            time_end = time_end[mask]

        if x.size == 0:
            raise RuntimeError(f"Event {event_index} in {file_path} has no valid hits after masking.")

        metadata: dict[str, float | int | str | None] = {
            "n_hits": int(x.size),
            "amp_sum": float(np.sum(amp)),
            "time_min": float(np.min(time)),
            "time_max": float(np.max(time)),
        }
        for key in ("E_gen", "E_dep", "E_leak"):
            if key in handle:
                scalar = np.asarray(handle[key][event_index]).reshape(-1)
                metadata[key] = float(scalar[0]) if scalar.size and np.isfinite(scalar[0]) else None
            else:
                metadata[key] = None
        for key in ("GenParticles.PDG", "GenParticles.momentum.theta", "GenParticles.momentum.phi"):
            if key in handle:
                scalar = np.asarray(handle[key][event_index]).reshape(-1)
                metadata[key] = float(scalar[0]) if scalar.size and np.isfinite(scalar[0]) else None
            else:
                metadata[key] = None

    return EventCloud(
        file_path=file_path,
        event_index=int(event_index),
        x=x,
        y=y,
        z=z,
        amp=amp,
        time=time,
        time_end=time_end,
        metadata=metadata,
    )


def _project(cloud: EventCloud, projection: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    if projection == "xy":
        return cloud.x, cloud.y, "x", "y"
    if projection == "yz":
        return cloud.y, cloud.z, "y", "z"
    return cloud.x, cloud.z, "x", "z"


def _subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _apply_energy_threshold(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    amp: np.ndarray,
    time: np.ndarray,
    time_end: np.ndarray | None,
    *,
    min_hit_energy: float,
    max_hit_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    keep = np.isfinite(x) & (x > 0)
    keep &= np.isfinite(amp) & (amp >= min_hit_energy)
    if time_end is not None and max_hit_duration > 0:
        keep &= np.isfinite(time_end) & np.isfinite(time) & ((time_end - time) <= max_hit_duration)
    out_time_end = time_end[keep] if time_end is not None else None
    return x[keep], y[keep], z[keep], amp[keep], time[keep], out_time_end


def _scaled_sizes(amp: np.ndarray) -> np.ndarray:
    finite = amp[np.isfinite(amp)]
    if finite.size == 0:
        return np.full(amp.shape, 2.5, dtype=np.float64)
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(amp.shape, 3.0, dtype=np.float64)
    scaled = np.clip((amp - lo) / (hi - lo), 0.0, 1.0)
    return 3.6 + 48.0 * np.power(scaled, 0.9)


def _timing_norm(time: np.ndarray) -> Normalize:
    finite = time[np.isfinite(time)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.percentile(finite, 5))
    vmax = float(np.percentile(finite, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def _event_timing_value(time: np.ndarray, time_end: np.ndarray | None) -> np.ndarray:
    if time_end is None or time_end.size != time.size:
        return time
    return 0.5 * (time + time_end)


def _event_duration(time: np.ndarray, time_end: np.ndarray | None) -> np.ndarray:
    if time_end is None or time_end.size != time.size:
        return np.zeros_like(time)
    return np.maximum(time_end - time, 0.0)


def _plot_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    sizes: np.ndarray,
    colors: np.ndarray | None,
    alphas: np.ndarray | None = None,
    cmap: str = "coolwarm",
    norm: Normalize | None = None,
    raw_color: str = "#9a9a9a",
) -> None:
    if colors is None:
        ax.scatter(x, y, s=sizes, c=raw_color, alpha=0.45, linewidths=0.0, edgecolors="none", marker="s")
    else:
        facecolors = plt.get_cmap(cmap)(norm(colors) if norm is not None else colors)
        if alphas is not None:
            facecolors = np.asarray(facecolors)
            facecolors[:, 3] = alphas
        sc = ax.scatter(
            x,
            y,
            s=sizes,
            c=facecolors,
            linewidths=0.0,
            edgecolors="none",
            marker="s",
        )
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(colors)
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("time (ns)")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.set_aspect("equal", adjustable="datalim")


def _set_centered_window(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, width: float) -> None:
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if finite_x.size == 0 or finite_y.size == 0:
        return
    half = 0.5 * float(width)
    x_center = float(np.median(finite_x))
    y_center = float(np.median(finite_y))
    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)


def _save_x_timing_scatter(
    *,
    x: np.ndarray,
    timing: np.ndarray,
    sizes: np.ndarray,
    duration_alpha: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    sc = ax.scatter(
        x,
        timing,
        s=sizes,
        c=timing,
        cmap="coolwarm",
        alpha=duration_alpha,
        linewidths=0.0,
        edgecolors="none",
        marker="s",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("midpoint time")
    ax.set_xlabel("x")
    ax.set_ylabel("midpoint time")
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.28)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    files = _resolve_paths(args.files)
    if not files:
        raise FileNotFoundError("No input files found.")

    file_index = int(np.clip(args.file_index, 0, len(files) - 1))
    file_path = files[file_index]

    with h5py.File(file_path, "r") as handle:
        if args.event_index is None:
            event_index = _select_event_index(
                handle,
                selection=args.selection,
                scan_events=args.scan_events,
                min_theta=float(args.min_theta),
                min_e_gen=float(args.min_e_gen),
                seed=int(args.shuffle_seed),
            )
        else:
            preferred_index = int(args.event_index)
            event_index = preferred_index
            theta_key = "GenParticles.momentum.theta"
            e_gen_key = "E_gen"
            theta_vals = np.asarray(handle[theta_key][preferred_index]).reshape(-1) if theta_key in handle else np.empty(0)
            e_gen_vals = np.asarray(handle[e_gen_key][preferred_index]).reshape(-1) if e_gen_key in handle else np.empty(0)
            theta_value = float(theta_vals[0]) if theta_vals.size > 0 and np.isfinite(theta_vals[0]) else None
            e_gen_value = float(e_gen_vals[0]) if e_gen_vals.size > 0 and np.isfinite(e_gen_vals[0]) else None
            if (
                theta_value is None
                or e_gen_value is None
                or theta_value < float(args.min_theta)
                or e_gen_value < float(args.min_e_gen)
            ):
                fallback_index = _fallback_event_index(
                    handle,
                    min_theta=float(args.min_theta),
                    min_e_gen=float(args.min_e_gen),
                    seed=int(args.shuffle_seed),
                    preferred_index=preferred_index + 1,
                )
                print(
                    f"Selected event theta={theta_value if theta_value is not None else float('nan'):.3g}, "
                    f"E_gen={e_gen_value if e_gen_value is not None else float('nan'):.3g} "
                    f"does not satisfy min_theta={float(args.min_theta):.3g} and min_e_gen={float(args.min_e_gen):.3g}; "
                    f"using event {fallback_index} instead."
                )
                event_index = fallback_index

    cloud = _load_event(file_path, event_index)

    x, y, z, amp, time, time_end = _apply_energy_threshold(
        cloud.x,
        cloud.y,
        cloud.z,
        cloud.amp,
        cloud.time,
        cloud.time_end,
        min_hit_energy=float(args.min_hit_energy),
        max_hit_duration=float(args.max_hit_duration),
    )

    indices = _subsample_indices(x.size, int(args.max_points), seed=int(args.shuffle_seed))
    x = x[indices]
    y = y[indices]
    z = z[indices]
    amp = amp[indices]
    time = time[indices]
    if time_end is not None:
        time_end = time_end[indices]

    timing = _event_timing_value(time, time_end)
    duration = _event_duration(time, time_end)
    if float(args.max_hit_duration) > 0:
        duration_alpha = np.clip(
            0.90 - 0.85 * np.minimum(duration / 2.0, 1.0),
            0.10,
            0.90,
        )
    else:
        duration_alpha = np.full(duration.shape, 0.90, dtype=np.float64)

    rng = np.random.default_rng(int(args.shuffle_seed))
    shuffled_time = rng.permutation(timing)
    sizes = _scaled_sizes(amp)
    time_norm = _timing_norm(timing)

    order = np.argsort(-timing, kind="stable")
    x = x[order]
    y = y[order]
    z = z[order]
    amp = amp[order]
    time = time[order]
    timing = timing[order]
    shuffled_time = shuffled_time[order]
    sizes = sizes[order]
    duration_alpha = duration_alpha[order]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    axes = np.atleast_2d(axes)

    projections = [
        ("yz", y, z, "y", "z"),
    ]
    for row, (proj_name, px, py, xlabel, ylabel) in enumerate(projections):
        row_axes = axes[row]
        row_axes[0].text(
            -0.12,
            0.5,
            f"{proj_name.upper()} plane",
            transform=row_axes[0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        _plot_panel(
            row_axes[0],
            px,
            py,
            title="1. No Timing",
            xlabel=xlabel,
            ylabel=ylabel,
            sizes=sizes,
            colors=None,
        )
        _plot_panel(
            row_axes[1],
            px,
            py,
            title="2. With Timing",
            xlabel=xlabel,
            ylabel=ylabel,
            sizes=sizes,
            colors=timing,
            alphas=duration_alpha,
            norm=time_norm,
        )
        _plot_panel(
            row_axes[2],
            px,
            py,
            title="3. Shuffled Timing",
            xlabel=xlabel,
            ylabel=ylabel,
            sizes=sizes,
            colors=shuffled_time,
            alphas=duration_alpha,
            norm=time_norm,
        )
        _set_centered_window(row_axes[0], px, py, width=50.0)
        _set_centered_window(row_axes[1], px, py, width=50.0)
        _set_centered_window(row_axes[2], px, py, width=50.0)

    title_parts = [
        safe_name(file_path.stem),
        f"event {event_index}",
        f"hits={int(cloud.metadata['n_hits'])}",
    ]
    if float(args.min_hit_energy) > 0:
        title_parts.append(f"min_hit_energy={float(args.min_hit_energy):.3g}")
    if float(args.max_hit_duration) > 0:
        title_parts.append(f"max_hit_duration={float(args.max_hit_duration):.3g}")
    title_parts.append("x>0")
    title_parts.append(f"min_theta={float(args.min_theta):.3g}")
    title_parts.append(f"min_e_gen={float(args.min_e_gen):.3g}")
    if cloud.metadata.get("E_gen") is not None:
        title_parts.append(f"E_gen={float(cloud.metadata['E_gen']):.3g}")
    if cloud.metadata.get("GenParticles.momentum.theta") is not None:
        title_parts.append(f"theta={float(cloud.metadata['GenParticles.momentum.theta']):.3g}")

    fig.suptitle(" | ".join(title_parts), fontsize=16)


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")

    scatter_title = "x vs midpoint time"
    if title_parts:
        scatter_title = f"{scatter_title} | " + " | ".join(title_parts[:4])
    scatter_path = out_dir / "x_vs_midpoint_time_scatter.png"
    _save_x_timing_scatter(
        x=x,
        timing=timing,
        sizes=sizes,
        duration_alpha=duration_alpha,
        out_path=scatter_path,
        title=scatter_title,
    )
    print(f"Wrote {scatter_path}")


if __name__ == "__main__":
    main()
