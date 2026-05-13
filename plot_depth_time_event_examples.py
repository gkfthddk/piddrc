#!/usr/bin/env python
"""Plot random event examples of depth-time hit structure for gamma and pi0."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plot_common import SIM3D_E, SIM3D_Z, TIME_KEY, default_sample_label, projected_shower_depths_mm, read_row, resolve_paths
from utils.plot_helpers import set_publication_style

set_publication_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot random gamma/pi0 event examples of depth-time hit structure.")
    p.add_argument(
        "--files",
        nargs="+",
        default=["h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"],
        help="Input HDF5 files. Default compares gamma and pi0 samples.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --files order. Defaults to file stem names.",
    )
    p.add_argument(
        "--examples-per-sample",
        type=int,
        default=4,
        help="Number of random events to plot for each sample.",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=10000,
        help="Maximum number of events to scan per file (<=0 means all).",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=20,
        help="Ignore events with fewer than this many valid hits.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for event sampling.",
    )
    p.add_argument(
        "--out-dir",
        default="plots/depth_time_event_examples",
        help="Output directory.",
    )
    p.add_argument(
        "--out-name",
        default="gamma_vs_pi0_depth_time_event_examples.png",
        help="Output filename.",
    )
    return p.parse_args()



def _sample_events(file_path: Path, *, max_events: int, min_hits: int, n_examples: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    valid_events: list[int] = []
    with h5py.File(file_path, "r") as handle:
        if SIM3D_Z not in handle:
            raise KeyError(f"{SIM3D_Z} missing in {file_path}")
        if TIME_KEY not in handle:
            raise KeyError(f"{TIME_KEY} missing in {file_path}")
        n_events = int(handle[SIM3D_Z].shape[0])
        take = n_events if int(max_events) <= 0 else min(n_events, int(max_events))
        for idx in tqdm(range(take), desc=f"{file_path.stem} sample", unit="event", leave=False):
            z = read_row(handle, SIM3D_Z, idx)
            t = read_row(handle, TIME_KEY, idx)
            e = read_row(handle, SIM3D_E, idx)
            n = min(z.size, t.size, e.size)
            if n == 0:
                continue
            valid = np.isfinite(z[:n]) & np.isfinite(t[:n]) & np.isfinite(e[:n]) & (e[:n] > 0.0)
            if int(np.sum(valid)) >= int(min_hits):
                valid_events.append(idx)
    if not valid_events:
        return []
    k = min(int(n_examples), len(valid_events))
    if k <= 0:
        return []
    return list(rng.choice(np.asarray(valid_events, dtype=np.int64), size=k, replace=False))



def _collect_event_points(file_path: Path, event_idx: int, *, min_hits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(file_path, "r") as handle:
        depth, e = projected_shower_depths_mm(handle, event_idx)
        t = read_row(handle, TIME_KEY, event_idx)
    n = min(depth.size, t.size, e.size)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    depth = depth[:n]
    t = t[:n]
    e = e[:n]
    valid = np.isfinite(depth) & np.isfinite(t) & np.isfinite(e) & (e > 0.0)
    if int(np.sum(valid)) < int(min_hits):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.asarray(depth[valid], dtype=np.float64), np.asarray(t[valid], dtype=np.float64), np.asarray(e[valid], dtype=np.float64)



def _axis_limits(samples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[tuple[float, float], tuple[float, float]]:
    zvals = [z.ravel() for z, _ in samples if z.size > 0]
    tvals = [t.ravel() for _, t in samples if t.size > 0]
    if not zvals or not tvals:
        return (0.0, 1.0), (0.0, 1.0)
    zall = np.concatenate(zvals)
    tall = np.concatenate(tvals)
    zlo, zhi = np.percentile(zall[np.isfinite(zall)], [2.5, 97.5])
    tlo, thi = np.percentile(tall[np.isfinite(tall)], [2.5, 97.5])
    if zlo == zhi:
        zlo -= 1.0
        zhi += 1.0
    if tlo == thi:
        tlo -= 1.0
        thi += 1.0
    return (float(zlo), float(zhi)), (float(tlo), float(thi))



def main() -> None:
    args = parse_args()
    files = resolve_paths(args.files)
    if not files:
        raise FileNotFoundError("No input files found.")
    if args.labels is not None and len(args.labels) != len(files):
        raise ValueError("--labels must match --files count.")

    labels = list(args.labels) if args.labels is not None else [default_sample_label(Path(str(p))) for p in files]

    sampled: list[tuple[str, Path, list[int]]] = []
    for i, (label, file_path) in enumerate(zip(labels, files)):
        idxs = _sample_events(
            file_path,
            max_events=int(args.max_events),
            min_hits=int(args.min_hits),
            n_examples=int(args.examples_per_sample),
            seed=int(args.seed) + i,
        )
        sampled.append((label, file_path, idxs))

    all_points: list[tuple[np.ndarray, np.ndarray]] = []
    all_energies: list[np.ndarray] = []
    examples: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for label, file_path, idxs in sampled:
        for idx in idxs:
            z, t, e = _collect_event_points(file_path, idx, min_hits=int(args.min_hits))
            if z.size == 0:
                continue
            all_points.append((z, t))
            all_energies.append(e)
            examples.append((label, idx, z, t, e))

    if not examples:
        raise RuntimeError("No valid event examples found.")

    (zlo, zhi), (tlo, thi) = _axis_limits(all_points)
    emax = max(float(np.max(e)) for e in all_energies if e.size > 0)
    emin = min(float(np.min(e[e > 0.0])) for e in all_energies if np.any(e > 0.0))

    n_rows = len(sampled)
    n_cols = int(args.examples_per_sample)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), constrained_layout=True, sharex=True, sharey=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    last_mesh = None
    for row, (label, file_path, idxs) in enumerate(sampled):
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(idxs):
                ax.axis("off")
                continue
            idx = idxs[col]
            z, t, e = _collect_event_points(file_path, idx, min_hits=int(args.min_hits))
            if z.size == 0:
                ax.axis("off")
                continue
            mesh = ax.scatter(
                z,
                t,
                c=e,
                s=4.0,
                alpha=0.65,
                cmap="viridis",
                vmin=emin,
                vmax=emax,
                linewidths=0.0,
            )
            last_mesh = mesh
            ax.set_xlim(zlo, zhi)
            ax.set_ylim(tlo, thi)
            if row == 0:
                ax.set_title(f"event {idx}")
            if col == 0:
                ax.set_ylabel(f"{label}\ntime [ns]")
            if row == n_rows - 1:
                ax.set_xlabel("shower depth [mm]")

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label("hit energy")

    fig.suptitle(r"Random event examples of raw shower-depth structure: $\gamma$ vs $\pi^0$")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
