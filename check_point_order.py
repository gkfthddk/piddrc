"""Visualize point-cloud ordering in HDF5 events."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect point-cloud hit ordering in HDF5 files.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["h5s/*_1-100GeV_10.h5py"],
        help="Input .h5py files (paths or globs).",
    )
    parser.add_argument(
        "--mask-feature",
        default="DRcalo3dHits.amplitude_sum",
        help="Dataset used to select valid hits (non-zero, finite).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=5000,
        help="Max events per file to scan (<=0 means all).",
    )
    parser.add_argument(
        "--plot-events",
        type=int,
        default=3,
        help="Number of events per file to plot.",
    )
    parser.add_argument(
        "--y-feature",
        default="DRcalo3dHits.amplitude_sum",
        help="Feature for y-axis in ordering plots.",
    )
    parser.add_argument(
        "--color-feature",
        default=None,
        help="Optional feature used as color map in scatter plot.",
    )
    parser.add_argument(
        "--plot-dir",
        default="plots/point_order",
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="Use line plot instead of scatter.",
    )
    parser.add_argument(
        "--focus-index",
        type=int,
        default=1000,
        help="Center hit index for detailed local distribution checks.",
    )
    parser.add_argument(
        "--focus-window",
        type=int,
        default=100,
        help="Half-window around focus index (e.g. 100 => [900, 1100]).",
    )
    parser.add_argument(
        "--waveform",
        action="store_true",
        help="Also create single-readout waveform plots (time/time_end/amplitude).",
    )
    parser.add_argument(
        "--wave-amp-feature",
        default="DRcalo3dHits.amplitude",
        help="Amplitude feature for waveform y-axis.",
    )
    parser.add_argument(
        "--wave-time-feature",
        default="DRcalo3dHits.time",
        help="Start-time feature for waveform x-axis.",
    )
    parser.add_argument(
        "--wave-time-end-feature",
        default="DRcalo3dHits.time_end",
        help="End-time feature for waveform x-axis.",
    )
    parser.add_argument(
        "--wave-cellid-feature",
        default="DRcalo3dHits.cellID",
        help="CellID feature used to build single-readout waveforms.",
    )
    parser.add_argument(
        "--wave-cells-per-event",
        type=int,
        default=6,
        help="Number of cellIDs to draw per event (top total amplitude first).",
    )
    parser.add_argument(
        "--linked-waveform",
        action="store_true",
        help="Plot model-input style linked waveforms in point-cloud order.",
    )
    parser.add_argument(
        "--linked-gap",
        type=float,
        default=0.0,
        help="Optional gap inserted between consecutive point waveforms on stacked time axis.",
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
    unique = []
    seen = set()
    for path in out:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _valid_mask(handle: h5py.File, mask_feature: str, event_idx: int, row_shape: Tuple[int, ...]) -> np.ndarray:
    if mask_feature in handle:
        base = np.asarray(handle[mask_feature][event_idx])
        if base.shape == row_shape:
            return np.isfinite(base) & (base != 0)
    return np.ones(row_shape, dtype=bool)


def plot_event_ordering(
    path: Path,
    *,
    y_feature: str,
    color_feature: str | None,
    mask_feature: str,
    num_events: int,
    max_events: int,
    out_dir: Path,
    use_line: bool,
    focus_index: int,
    focus_window: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with h5py.File(path, "r") as handle:
        if y_feature not in handle:
            print(f"Skipping plots for {path}: missing {y_feature}")
            return 0

        n_events = handle[y_feature].shape[0]
        n_take = min(max(num_events, 0), n_events)
        if max_events > 0:
            n_take = min(n_take, max_events)
        if n_take == 0:
            return 0

        event_iter = tqdm(
            range(n_take),
            desc=f"plot events {path.stem}",
            leave=False,
        )
        for event_idx in event_iter:
            y_raw = np.asarray(handle[y_feature][event_idx])
            if y_raw.ndim != 1:
                y_raw = y_raw.reshape(-1)
            mask = _valid_mask(handle, mask_feature, event_idx, y_raw.shape)
            y = y_raw[mask]
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue

            x = np.arange(y.size, dtype=np.int64)
            fig, ax = plt.subplots(figsize=(7, 4))

            if color_feature and color_feature in handle:
                c_raw = np.asarray(handle[color_feature][event_idx]).reshape(-1)
                c = c_raw[mask]
                c = c[: y.size]
                sc = ax.scatter(x, y, c=c, s=8, cmap="viridis", alpha=0.8)
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(color_feature)
            elif use_line:
                ax.plot(x, y, linewidth=1.0)
            else:
                ax.scatter(x, y, s=8, alpha=0.8)

            ax.set_xlabel("hit index (stored order)")
            ax.set_ylabel(y_feature)
            ax.set_title(f"{path.stem} event={event_idx}")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()

            out_path = out_dir / f"{path.stem}_event{event_idx}_{y_feature.replace('.', '_')}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved += 1

            # Zoomed view around requested ordering index.
            if focus_window >= 0 and focus_index >= 0 and y.size > focus_index:
                z_start = max(0, focus_index - focus_window)
                z_stop = min(y.size, focus_index + focus_window + 1)
                z_x = x[z_start:z_stop]
                z_y = y[z_start:z_stop]
                if z_y.size > 0:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    if use_line:
                        ax.plot(z_x, z_y, linewidth=1.0)
                    else:
                        ax.scatter(z_x, z_y, s=10, alpha=0.85)
                    ax.axvline(focus_index, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
                    ax.set_xlabel("hit index (stored order)")
                    ax.set_ylabel(y_feature)
                    ax.set_title(
                        f"{path.stem} event={event_idx} zoom@{focus_index} +- {focus_window}"
                    )
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    zoom_out = out_dir / (
                        f"{path.stem}_event{event_idx}_{y_feature.replace('.', '_')}"
                        f"_zoom_idx{focus_index}_w{focus_window}.png"
                    )
                    fig.savefig(zoom_out, dpi=150)
                    plt.close(fig)
                    saved += 1

    return saved


def _get_event_feature(handle: h5py.File, feature: str, event_idx: int) -> np.ndarray:
    arr = np.asarray(handle[feature][event_idx])
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def plot_event_waveforms(
    path: Path,
    *,
    mask_feature: str,
    num_events: int,
    max_events: int,
    out_dir: Path,
    amp_feature: str,
    time_feature: str,
    time_end_feature: str,
    cellid_feature: str,
    cells_per_event: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with h5py.File(path, "r") as handle:
        required = [amp_feature, time_feature, time_end_feature, cellid_feature]
        missing = [name for name in required if name not in handle]
        if missing:
            print(f"Skipping waveform plots for {path}: missing {', '.join(missing)}")
            return 0

        n_events = handle[amp_feature].shape[0]
        n_take = min(max(num_events, 0), n_events)
        if max_events > 0:
            n_take = min(n_take, max_events)
        if n_take == 0:
            return 0

        event_iter = tqdm(range(n_take), desc=f"waveform events {path.stem}", leave=False)
        for event_idx in event_iter:
            amp = _get_event_feature(handle, amp_feature, event_idx)
            t0 = _get_event_feature(handle, time_feature, event_idx)
            t1 = _get_event_feature(handle, time_end_feature, event_idx)
            cell = _get_event_feature(handle, cellid_feature, event_idx)
            if not (amp.shape == t0.shape == t1.shape == cell.shape):
                continue

            mask = _valid_mask(handle, mask_feature, event_idx, amp.shape)
            finite = np.isfinite(amp) & np.isfinite(t0) & np.isfinite(t1) & np.isfinite(cell)
            valid = mask & finite
            if not np.any(valid):
                continue

            amp = amp[valid]
            t0 = t0[valid]
            t1 = t1[valid]
            cell = cell[valid].astype(np.int64, copy=False)
            t1 = np.maximum(t1, t0)

            fig, ax = plt.subplots(figsize=(8, 4))

            unique_cells, inverse = np.unique(cell, return_inverse=True)
            total_amp = np.bincount(inverse, weights=amp, minlength=unique_cells.size)
            cell_order = np.argsort(total_amp)[::-1]
            if cells_per_event > 0:
                cell_order = cell_order[:cells_per_event]

            for rank, cell_pos in enumerate(cell_order):
                sel = inverse == cell_pos
                ct0 = t0[sel]
                ct1 = t1[sel]
                camp = amp[sel]
                if camp.size == 0:
                    continue

                order = np.argsort(ct0, kind="stable")
                ct0 = ct0[order]
                ct1 = ct1[order]
                camp = camp[order]

                # Connected waveform for one readout channel (one cellID).
                x_line: List[float] = []
                y_line: List[float] = []
                for i in range(camp.size):
                    x_line.append(float(ct0[i]))
                    y_line.append(float(camp[i]))
                    x_line.append(float(ct1[i]))
                    y_line.append(float(camp[i]))

                cid = int(unique_cells[cell_pos])
                ax.plot(
                    x_line,
                    y_line,
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"cell {cid}" if rank < 10 else None,
                )

            ax.set_xlabel(time_feature)
            ax.set_ylabel(amp_feature)
            ax.set_title(f"{path.stem} event={event_idx} waveform by cellID")
            ax.grid(True, linestyle="--", alpha=0.3)
            if cell_order.size > 0:
                ax.legend(loc="best", fontsize=7, ncol=2)
            fig.tight_layout()

            out_path = out_dir / (
                f"{path.stem}_event{event_idx}_waveform_"
                f"{amp_feature.replace('.', '_')}_"
                f"{time_feature.replace('.', '_')}_"
                f"{time_end_feature.replace('.', '_')}_"
                f"{cellid_feature.replace('.', '_')}.png"
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved += 1

    return saved


def plot_event_linked_waveforms(
    path: Path,
    *,
    mask_feature: str,
    num_events: int,
    max_events: int,
    out_dir: Path,
    amp_feature: str,
    time_feature: str,
    time_end_feature: str,
    linked_gap: float,
) -> int:
    """Plot waveform0->waveform1->... using original point-cloud ordering."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with h5py.File(path, "r") as handle:
        required = [amp_feature, time_feature, time_end_feature]
        missing = [name for name in required if name not in handle]
        if missing:
            print(f"Skipping linked-waveform plots for {path}: missing {', '.join(missing)}")
            return 0

        n_events = handle[amp_feature].shape[0]
        n_take = min(max(num_events, 0), n_events)
        if max_events > 0:
            n_take = min(n_take, max_events)
        if n_take == 0:
            return 0

        event_iter = tqdm(range(n_take), desc=f"linked events {path.stem}", leave=False)
        for event_idx in event_iter:
            amp = _get_event_feature(handle, amp_feature, event_idx)
            t0 = _get_event_feature(handle, time_feature, event_idx)
            t1 = _get_event_feature(handle, time_end_feature, event_idx)
            if not (amp.shape == t0.shape == t1.shape):
                continue

            mask = _valid_mask(handle, mask_feature, event_idx, amp.shape)
            finite = np.isfinite(amp) & np.isfinite(t0) & np.isfinite(t1)
            valid = mask & finite
            if not np.any(valid):
                continue

            # Keep original point-cloud order; no sorting by time.
            amp = amp[valid]
            t0 = t0[valid]
            t1 = t1[valid]

            duration = np.maximum(t1 - t0, 0.0)
            # Avoid invisible segments for zero-duration entries.
            duration = np.where(duration > 0.0, duration, 1e-6)

            x_line: List[float] = []
            y_line: List[float] = []
            cursor = 0.0
            gap = max(float(linked_gap), 0.0)
            for a, d in zip(amp, duration):
                start = cursor
                end = cursor + float(d)
                x_line.extend([start, end])
                y_line.extend([float(a), float(a)])
                cursor = end + gap

            if not x_line:
                continue

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x_line, y_line, linewidth=1.0, alpha=0.95)
            ax.plot(x_line[:1000], y_line[:1000], linewidth=1.0, alpha=0.95)
            ax.set_xlabel("stacked time (point-cloud order)")
            ax.set_ylabel(amp_feature)
            ax.set_title(f"{path.stem} event={event_idx} linked waveform (input order)")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()

            out_path = out_dir / (
                f"{path.stem}_event{event_idx}_linked_waveform_"
                f"{amp_feature.replace('.', '_')}_"
                f"{time_feature.replace('.', '_')}_"
                f"{time_end_feature.replace('.', '_')}.png"
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved += 1

    return saved


def main() -> None:
    args = parse_args()
    files = [p for p in expand_paths(args.files) if p.exists()]
    if not files:
        raise SystemExit("No input files found.")

    plot_dir = Path(args.plot_dir)
    total = 0
    plot_file_iter = tqdm(files, desc="plot files")
    for path in plot_file_iter:
        total += plot_event_ordering(
            path,
            y_feature=args.y_feature,
            color_feature=args.color_feature,
            mask_feature=args.mask_feature,
            num_events=args.plot_events,
            max_events=args.max_events,
            out_dir=plot_dir,
            use_line=args.line,
            focus_index=args.focus_index,
            focus_window=args.focus_window,
        )
    if args.waveform:
        wave_iter = tqdm(files, desc="waveform files")
        for path in wave_iter:
            total += plot_event_waveforms(
                path,
                mask_feature=args.mask_feature,
                num_events=args.plot_events,
                max_events=args.max_events,
                out_dir=plot_dir,
                amp_feature=args.wave_amp_feature,
                time_feature=args.wave_time_feature,
                time_end_feature=args.wave_time_end_feature,
                cellid_feature=args.wave_cellid_feature,
                cells_per_event=args.wave_cells_per_event,
            )
    if args.linked_waveform:
        linked_iter = tqdm(files, desc="linked files")
        for path in linked_iter:
            total += plot_event_linked_waveforms(
                path,
                mask_feature=args.mask_feature,
                num_events=args.plot_events,
                max_events=args.max_events,
                out_dir=plot_dir,
                amp_feature=args.wave_amp_feature,
                time_feature=args.wave_time_feature,
                time_end_feature=args.wave_time_end_feature,
                linked_gap=args.linked_gap,
            )
    print(f"Saved {total} plot(s) to {plot_dir}")


if __name__ == "__main__":
    main()
