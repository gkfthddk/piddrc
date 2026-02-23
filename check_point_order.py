"""Visualize point-cloud ordering from collated batches used before training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pid import DualReadoutEventDataset, collate_events


@dataclass
class PreparedEvent:
    event_idx: int
    points: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect pre-training (collated) point-cloud ordering.")
    parser.add_argument("--files", nargs="+", default=["h5s/*_1-100GeV_10.h5py"], help="Input file names or globs.")
    parser.add_argument("--stat-file", default="stats_1-100GeV.yaml", help="Stats YAML used by pid.data normalization.")
    parser.add_argument("--label-key", default="GenParticles.PDG", help="Label dataset name.")
    parser.add_argument("--energy-key", default="E_gen", help="Energy dataset name.")
    parser.add_argument("--pool", type=int, default=1, help="Pooling factor forwarded to DualReadoutEventDataset.")
    parser.add_argument("--max-points", type=int, default=500, help="Per-event point cap (same as run.py).")
    parser.add_argument("--max-events", type=int, default=100, help="Max events per file to scan (<=0 means all).")
    parser.add_argument("--plot-events", type=int, default=3, help="Number of events per file to plot.")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size used before plotting.")
    parser.add_argument("--mask-feature", default="DRcalo3dHits.amplitude_sum", help="Feature used by dataset summary/filter path.")
    parser.add_argument("--is-cherenkov-feature", default="DRcalo3dHits.type", help="Feature used by dataset summary function.")
    parser.add_argument("--y-feature", default="DRcalo3dHits.amplitude_sum", help="Feature for y-axis in ordering plots.")
    parser.add_argument("--color-feature", default=None, help="Optional feature used as color map in scatter plot.")
    parser.add_argument("--plot-dir", default="plots/point_order", help="Directory for output plots.")
    parser.add_argument("--line", action="store_true", help="Use line plot instead of scatter.")
    parser.add_argument("--focus-index", type=int, default=1000, help="Center hit index for local checks.")
    parser.add_argument("--focus-window", type=int, default=100, help="Half-window around focus index.")
    parser.add_argument("--waveform", action="store_true", help="Also create single-readout waveform plots.")
    parser.add_argument("--wave-amp-feature", default="DRcalo3dHits.amplitude", help="Amplitude feature for waveform y-axis.")
    parser.add_argument("--wave-time-feature", default="DRcalo3dHits.time", help="Start-time feature for waveform x-axis.")
    parser.add_argument("--wave-time-end-feature", default="DRcalo3dHits.time_end", help="End-time feature for waveform x-axis.")
    parser.add_argument("--wave-cellid-feature", default="DRcalo3dHits.cellID", help="CellID feature used to build waveforms.")
    parser.add_argument("--wave-cells-per-event", type=int, default=6, help="Number of cellIDs to draw per event.")
    parser.add_argument("--linked-waveform", action="store_true", help="Plot linked waveforms in collated order.")
    parser.add_argument("--linked-gap", type=float, default=0.0, help="Optional gap on stacked time axis.")
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


def _dataset_file_arg(path: Path) -> str:
    # pid.data prefixes files with /store/ml/dual-readout/h5s, so pass basename.
    return path.name


def _required_hit_features(args: argparse.Namespace) -> List[str]:
    names: List[str] = [
        args.mask_feature,
        args.is_cherenkov_feature,
        args.y_feature,
        args.wave_amp_feature,
        args.wave_time_feature,
        args.wave_time_end_feature,
        args.wave_cellid_feature,
    ]
    if args.color_feature:
        names.append(args.color_feature)
    return list(dict.fromkeys(names))


def _build_dataset(path: Path, args: argparse.Namespace) -> DualReadoutEventDataset:
    return DualReadoutEventDataset(
        [_dataset_file_arg(path)],
        hit_features=_required_hit_features(args),
        pos_keys=None,
        label_key=args.label_key,
        energy_key=args.energy_key,
        stat_file=args.stat_file,
        max_points=args.max_points,
        pool=args.pool,
        is_cherenkov_key=args.is_cherenkov_feature,
        amp_sum_key=args.mask_feature,
        cid_key=args.wave_cellid_feature,
        max_events=(args.max_events if args.max_events > 0 else None),
        progress=False,
    )


def _collect_collated_events(dataset: DualReadoutEventDataset, n_take: int, batch_size: int) -> List[PreparedEvent]:
    if n_take <= 0:
        return []
    take_batch_size = max(1, min(batch_size, n_take))
    loader = DataLoader(dataset, batch_size=take_batch_size, shuffle=False, num_workers=0, collate_fn=collate_events)

    events: List[PreparedEvent] = []
    for batch in loader:
        points = batch["points"].detach().cpu().numpy()
        mask = batch["mask"].detach().cpu().numpy()
        event_id = batch["event_id"].detach().cpu().numpy()
        for i in range(points.shape[0]):
            valid = mask[i]
            trimmed_points = points[i, valid]
            events.append(PreparedEvent(event_idx=int(event_id[i, 1]), points=trimmed_points))
            if len(events) >= n_take:
                return events
    return events


def plot_event_ordering(
    *,
    events: List[PreparedEvent],
    feature_to_index: dict[str, int],
    file_label: str,
    y_feature: str,
    color_feature: str | None,
    out_dir: Path,
    use_line: bool,
    focus_index: int,
    focus_window: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    if y_feature not in feature_to_index:
        print(f"Skipping plots for {file_label}: missing {y_feature}")
        return 0

    y_idx = feature_to_index[y_feature]
    color_idx = feature_to_index.get(color_feature) if color_feature else None

    event_iter = tqdm(events, total=len(events), desc=f"plot events {file_label}", leave=False)
    for event in event_iter:
        y_raw = event.points[:, y_idx]
        finite = np.isfinite(y_raw)
        y = y_raw[finite]
        if y.size == 0:
            continue

        x = np.arange(y.size, dtype=np.int64)
        fig, ax = plt.subplots(figsize=(7, 4))

        if color_idx is not None:
            c_raw = event.points[:, color_idx]
            c = c_raw[finite]
            sc = ax.scatter(x, y, c=c, s=8, cmap="viridis", alpha=0.8)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(color_feature)
        elif use_line:
            ax.plot(x, y, linewidth=1.0)
        else:
            ax.scatter(x, y, s=8, alpha=0.8)

        ax.set_xlabel("hit index (collated order)")
        ax.set_ylabel(y_feature)
        ax.set_title(f"{file_label} event={event.event_idx}")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"{Path(file_label).stem}_event{event.event_idx}_{y_feature.replace('.', '_')}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved += 1

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
                ax.set_xlabel("hit index (collated order)")
                ax.set_ylabel(y_feature)
                ax.set_title(f"{file_label} event={event.event_idx} zoom@{focus_index} +- {focus_window}")
                ax.grid(True, linestyle="--", alpha=0.3)
                fig.tight_layout()
                zoom_out = out_dir / (
                    f"{Path(file_label).stem}_event{event.event_idx}_{y_feature.replace('.', '_')}"
                    f"_zoom_idx{focus_index}_w{focus_window}.png"
                )
                fig.savefig(zoom_out, dpi=150)
                plt.close(fig)
                saved += 1

    return saved


def plot_event_waveforms(
    *,
    events: List[PreparedEvent],
    feature_to_index: dict[str, int],
    file_label: str,
    out_dir: Path,
    amp_feature: str,
    time_feature: str,
    time_end_feature: str,
    cellid_feature: str,
    cells_per_event: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    required = [amp_feature, time_feature, time_end_feature, cellid_feature]
    missing = [name for name in required if name not in feature_to_index]
    if missing:
        print(f"Skipping waveform plots for {file_label}: missing {', '.join(missing)}")
        return 0

    amp_idx = feature_to_index[amp_feature]
    t0_idx = feature_to_index[time_feature]
    t1_idx = feature_to_index[time_end_feature]
    cid_idx = feature_to_index[cellid_feature]

    event_iter = tqdm(events, total=len(events), desc=f"waveform events {file_label}", leave=False)
    for event in event_iter:
        amp = event.points[:, amp_idx]
        t0 = event.points[:, t0_idx]
        t1 = event.points[:, t1_idx]
        cell = event.points[:, cid_idx]

        finite = np.isfinite(amp) & np.isfinite(t0) & np.isfinite(t1) & np.isfinite(cell)
        if not np.any(finite):
            continue

        amp = amp[finite]
        t0 = t0[finite]
        t1 = t1[finite]
        cell = cell[finite].astype(np.int64, copy=False)
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

            x_line: List[float] = []
            y_line: List[float] = []
            for j in range(camp.size):
                x_line.append(float(ct0[j]))
                y_line.append(float(camp[j]))
                x_line.append(float(ct1[j]))
                y_line.append(float(camp[j]))

            cid = int(unique_cells[cell_pos])
            ax.plot(x_line, y_line, linewidth=1.0, alpha=0.9, label=f"cell {cid}" if rank < 10 else None)

        ax.set_xlabel(time_feature)
        ax.set_ylabel(amp_feature)
        ax.set_title(f"{file_label} event={event.event_idx} waveform by cellID")
        ax.grid(True, linestyle="--", alpha=0.3)
        if cell_order.size > 0:
            ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()

        out_path = out_dir / (
            f"{Path(file_label).stem}_event{event.event_idx}_waveform_"
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
    *,
    events: List[PreparedEvent],
    feature_to_index: dict[str, int],
    file_label: str,
    out_dir: Path,
    amp_feature: str,
    time_feature: str,
    time_end_feature: str,
    linked_gap: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    required = [amp_feature, time_feature, time_end_feature]
    missing = [name for name in required if name not in feature_to_index]
    if missing:
        print(f"Skipping linked-waveform plots for {file_label}: missing {', '.join(missing)}")
        return 0

    amp_idx = feature_to_index[amp_feature]
    t0_idx = feature_to_index[time_feature]
    t1_idx = feature_to_index[time_end_feature]

    event_iter = tqdm(events, total=len(events), desc=f"linked events {file_label}", leave=False)
    for event in event_iter:
        amp = event.points[:, amp_idx]
        t0 = event.points[:, t0_idx]
        t1 = event.points[:, t1_idx]

        finite = np.isfinite(amp) & np.isfinite(t0) & np.isfinite(t1)
        if not np.any(finite):
            continue

        amp = amp[finite]
        t0 = t0[finite]
        t1 = t1[finite]

        duration = np.maximum(t1 - t0, 0.0)
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
        ax.set_xlabel("stacked time (collated order)")
        ax.set_ylabel(amp_feature)
        ax.set_title(f"{file_label} event={event.event_idx} linked waveform (input order)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / (
            f"{Path(file_label).stem}_event{event.event_idx}_linked_waveform_"
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
    files = expand_paths(args.files)
    if not files:
        raise SystemExit("No input files found.")
    plot_dir = Path(args.plot_dir)
    total = 0

    file_iter = tqdm(files, desc="files")
    for path in file_iter:
        dataset = _build_dataset(path, args)
        label = path.name
        try:
            n_take = min(max(args.plot_events, 0), len(dataset))
            events = _collect_collated_events(dataset, n_take=n_take, batch_size=args.batch_size)
            if not events:
                continue

            total += plot_event_ordering(
                events=events,
                feature_to_index=dataset.feature_to_index,
                file_label=label,
                y_feature=args.y_feature,
                color_feature=args.color_feature,
                out_dir=plot_dir,
                use_line=args.line,
                focus_index=args.focus_index,
                focus_window=args.focus_window,
            )

            if args.waveform:
                total += plot_event_waveforms(
                    events=events,
                    feature_to_index=dataset.feature_to_index,
                    file_label=label,
                    out_dir=plot_dir,
                    amp_feature=args.wave_amp_feature,
                    time_feature=args.wave_time_feature,
                    time_end_feature=args.wave_time_end_feature,
                    cellid_feature=args.wave_cellid_feature,
                    cells_per_event=args.wave_cells_per_event,
                )

            if args.linked_waveform:
                total += plot_event_linked_waveforms(
                    events=events,
                    feature_to_index=dataset.feature_to_index,
                    file_label=label,
                    out_dir=plot_dir,
                    amp_feature=args.wave_amp_feature,
                    time_feature=args.wave_time_feature,
                    time_end_feature=args.wave_time_end_feature,
                    linked_gap=args.linked_gap,
                )
        finally:
            dataset.close()

    print(f"Saved {total} plot(s) to {plot_dir}")


if __name__ == "__main__":
    main()
