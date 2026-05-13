#!/usr/bin/env python
"""Extract and inspect outlier events directly from run outputs and HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EVENT_VARIABLE_KEYS = {
    "E_gen": "E_gen",
    "E_dep": "E_dep",
    "E_leak": "E_leak",
    "S_amp": "S_amp",
    "C_amp": "C_amp",
    "theta": "GenParticles.momentum.theta",
    "phi": "GenParticles.momentum.phi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and inspect outlier events from run.py outputs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing config.json and output.json.")
    parser.add_argument("--outliers-json", type=Path, default=None, help="Defaults to <run-dir>/outliers.json.")
    parser.add_argument("--output-json", type=Path, default=None, help="Defaults to <run-dir>/output.json.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to <run-dir>/inspected_outliers.")
    parser.add_argument("--min-abs-error", type=float, default=50.0, help="Minimum absolute energy error for outlier extraction.")
    parser.add_argument("--top-n", type=int, default=100, help="Maximum number of extracted outliers to keep.")
    parser.add_argument("--refresh-outliers", action="store_true", help="Recompute outliers.json from output.json even if it already exists.")
    parser.add_argument("--extract-only", action="store_true", help="Only extract outliers.json and skip per-event inspection.")
    parser.add_argument("--start", type=int, default=0, help="Starting outlier rank.")
    parser.add_argument("--count", type=int, default=5, help="Number of outliers to inspect.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _extract_outliers(records: list[dict[str, Any]], min_abs_error: float, top_n: int) -> list[dict[str, Any]]:
    outliers: list[dict[str, Any]] = []
    for record in records:
        energy_pred = float(record["energy_pred"])
        energy_true = float(record["energy_true"])
        abs_error = abs(energy_pred - energy_true)
        if abs_error < min_abs_error:
            continue
        enriched = dict(record)
        enriched["abs_error"] = abs_error
        enriched["signed_error"] = energy_pred - energy_true
        enriched["squared_error"] = abs_error * abs_error
        enriched["relative_error"] = (energy_pred - energy_true) / max(abs(energy_true), 1e-6)
        outliers.append(enriched)
    outliers.sort(key=lambda item: float(item["abs_error"]), reverse=True)
    if top_n > 0:
        outliers = outliers[:top_n]
    return outliers


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    alt = Path("/store/ml/dual-readout/h5s") / path.name
    if alt.exists():
        return alt
    alt_local = Path("h5s") / path.name
    if alt_local.exists():
        return alt_local
    return path


def _load_feature_max(stat_file: Path, hit_features: list[str]) -> dict[str, float]:
    import yaml

    if not stat_file.exists():
        return {feature: 1.0 for feature in hit_features}
    payload = yaml.safe_load(stat_file.read_text())
    feature_max: dict[str, float] = {}
    for feature in hit_features:
        stats = payload.get(feature, {}) if isinstance(payload, dict) else {}
        max_value = None
        if isinstance(stats, dict):
            max_value = stats.get("max")
        if max_value is None or not np.isfinite(max_value) or float(max_value) == 0.0:
            feature_max[feature] = 1.0
        else:
            feature_max[feature] = float(max_value)
    return feature_max


def _load_event_points(
    file_path: Path,
    event_id: int,
    hit_features: list[str],
    feature_max: dict[str, float],
    max_points: int | None,
) -> np.ndarray:
    import h5py

    cols = []
    with h5py.File(file_path, "r") as handle:
        for feature in hit_features:
            data = np.asarray(handle[feature][event_id], dtype=np.float32)
            if max_points is not None:
                data = data[:max_points]
            cols.append(data / feature_max.get(feature, 1.0))
    return np.stack(cols, axis=-1)


def _read_event_variables(file_path: Path, event_id: int) -> dict[str, float | None]:
    import h5py

    event_vars: dict[str, float | None] = {}
    with h5py.File(file_path, "r") as handle:
        for out_key, source_key in EVENT_VARIABLE_KEYS.items():
            if source_key not in handle:
                event_vars[out_key] = None
                continue
            value = np.asarray(handle[source_key][event_id]).reshape(-1)
            if value.size == 0:
                event_vars[out_key] = None
                continue
            scalar = float(value[0])
            event_vars[out_key] = scalar if np.isfinite(scalar) else None
    return event_vars


def _save_hit_index_plot(points: np.ndarray, feature_to_index: dict[str, int], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("DRcalo3dHits.amplitude_sum", "amplitude_sum"),
        ("DRcalo3dHits.time", "time"),
        ("DRcalo3dHits.time_end", "time_end"),
    ]
    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(10, 9), sharex=True)
    x = np.arange(points.shape[0], dtype=np.int64)
    for ax, (feature_name, label) in zip(axes, plot_specs):
        feature_idx = feature_to_index.get(feature_name)
        if feature_idx is None:
            ax.text(0.5, 0.5, f"missing {label}", transform=ax.transAxes, ha="center", va="center")
        else:
            ax.plot(x, points[:, feature_idx], linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("hit index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_position_plot(points: np.ndarray, feature_to_index: dict[str, int], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_idx = feature_to_index.get("DRcalo3dHits.position.x")
    y_idx = feature_to_index.get("DRcalo3dHits.position.y")
    amp_idx = feature_to_index.get("DRcalo3dHits.amplitude_sum")
    if x_idx is None or y_idx is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = points[:, amp_idx] if amp_idx is not None else None
    if colors is None:
        ax.scatter(points[:, x_idx], points[:, y_idx], s=8, alpha=0.7)
    else:
        sc = ax.scatter(points[:, x_idx], points[:, y_idx], c=colors, s=8, alpha=0.7, cmap="viridis")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("amplitude_sum")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_waveform_plot(points: np.ndarray, feature_to_index: dict[str, int], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amp_idx = feature_to_index.get("DRcalo3dHits.amplitude_sum")
    time_idx = feature_to_index.get("DRcalo3dHits.time")
    if amp_idx is None or time_idx is None:
        return
    cid_idx = feature_to_index.get("DRcalo3dHits.cellID")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = points[:, cid_idx] if cid_idx is not None else None
    sc = ax.scatter(points[:, time_idx], points[:, amp_idx], c=colors, s=10, alpha=0.7, cmap="tab20")
    if colors is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("cellID")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude_sum")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    config = _load_json(run_dir / "config.json")
    outliers_path = args.outliers_json or (run_dir / "outliers.json")
    output_path = args.output_json or (run_dir / "output.json")
    outputs = _load_json(output_path)
    if args.refresh_outliers or not outliers_path.exists():
        outliers = _extract_outliers(outputs, min_abs_error=args.min_abs_error, top_n=args.top_n)
        outliers_path.parent.mkdir(parents=True, exist_ok=True)
        outliers_path.write_text(json.dumps(outliers, indent=2))
        print(f"extracted_outliers: {len(outliers)}")
        print(f"outliers_json: {outliers_path}")
    else:
        outliers = _load_json(outliers_path)
        print(f"using existing outliers_json: {outliers_path}")
        print(f"loaded_outliers: {len(outliers)}")
    if args.extract_only:
        return
    output_map = {tuple(record["event_id"]): record for record in outputs}

    hit_features = list(config["hit_features"])
    feature_to_index = {name: idx for idx, name in enumerate(hit_features)}
    stat_file = _resolve_path(config["stat_file"])
    feature_max = _load_feature_max(stat_file, hit_features)
    train_files = [_resolve_path(path) for path in config["train_files"]]
    max_points = config.get("max_points")

    out_dir = args.out_dir or (run_dir / "inspected_outliers")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = outliers[args.start: args.start + args.count]
    for rank_offset, outlier in enumerate(selected, start=args.start):
        file_id, event_id = int(outlier["event_id"][0]), int(outlier["event_id"][1])
        file_path = train_files[file_id]
        points = _load_event_points(file_path, event_id, hit_features, feature_max, max_points)
        event_dir = out_dir / f"rank_{rank_offset:03d}_file{file_id}_event{event_id}"
        event_dir.mkdir(parents=True, exist_ok=True)

        metadata = dict(outlier)
        metadata["file_path"] = str(file_path)
        metadata["num_points_loaded"] = int(points.shape[0])
        metadata["num_features"] = int(points.shape[1])
        metadata["event_variables"] = _read_event_variables(file_path, event_id)
        metadata["output_record"] = output_map.get((file_id, event_id))
        (event_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        np.savez_compressed(event_dir / "points.npz", points=points, hit_features=np.asarray(hit_features, dtype=object))

        title = (
            f"rank={rank_offset} file={file_id} event={event_id} "
            f"true={float(outlier['energy_true']):.4f} pred={float(outlier['energy_pred']):.4f}"
        )
        _save_hit_index_plot(points, feature_to_index, event_dir / "hit_index.png", title)
        _save_position_plot(points, feature_to_index, event_dir / "xy.png", title)
        _save_waveform_plot(points, feature_to_index, event_dir / "waveform.png", title)
        print(f"saved {event_dir}")


if __name__ == "__main__":
    main()
