#!/usr/bin/env python
"""Analyze distribution drift across event-index ranges inside HDF5 files.

Example
-------
python analyze_event_index_bias.py \
  --files pi0_1-120GeV.h5py \
  --properties theta phi e_gen e_dep e_leak s_amp c_amp num_hits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from tqdm.auto import tqdm


ALIASES: Dict[str, str] = {
    "theta": "GenParticles.momentum.theta",
    "phi": "GenParticles.momentum.phi",
    "e_gen": "E_gen",
    "e_dep": "E_dep",
    "e_leak": "E_leak",
    "s_amp": "S_amp",
    "c_amp": "C_amp",
    "num_hits": "num_hits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prefix/suffix drift inside event files.")
    parser.add_argument("--files", nargs="+", required=True, help="Input files or globs.")
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["theta", "phi", "e_gen", "e_dep", "e_leak", "s_amp", "c_amp", "num_hits"],
        help="Scalar properties or aliases to compare across event-index segments.",
    )
    parser.add_argument("--segments", type=int, default=3, help="Number of equal event-index segments per file.")
    parser.add_argument("--max-events", type=int, default=0, help="Per-file event cap. <=0 means all events.")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bins for saved plots.")
    parser.add_argument("--mask-feature", default="DRcalo3dHits.amplitude_sum", help="Feature used to count valid hits.")
    parser.add_argument(
        "--out-dir",
        default="analysis/event_index_bias",
        help="Output directory for JSON summaries and plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip histogram/trend plot generation and only write JSON/text summary.",
    )
    return parser.parse_args()


def _canonical_key(name: str) -> str:
    return ALIASES.get(name.strip().lower(), name)


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


def _event_count(handle: Any) -> int:
    candidates = [
        "GenParticles.PDG",
        "E_gen",
        "E_dep",
        "GenParticles.momentum.theta",
        "S_amp",
    ]
    for key in candidates:
        if key in handle and handle[key].ndim >= 1:
            return int(handle[key].shape[0])
    raise KeyError("Could not infer event count from known event-level datasets")


def _read_scalar(handle: Any, key: str, start: int, stop: int) -> np.ndarray:
    arr = np.asarray(handle[key][start:stop], dtype=np.float64)
    return arr.reshape(-1)


def _read_num_hits(handle: Any, key: str, start: int, stop: int) -> np.ndarray:
    if key not in handle:
        return np.full(stop - start, np.nan, dtype=np.float64)
    arr = np.asarray(handle[key][start:stop])
    if arr.ndim == 1:
        valid = np.isfinite(arr) & (arr != 0)
        return valid.astype(np.float64)
    valid = np.isfinite(arr) & (arr != 0)
    return np.sum(valid, axis=1, dtype=np.int64).astype(np.float64)


def _segment_bounds(total: int, segments: int) -> List[tuple[int, int]]:
    bounds: List[tuple[int, int]] = []
    for i in range(segments):
        start = int(np.floor(total * i / segments))
        stop = int(np.floor(total * (i + 1) / segments))
        bounds.append((start, stop))
    return bounds


def _summary_stats(values: np.ndarray) -> Dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "p05": None, "p95": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def _std_mean_diff(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None
    pooled = np.sqrt(0.5 * (np.var(a) + np.var(b)))
    if not np.isfinite(pooled) or pooled == 0.0:
        return None
    return float((np.mean(b) - np.mean(a)) / pooled)


def _logreg_separability(
    feature_map: Dict[str, tuple[np.ndarray, np.ndarray]],
    first_label: int,
    last_label: int,
) -> Dict[str, float | None]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return {"logreg_accuracy": None, "logreg_auc": None}

    names = sorted(feature_map)
    first_len = int(feature_map[names[0]][0].shape[0])
    last_len = int(feature_map[names[0]][1].shape[0])
    cols = []
    for name in names:
        first_values, last_values = feature_map[name]
        cols.append(np.concatenate([first_values.reshape(-1), last_values.reshape(-1)]).reshape(-1, 1))
    x_all = np.concatenate(cols, axis=1)
    valid = np.all(np.isfinite(x_all), axis=1)
    x_all = x_all[valid]
    y_all = np.concatenate(
        [
            np.full(first_len, first_label, dtype=np.int64),
            np.full(last_len, last_label, dtype=np.int64),
        ]
    )
    y_all = y_all[valid]
    if x_all.shape[0] < 20 or np.unique(y_all).size < 2:
        return {"logreg_accuracy": None, "logreg_auc": None}
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.3,
        random_state=1234,
        stratify=y_all,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(np.int64)
    return {
        "logreg_accuracy": float(accuracy_score(y_test, y_pred)),
        "logreg_auc": float(roc_auc_score(y_test, y_score)),
    }


def _plot_feature_segments(
    file_label: str,
    feature_name: str,
    segment_values: Sequence[np.ndarray],
    out_dir: Path,
    bins: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, values in enumerate(segment_values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        ax.hist(finite, bins=bins, histtype="step", linewidth=1.3, density=True, label=f"seg{idx}")
    ax.set_title(f"{file_label}: {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("density")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{Path(file_label).stem}_{feature_name.replace('.', '_')}_segments.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_trend(
    file_label: str,
    feature_name: str,
    segment_values: Sequence[np.ndarray],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    medians = []
    means = []
    for values in segment_values:
        finite = values[np.isfinite(values)]
        medians.append(float(np.median(finite)) if finite.size else np.nan)
        means.append(float(np.mean(finite)) if finite.size else np.nan)
    x = np.arange(len(segment_values))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, means, marker="o", label="mean")
    ax.plot(x, medians, marker="s", label="median")
    ax.set_title(f"{file_label}: {feature_name} trend")
    ax.set_xlabel("segment index")
    ax.set_ylabel(feature_name)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{Path(file_label).stem}_{feature_name.replace('.', '_')}_trend.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import h5py

    files = _resolve_paths(args.files)
    if not files:
        raise SystemExit("No input files found.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    properties = [_canonical_key(name) for name in args.properties]
    summary: Dict[str, Any] = {"files": {}}

    for file_path in tqdm(files, desc="files"):
        file_out_dir = out_dir / Path(file_path).stem
        file_out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(file_path, "r") as handle:
            total_events = _event_count(handle)
            take = total_events if args.max_events <= 0 else min(total_events, args.max_events)
            bounds = _segment_bounds(take, args.segments)
            file_summary: Dict[str, Any] = {
                "path": str(file_path),
                "available_events": total_events,
                "used_events": take,
                "segments": [],
                "separability": {},
            }
            property_segments: Dict[str, List[np.ndarray]] = {prop: [] for prop in properties}
            for seg_idx, (start, stop) in enumerate(bounds):
                seg_info: Dict[str, Any] = {
                    "segment_index": seg_idx,
                    "start_event": start,
                    "stop_event": stop,
                    "size": stop - start,
                    "properties": {},
                }
                for prop in properties:
                    if prop == "num_hits":
                        values = _read_num_hits(handle, args.mask_feature, start, stop)
                    elif prop in handle:
                        values = _read_scalar(handle, prop, start, stop)
                    else:
                        values = np.full(stop - start, np.nan, dtype=np.float64)
                    property_segments[prop].append(values)
                    seg_info["properties"][prop] = _summary_stats(values)
                file_summary["segments"].append(seg_info)

            first_last_features: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for prop, seg_values in property_segments.items():
                first = seg_values[0]
                last = seg_values[-1]
                smd = _std_mean_diff(first, last)
                file_summary["separability"][prop] = {"std_mean_diff_first_vs_last": smd}
                first_last_features[prop] = (first.reshape(-1), last.reshape(-1))
                if not args.skip_plots:
                    _plot_feature_segments(file_path.name, prop, seg_values, file_out_dir, args.bins)
                    _plot_feature_trend(file_path.name, prop, seg_values, file_out_dir)

            sep_metrics = _logreg_separability(first_last_features, 0, 1)
            file_summary["separability"]["combined_first_vs_last"] = sep_metrics
            summary["files"][file_path.name] = file_summary

            print(f"\n{file_path.name}")
            print(f"  events: used={take:,} available={total_events:,}")
            combined = sep_metrics
            print(
                "  first-vs-last separability: "
                f"logreg_acc={combined['logreg_accuracy']} logreg_auc={combined['logreg_auc']}"
            )
            ranked = []
            for prop, info in file_summary["separability"].items():
                if prop == "combined_first_vs_last":
                    continue
                score = info["std_mean_diff_first_vs_last"]
                ranked.append((abs(score) if score is not None else -1.0, prop, score))
            ranked.sort(reverse=True)
            for _, prop, score in ranked[:5]:
                print(f"    {prop}: std_mean_diff_first_vs_last={score}")

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary_json: {out_path}")


if __name__ == "__main__":
    main()
