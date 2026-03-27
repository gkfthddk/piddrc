"""Compare performance metrics across specific saved run directories.

Example:
  python plot_run_comparison.py --run-dirs save/test2 save/test3 save/test4 save/test5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from utils.plot_helpers import (
    PAIR_LABELS,
    PAIR_SPECS,
    config_label,
    differing_config_keys,
    infer_class_to_label,
    pair_key,
    resolve_run_dir,
    safe_name,
)


CONFIG_DIFF_EXCLUDE = {
    "cache_size",
    "max_cache_chunks",
    "eval_batch_size",
    "log_batch_lengths",
    "num_workers",
    "log_cuda_memory",
    "shuffle_hit_feature_seed",
    "force_math_sdp",
}
THETA_KEY = "GenParticles.momentum.theta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare metrics across saved run folders.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories or run names (e.g. test2).",
    )
    parser.add_argument(
        "--base-dir",
        default="save",
        help="Base directory used when run names are provided (default: save).",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Output directory for generated plots.",
    )
    parser.add_argument(
        "--barrel-theta-min",
        type=float,
        default=0.55,
        help="Barrel lower theta bound in radians.",
    )
    parser.add_argument(
        "--barrel-theta-max",
        type=float,
        default=float(np.pi / 2 + 0.55),
        help="Barrel upper theta bound in radians.",
    )
    parser.add_argument(
        "--cache-dir",
        default="plots/.run_compare_cache",
        help="Directory used to cache extracted theta arrays across runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk caching of extracted theta arrays.",
    )
    parser.add_argument(
        "--show-hdf5-progress",
        action="store_true",
        help="Show detailed HDF5 file/chunk progress while loading theta.",
    )
    return parser.parse_args()


def _resolve_h5_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        Path(".") / raw,
        Path("h5s") / raw,
        Path("/store/ml/dual-readout/h5s") / raw,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


def _extract_event_ids(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    file_ids = np.full(len(records), -1, dtype=np.int64)
    event_ids = np.full(len(records), -1, dtype=np.int64)
    for i, rec in enumerate(records):
        event_id = rec.get("event_id", [])
        if isinstance(event_id, list) and len(event_id) >= 2:
            file_ids[i] = int(event_id[0])
            event_ids[i] = int(event_id[1])
    return file_ids, event_ids


def _load_var_from_h5(
    records: List[Dict],
    files: List[str],
    dataset_key: str,
    cache: Dict[Tuple[str, Tuple[str, ...], bytes, bytes], np.ndarray] | None = None,
    *,
    desc: str | None = None,
    chunk_size: int = 8192,
) -> np.ndarray:
    file_ids, event_ids = _extract_event_ids(records)
    cache_key = (
        dataset_key,
        tuple(str(f) for f in files),
        file_ids.tobytes(),
        event_ids.tobytes(),
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key].copy()

    out = np.full(len(records), np.nan, dtype=np.float64)
    idx_by_file: Dict[int, List[int]] = {}
    for i in range(len(records)):
        if file_ids[i] < 0 or event_ids[i] < 0:
            continue
        idx_by_file.setdefault(int(file_ids[i]), []).append(i)

    file_items = list(idx_by_file.items())
    if desc is not None:
        file_items = tqdm(file_items, desc=desc, unit="file", leave=False)

    for fidx, rec_indices in file_items:
        if not (0 <= fidx < len(files)):
            continue
        h5_path = _resolve_h5_path(str(files[fidx]))
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h5:
            if dataset_key not in h5:
                continue
            ds = h5[dataset_key]
            ev = np.asarray([event_ids[i] for i in rec_indices], dtype=np.int64)
            order = np.argsort(ev, kind="stable")
            ev_sorted = ev[order]
            rec_idx_sorted = np.asarray(rec_indices, dtype=np.int64)[order]
            chunk_count = (ev_sorted.size + chunk_size - 1) // chunk_size
            chunk_iter = range(0, ev_sorted.size, chunk_size)
            if desc is not None and chunk_count > 1:
                chunk_iter = tqdm(
                    chunk_iter,
                    total=chunk_count,
                    desc=f"{Path(str(files[fidx])).name}:{dataset_key}",
                    unit="chunk",
                    leave=False,
                )
            for start in chunk_iter:
                stop = min(start + chunk_size, ev_sorted.size)
                out[rec_idx_sorted[start:stop]] = np.asarray(
                    ds[ev_sorted[start:stop]],
                    dtype=np.float64,
                ).reshape(-1)

    if cache is not None:
        cache[cache_key] = out.copy()
    return out


def _theta_cache_path(
    run_dir: Path,
    config_path: Path,
    output_path: Path,
    files: Sequence[str],
    dataset_key: str,
    cache_dir: Path,
) -> Path:
    payload = {
        "run_dir": str(run_dir.resolve()),
        "config_mtime_ns": config_path.stat().st_mtime_ns,
        "output_mtime_ns": output_path.stat().st_mtime_ns,
        "files": [str(f) for f in files],
        "dataset_key": dataset_key,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{safe_name(run_dir.name)}_{digest}.npz"


def load_run_theta(
    run_dir: Path,
    config_path: Path,
    output_path: Path,
    records: List[Dict],
    files: List[str],
    dataset_key: str,
    memory_cache: Dict[Tuple[str, Tuple[str, ...], bytes, bytes], np.ndarray] | None,
    *,
    cache_dir: Path | None,
    show_hdf5_progress: bool,
) -> np.ndarray:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _theta_cache_path(run_dir, config_path, output_path, files, dataset_key, cache_dir)
        if cache_path.exists():
            with np.load(cache_path) as cached:
                return np.asarray(cached[dataset_key], dtype=np.float64)

    out = _load_var_from_h5(
        records,
        files,
        dataset_key,
        memory_cache,
        desc=f"theta {run_dir.name}" if show_hdf5_progress else None,
    )
    if cache_path is not None:
        np.savez_compressed(cache_path, **{dataset_key: out})
    return out


def pair_auc_from_arrays(
    labels: np.ndarray,
    probs: np.ndarray,
    class_to_label: Mapping[str, int],
    pos: str,
    neg: str,
) -> float:
    y_true, y_score = pair_targets_from_arrays(labels, probs, class_to_label, pos, neg)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return math.nan


def pair_targets_from_arrays(
    labels: np.ndarray,
    probs: np.ndarray,
    class_to_label: Mapping[str, int],
    pos: str,
    neg: str,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if pos not in class_to_label or neg not in class_to_label:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    pos_idx = int(class_to_label[pos])
    neg_idx = int(class_to_label[neg])
    mask = np.isin(labels, [pos_idx, neg_idx])
    if extra_mask is not None:
        mask = mask & extra_mask
    if not np.any(mask):
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    y_true = (labels[mask] == pos_idx).astype(np.int32)
    y_score = probs[mask, pos_idx].astype(np.float64, copy=False)
    return y_true, y_score


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)

    theta_min = float(args.barrel_theta_min)
    theta_max = float(args.barrel_theta_max)
    if not (0.0 <= theta_min < theta_max <= math.pi):
        raise ValueError(
            f"Invalid barrel theta range: [{theta_min}, {theta_max}] must satisfy 0 <= min < max <= pi."
        )

    run_labels: List[str] = []
    metric_rows: List[Dict[str, float]] = []
    pair_rows: Dict[str, List[float]] = {pair_key(p, n): [] for p, n in PAIR_SPECS}
    run_cfgs: List[Dict[str, object]] = []
    run_labels_np: List[np.ndarray] = []
    run_probs: List[np.ndarray] = []
    run_class_maps: List[Dict[str, int]] = []
    run_thetas: List[np.ndarray] = []
    theta_cache: Dict[Tuple[str, Tuple[str, ...], bytes, bytes], np.ndarray] = {}
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    for run_dir_raw in tqdm(args.run_dirs, desc="Loading runs", unit="run"):
        run_dir = resolve_run_dir(run_dir_raw, base_dir)
        label = run_dir.name
        run_labels.append(label)

        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        output_path = run_dir / "output.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
        metrics = json.loads(metrics_path.read_text())
        test = metrics.get("test", {})
        metric_rows.append(
            {
                "roc_auc": float(test.get("roc_auc", math.nan)),
                "accuracy": float(test.get("accuracy", math.nan)),
                "loss": float(test.get("loss", math.nan)),
                "energy_rmse": float(test.get("energy_rmse", math.nan)),
            }
        )

        if config_path.exists() and output_path.exists():
            cfg = json.loads(config_path.read_text())
            records = json.loads(output_path.read_text())
            labels_np = np.asarray([int(rec["label"]) for rec in records], dtype=np.int64)
            logits = np.asarray([rec["logits"] for rec in records], dtype=np.float64)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            source_files = cfg.get("test_files") or cfg.get("train_files") or []
            class_to_label = infer_class_to_label(records, source_files)
            theta = load_run_theta(
                run_dir,
                config_path,
                output_path,
                records,
                source_files,
                THETA_KEY,
                theta_cache,
                cache_dir=cache_dir,
                show_hdf5_progress=args.show_hdf5_progress,
            )
            run_cfgs.append(cfg)
            run_labels_np.append(labels_np)
            run_probs.append(probs)
            run_class_maps.append(class_to_label)
            run_thetas.append(theta)
            for pos, neg in PAIR_SPECS:
                key = pair_key(pos, neg)
                pair_rows[key].append(pair_auc_from_arrays(labels_np, probs, class_to_label, pos, neg))
        else:
            run_cfgs.append({})
            run_labels_np.append(np.array([], dtype=np.int64))
            run_probs.append(np.empty((0, 0), dtype=np.float64))
            run_class_maps.append({})
            run_thetas.append(np.array([], dtype=np.float64))
            for pos, neg in PAIR_SPECS:
                pair_rows[pair_key(pos, neg)].append(math.nan)

    x = np.arange(len(run_labels))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metric_specs = [
        ("roc_auc", "Test ROC-AUC"),
        ("accuracy", "Test Accuracy"),
        ("loss", "Test Loss"),
        ("energy_rmse", "Test Energy RMSE"),
    ]
    for ax, (key, title) in zip(axes.ravel(), metric_specs):
        ys = [row[key] for row in metric_rows]
        ax.bar(x, ys, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_metrics = out_dir / "run_compare_metrics.png"
    fig.savefig(out_metrics, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_metrics}")

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, num=len(PAIR_SPECS))
    for off, (pos, neg) in zip(offsets, PAIR_SPECS):
        key = pair_key(pos, neg)
        ys = pair_rows[key]
        ax.bar(x + off, ys, width=width / max(len(PAIR_SPECS), 1), label=PAIR_LABELS.get(key, key))
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Pairwise ROC-AUC by Run")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_pair = out_dir / "run_compare_pairwise_auc.png"
    fig.savefig(out_pair, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_pair}")

    diff_keys = differing_config_keys(run_cfgs, CONFIG_DIFF_EXCLUDE)
    curve_labels = [
        config_label(name, cfg, diff_keys, width=44) for name, cfg in zip(run_labels, run_cfgs)
    ]
    for pos, neg in PAIR_SPECS:
        key = pair_key(pos, neg)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = False
        for i, (labels_np, probs, class_map) in enumerate(zip(run_labels_np, run_probs, run_class_maps)):
            y_true, y_score = pair_targets_from_arrays(labels_np, probs, class_map, pos, neg)
            if y_true.size == 0 or np.unique(y_true).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = float(roc_auc_score(y_true, y_score))
            ax.plot(fpr, tpr, linewidth=1.8, label=f"{curve_labels[i]}\nAUC={auc_val:.5f}")
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
        if key == "pi+ vs e-":
            ax.set_xlim(0.0, 0.5)
            ax.set_ylim(0.5, 1.0)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve Comparison: {PAIR_LABELS.get(key, key)}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, labelspacing=0.9)
        fig.tight_layout()
        safe = safe_name(key)
        out_curve = out_dir / f"run_compare_roc_curve_{safe}.png"
        fig.savefig(out_curve, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_curve}")

        region_specs = [
            ("barrel", f"Barrel: {theta_min:.3f} <= theta <= {theta_max:.3f} rad", "-"),
            ("endcap", f"Endcap: theta < {theta_min:.3f} or theta > {theta_max:.3f} rad", "--"),
        ]

        fig, ax = plt.subplots(figsize=(11.5, 6))
        any_region_overlay = False
        for i, (labels_np, probs, class_map, theta) in enumerate(tqdm(list(zip(run_labels_np, run_probs, run_class_maps, run_thetas)), desc=f"{key} overlay runs", unit="run", leave=False)):
            if theta.size == 0:
                continue
            finite = np.isfinite(theta)
            for region_key, _, linestyle in region_specs:
                if region_key == "barrel":
                    region_mask = finite & (theta >= theta_min) & (theta <= theta_max)
                    region_name = "barrel"
                else:
                    region_mask = finite & ((theta < theta_min) | (theta > theta_max))
                    region_name = "endcap"
                y_true, y_score = pair_targets_from_arrays(labels_np, probs, class_map, pos, neg, extra_mask=region_mask)
                if y_true.size == 0 or np.unique(y_true).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_val = float(roc_auc_score(y_true, y_score))
                ax.plot(
                    fpr,
                    tpr,
                    linewidth=1.8,
                    linestyle=linestyle,
                    label=f"{curve_labels[i]}\n{region_name} AUC={auc_val:.5f}",
                )
                any_region_overlay = True
        if any_region_overlay:
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
            if key == "pi+ vs e-":
                ax.set_xlim(0.0, 0.5)
                ax.set_ylim(0.5, 1.0)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve Comparison by Region: {PAIR_LABELS.get(key, key)}")
            ax.grid(True, linestyle="--", alpha=0.35)
            run_legend = ax.legend(
                fontsize=8,
                labelspacing=0.9,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            ax.add_artist(run_legend)
            style_handles = [
                Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="barrel"),
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="endcap"),
            ]
            ax.legend(
                handles=style_handles,
                title="Region style",
                loc="upper left",
                bbox_to_anchor=(1.02, 0.38),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
            out_region_overlay = out_dir / f"run_compare_roc_curve_{safe}_barrel_endcap_overlay.png"
            fig.savefig(out_region_overlay, dpi=150)
            print(f"Wrote {out_region_overlay}")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        any_region_plotted = False
        for ax_region, (region_key, region_title, _) in zip(axes, region_specs):
            plotted_region = False
            for i, (labels_np, probs, class_map, theta) in enumerate(tqdm(list(zip(run_labels_np, run_probs, run_class_maps, run_thetas)), desc=f"{region_key} runs", unit="run", leave=False)):
                if theta.size == 0:
                    continue
                finite = np.isfinite(theta)
                if region_key == "barrel":
                    region_mask = finite & (theta >= theta_min) & (theta <= theta_max)
                else:
                    region_mask = finite & ((theta < theta_min) | (theta > theta_max))
                y_true, y_score = pair_targets_from_arrays(labels_np, probs, class_map, pos, neg, extra_mask=region_mask)
                if y_true.size == 0 or np.unique(y_true).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_val = float(roc_auc_score(y_true, y_score))
                ax_region.plot(fpr, tpr, linewidth=1.8, label=f"{curve_labels[i]}\nAUC={auc_val:.5f}")
                plotted_region = True
                any_region_plotted = True
            ax_region.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
            if key == "pi+ vs e-":
                ax_region.set_xlim(0.0, 0.5)
                ax_region.set_ylim(0.5, 1.0)
            ax_region.set_xlabel("False Positive Rate")
            ax_region.set_title(region_title)
            ax_region.grid(True, linestyle="--", alpha=0.35)
            if not plotted_region:
                ax_region.text(0.5, 0.5, "No valid events", transform=ax_region.transAxes, ha="center", va="center")
        if any_region_plotted:
            axes[0].set_ylabel("True Positive Rate")
            axes[0].legend(fontsize=8, labelspacing=0.9)
            fig.suptitle(f"ROC Curve Comparison by Region: {PAIR_LABELS.get(key, key)} (left=barrel, right=endcap)")
            fig.tight_layout()
            out_region = out_dir / f"run_compare_roc_curve_{safe}_barrel_endcap.png"
            fig.savefig(out_region, dpi=150)
            print(f"Wrote {out_region}")
        plt.close(fig)

    summary = {
        "runs": run_labels,
        "config_diff_keys": diff_keys,
        "config_labels": curve_labels,
        "test_metrics": metric_rows,
        "pairwise_auc": pair_rows,
        "barrel_theta_min": theta_min,
        "barrel_theta_max": theta_max,
        "cache_dir": None if args.no_cache else args.cache_dir,
    }
    out_json = out_dir / "run_compare_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
