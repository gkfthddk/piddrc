"""Compare performance metrics across specific saved run directories.

Example:
  python plot_run_comparison.py --run-dirs save/test2 save/test3 save/test4 save/test5
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from utils.plot_helpers import (
    PAIR_LABELS,
    PAIR_SPECS,
    config_label,
    differing_config_keys,
    infer_class_to_label,
    pair_key,
    resolve_run_dir,
    safe_name,
    softmax,
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
    return parser.parse_args()


def pair_auc(records: List[Dict], class_to_label: Dict[str, int], pos: str, neg: str) -> float:
    if pos not in class_to_label or neg not in class_to_label:
        return math.nan
    pos_idx = class_to_label[pos]
    neg_idx = class_to_label[neg]
    y_true: List[int] = []
    y_score: List[float] = []
    for rec in records:
        lbl = int(rec["label"])
        if lbl not in (pos_idx, neg_idx):
            continue
        probs = softmax(rec["logits"])
        y_true.append(1 if lbl == pos_idx else 0)
        y_score.append(float(probs[pos_idx]))
    if not y_true:
        return math.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return math.nan


def pair_targets(records: List[Dict], class_to_label: Dict[str, int], pos: str, neg: str) -> Tuple[np.ndarray, np.ndarray]:
    if pos not in class_to_label or neg not in class_to_label:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    pos_idx = class_to_label[pos]
    neg_idx = class_to_label[neg]
    y_true: List[int] = []
    y_score: List[float] = []
    for rec in records:
        lbl = int(rec["label"])
        if lbl not in (pos_idx, neg_idx):
            continue
        probs = softmax(rec["logits"])
        y_true.append(1 if lbl == pos_idx else 0)
        y_score.append(float(probs[pos_idx]))
    return np.asarray(y_true, dtype=np.int32), np.asarray(y_score, dtype=np.float64)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)

    run_labels: List[str] = []
    metric_rows: List[Dict[str, float]] = []
    pair_rows: Dict[str, List[float]] = {pair_key(p, n): [] for p, n in PAIR_SPECS}
    run_cfgs: List[Dict[str, object]] = []
    run_records: List[List[Dict]] = []
    run_class_maps: List[Dict[str, int]] = []

    for run_dir_raw in args.run_dirs:
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
            class_to_label = infer_class_to_label(records, cfg.get("train_files") or [])
            run_cfgs.append(cfg)
            run_records.append(records)
            run_class_maps.append(class_to_label)
            for pos, neg in PAIR_SPECS:
                key = pair_key(pos, neg)
                pair_rows[key].append(pair_auc(records, class_to_label, pos, neg))
        else:
            run_cfgs.append({})
            run_records.append([])
            run_class_maps.append({})
            for pos, neg in PAIR_SPECS:
                pair_rows[pair_key(pos, neg)].append(math.nan)

    x = np.arange(len(run_labels))

    # Plot core test metrics.
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

    # Plot pairwise AUC comparison.
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

    # Plot ROC curves for each pair across runs.
    diff_keys = differing_config_keys(run_cfgs, CONFIG_DIFF_EXCLUDE)
    curve_labels = [
        config_label(name, cfg, diff_keys, width=44) for name, cfg in zip(run_labels, run_cfgs)
    ]
    for pos, neg in PAIR_SPECS:
        key = pair_key(pos, neg)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = False
        for i, (records, class_map) in enumerate(zip(run_records, run_class_maps)):
            y_true, y_score = pair_targets(records, class_map, pos, neg)
            if y_true.size == 0 or np.unique(y_true).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = pair_auc(records, class_map, pos, neg)
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

    # Save machine-readable summary.
    summary = {
        "runs": run_labels,
        "config_diff_keys": diff_keys,
        "config_labels": curve_labels,
        "test_metrics": metric_rows,
        "pairwise_auc": pair_rows,
    }
    out_json = out_dir / "run_compare_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
