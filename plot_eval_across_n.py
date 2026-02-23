"""Plot fixed-checkpoint evaluation results across max_points values.

Reads folders like:
  save/<run>/eval_same_ckpt_diff_n/n1000/metrics.json
  save/<run>/eval_same_ckpt_diff_n/n1000/output.json
and writes comparison plots under `plots/`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score


PAIR_SPECS: List[Tuple[str, str]] = [("pi0", "gamma"), ("pi+", "e-")]
PAIR_LABELS = {
    "pi0 vs gamma": r"$\pi^{0}$ vs $\gamma$",
    "pi+ vs e-": r"$\pi^{+}$ vs $e^{-}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval_same_ckpt_diff_n results.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory containing n*/metrics.json and n*/output.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="pi0_vs_gamma",
        choices=["all", "pi0_vs_gamma", "pi+_vs_e-"],
        help="Select which pairwise ROC-AUC curve to plot.",
    )
    return parser.parse_args()


def softmax(logits: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(logits), dtype=np.float64)
    arr = arr - arr.max()
    np.exp(arr, out=arr)
    arr /= arr.sum()
    return arr


def class_prefix(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("_", 1)[0]


def infer_class_to_label(records: List[Dict], train_files: List[str]) -> Dict[str, int]:
    names = [class_prefix(f) for f in train_files]
    freq: Dict[int, Counter] = defaultdict(Counter)
    for rec in records:
        label = int(rec["label"])
        event_id = rec.get("event_id", [])
        if isinstance(event_id, list) and len(event_id) >= 1:
            file_id = int(event_id[0])
            freq[label].update([file_id])

    out: Dict[str, int] = {}
    for label, cnt in freq.items():
        file_id, _ = cnt.most_common(1)[0]
        if 0 <= file_id < len(names):
            out[names[file_id]] = label
    return out


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


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir
    if not eval_dir.exists():
        raise SystemExit(f"Directory not found: {eval_dir}")

    run_dirs = [p for p in eval_dir.glob("n*") if p.is_dir()]
    if not run_dirs:
        raise SystemExit(f"No n* subdirectories found in {eval_dir}")

    rows = []
    pair_rows = []
    for run_dir in sorted(run_dirs, key=lambda p: int(p.name[1:])):
        n = int(run_dir.name[1:])
        metrics_path = run_dir / "metrics.json"
        output_path = run_dir / "output.json"
        cfg_path = run_dir / "config.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text())
        test = metrics.get("test", {})
        rows.append(
            {
                "n": n,
                "roc_auc": float(test.get("roc_auc", math.nan)),
                "accuracy": float(test.get("accuracy", math.nan)),
                "loss": float(test.get("loss", math.nan)),
                "energy_rmse": float(test.get("energy_rmse", math.nan)),
            }
        )

        if output_path.exists() and cfg_path.exists():
            records = json.loads(output_path.read_text())
            cfg = json.loads(cfg_path.read_text())
            class_to_label = infer_class_to_label(records, cfg.get("train_files") or [])
            for pos, neg in PAIR_SPECS:
                key = f"{pos} vs {neg}"
                auc = pair_auc(records, class_to_label, pos, neg)
                pair_rows.append({"n": n, "pair": key, "auc": auc})

    if not rows:
        raise SystemExit("No metrics found to plot.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["n"])
    xs = [r["n"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("roc_auc", "Test ROC-AUC"),
        ("accuracy", "Test Accuracy"),
        ("loss", "Test Loss"),
        ("energy_rmse", "Test Energy RMSE"),
    ]
    for ax, (k, title) in zip(axes.ravel(), panels):
        ys = [r[k] for r in rows]
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("max_points (n)")
        ax.set_ylabel(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_main = args.out_dir / f"{eval_dir.parent.name}_eval_across_n_metrics.png"
    fig.savefig(out_main, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_main}")

    if pair_rows:
        selected_pair = None
        if args.pair == "pi0_vs_gamma":
            selected_pair = "pi0 vs gamma"
        elif args.pair == "pi+_vs_e-":
            selected_pair = "pi+ vs e-"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        pairs = sorted(set(r["pair"] for r in pair_rows))
        if selected_pair is not None:
            pairs = [p for p in pairs if p == selected_pair]
        for pair in pairs:
            pts = sorted((r["n"], r["auc"]) for r in pair_rows if r["pair"] == pair)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker="o",
                label=PAIR_LABELS.get(pair, pair),
            )
        ax.set_xlabel("max_points (n)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Pairwise AUC Across n")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        pair_suffix = "pairwise_auc" if selected_pair is None else selected_pair.replace(" ", "_")
        out_pair = args.out_dir / f"{eval_dir.parent.name}_eval_across_n_{pair_suffix}.png"
        fig.savefig(out_pair, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_pair}")


if __name__ == "__main__":
    main()
