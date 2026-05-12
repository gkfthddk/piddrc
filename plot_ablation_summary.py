"""Generate ablation summary figure: bar chart comparing timing configurations.

Shows per-pair AUC and overall accuracy across all four ablation modes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_helpers import set_publication_style, softmax, infer_class_to_label

set_publication_style()

RUN_CONFIGS = [
    ("Full timing", "save/split_mamba_40k_rw005_lowlr_dw005"),
    ("Shuffled timing", "save/split_mamba_40k_rw005_lowlr_dw005_shuffletime"),
    ("Integrated amp.", "save/split_mamba_40k_rw005_lowlr_dw005_calo2d"),
    ("No timing", "save/split_mamba_40k_rw005_lowlr_dw005_notime"),
]

PAIRS = [
    ("pi0", "gamma", r"$\pi^0$ vs $\gamma$"),
    ("pi+", "e-", r"$\pi^+$ vs $e^-$"),
]

OUT_DIR = Path("plots/publication")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#2196F3", "#FF9800", "#9C27B0", "#607D8B"]


def compute_pair_auc(labels, probs, c2l, pos, neg):
    if pos not in c2l or neg not in c2l:
        return float("nan")
    pos_idx = c2l[pos]
    neg_idx = c2l[neg]
    mask = np.isin(labels, [pos_idx, neg_idx])
    y_true = (labels[mask] == pos_idx).astype(int)
    y_score = probs[mask, pos_idx]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def main():
    results = []
    for run_label, run_dir in RUN_CONFIGS:
        output_path = Path(run_dir) / "output.json"
        config_path = Path(run_dir) / "config.json"
        with open(output_path) as f:
            data = json.load(f)
        with open(config_path) as f:
            config = json.load(f)
        records = data["records"]
        source_files = config.get("train_files", [])
        c2l = infer_class_to_label(records, source_files)
        labels = np.array([int(r["label"]) for r in records])
        probs = np.array([softmax(r["logits"]) for r in records])
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == labels)

        pair_aucs = {}
        for pos, neg, pair_title in PAIRS:
            pair_aucs[pair_title] = compute_pair_auc(labels, probs, c2l, pos, neg)

        results.append({
            "label": run_label,
            "accuracy": accuracy,
            "pair_aucs": pair_aucs,
        })

    # Create grouped bar chart
    metrics = ["Overall Accuracy"] + [p[2] for p in PAIRS]
    n_metrics = len(metrics)
    n_runs = len(results)
    x = np.arange(n_metrics)
    width = 0.18
    offsets = np.arange(n_runs) - (n_runs - 1) / 2.0

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, res in enumerate(results):
        values = [res["accuracy"]]
        for _, _, pair_title in PAIRS:
            values.append(res["pair_aucs"].get(pair_title, 0))
        bars = ax.bar(x + offsets[i] * width, values, width * 0.9,
                      label=res["label"], color=COLORS[i], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Ablation Study: Impact of Timing Information", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left", framealpha=0.9)
    ax.set_ylim(0.55, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out_path = OUT_DIR / f"ablation_summary.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)

    # Print table for LaTeX
    print("\n=== LaTeX Table Data ===")
    for res in results:
        vals = [f"{res['accuracy']:.3f}"]
        for _, _, pt in PAIRS:
            vals.append(f"{res['pair_aucs'].get(pt, 0):.3f}")
        print(f"  {res['label']:20s} & {'  & '.join(vals)} \\\\")


if __name__ == "__main__":
    main()
