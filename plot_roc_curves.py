"""Generate ROC curves for key classification pairs with ablation overlay.

Produces ROC curves for gamma vs pi0 and pi+ vs e- comparing all four
timing configurations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_helpers import set_publication_style, softmax, infer_class_to_label

set_publication_style()

RUN_CONFIGS = [
    ("Full timing", "save/split_mamba_40k_rw005_lowlr_dw005", "-", 2.0),
    ("Shuffled timing", "save/split_mamba_40k_rw005_lowlr_dw005_shuffletime", "--", 1.6),
    ("No timing", "save/split_mamba_40k_rw005_lowlr_dw005_notime", "-.", 1.6),
    ("Integrated amp.", "save/split_mamba_40k_rw005_lowlr_dw005_calo2d", ":", 1.6),
]

PAIRS = [
    ("pi0", "gamma", r"$\pi^0$ vs $\gamma$"),
    ("pi+", "e-", r"$\pi^+$ vs $e^-$"),
]

OUT_DIR = Path("plots/publication")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_run_data(run_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    output_path = Path(run_dir) / "output.json"
    config_path = Path(run_dir) / "config.json"
    with open(output_path) as f:
        data = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    records = data["records"]
    source_files = config.get("train_files", [])
    class_to_label = infer_class_to_label(records, source_files)

    labels = np.array([int(r["label"]) for r in records])
    probs = np.array([softmax(r["logits"]) for r in records])
    return labels, probs, class_to_label


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for pair_idx, (pos, neg, pair_title) in enumerate(PAIRS):
        ax = axes[pair_idx]

        for run_label, run_dir, ls, lw in RUN_CONFIGS:
            labels, probs, c2l = load_run_data(run_dir)

            if pos not in c2l or neg not in c2l:
                print(f"Skipping {run_label}: class mapping not found for {pos}/{neg}")
                continue

            pos_idx = c2l[pos]
            neg_idx = c2l[neg]

            mask = np.isin(labels, [pos_idx, neg_idx])
            y_true = (labels[mask] == pos_idx).astype(int)
            y_score = probs[mask, pos_idx]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, linestyle=ls, linewidth=lw,
                    label=f"{run_label} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(pair_title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out_path = OUT_DIR / f"roc_curves_ablation.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
