"""Generate confusion matrix comparison: Full timing vs No timing.

Produces a side-by-side normalized confusion matrix for the two key
ablation configurations used in the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_helpers import set_publication_style, softmax

set_publication_style()

CLASS_ORDER = ["e-", "gamma", "pi0", "pi+"]
CLASS_LABELS = [r"$e^-$", r"$\gamma$", r"$\pi^0$", r"$\pi^+$"]

RUN_CONFIGS = {
    "Full timing": "save/split_mamba_40k_rw005_lowlr_dw005",
    "No timing": "save/split_mamba_40k_rw005_lowlr_dw005_notime",
}

OUT_DIR = Path("plots/publication")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_confusion_matrix(run_dir: str) -> np.ndarray:
    output_path = Path(run_dir) / "output.json"
    with open(output_path) as f:
        data = json.load(f)
    records = data["records"]

    n_classes = len(CLASS_ORDER)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for rec in records:
        true_label = int(rec["label"])
        logits = rec["logits"]
        probs = softmax(logits)
        pred_label = int(np.argmax(probs))
        if 0 <= true_label < n_classes and 0 <= pred_label < n_classes:
            cm[true_label, pred_label] += 1

    # Normalize by row (true label)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(np.float64) / row_sums.astype(np.float64)
    return cm_norm


def plot_cm(ax, cm_norm, title):
    n = len(CLASS_LABELS)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(CLASS_LABELS, fontsize=13)
    ax.set_yticks(range(n))
    ax.set_yticklabels(CLASS_LABELS, fontsize=13)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold")
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for idx, (label, run_dir) in enumerate(RUN_CONFIGS.items()):
        cm = load_confusion_matrix(run_dir)
        im = plot_cm(axes[idx], cm, label)

    fig.colorbar(im, ax=axes, shrink=0.85, label="Classification rate")
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out_path = OUT_DIR / f"confusion_matrix_comparison.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
