"""Generate max-points performance plots across saved runs.

Outputs:
- Pairwise ROC-AUC vs max_points from `output.json`:
  - pi0 vs gamma (pi0 treated as positive)
  - pi+ vs e-    (pi+ treated as positive)
- Training/evaluation summary metrics vs max_points from `history.json`
  and `metrics.json`:
  - best validation ROC-AUC
  - best validation loss
  - test ROC-AUC
  - test loss
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score


RUN_GLOB = "save/*r1_t10*/output.json"
PLOT_DIR = Path("plots")
PAIR_SPECS: List[Tuple[str, str]] = [("pi0", "gamma"), ("pi+", "e-")]
DISPLAY_LABELS = {
    "pi0 vs gamma": r"$\pi^{0}$ vs $\gamma$",
    "pi+ vs e-": r"$\pi^{+}$ vs $e^{-}$",
}
SUMMARY_METRICS: List[Tuple[str, str, str]] = [
    ("best_val_roc_auc", "Best val ROC-AUC", "max_points_best_val_roc_auc.png"),
    ("best_val_loss", "Best val loss", "max_points_best_val_loss.png"),
    ("test_roc_auc", "Test ROC-AUC", "max_points_test_roc_auc.png"),
    ("test_loss", "Test loss", "max_points_test_loss.png"),
]


def softmax(logits: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(logits), dtype=np.float64)
    arr = arr - arr.max()
    np.exp(arr, out=arr)
    arr /= arr.sum()
    return arr


def class_prefix(file_name: str) -> str:
    """Extract short class tag from a training file name."""
    stem = Path(file_name).stem
    return stem.split("_", 1)[0]


def load_run(output_path: Path) -> Tuple[str, int, Dict[str, int], List[Dict]]:
    """Return model name, max_points, class_name->index map, and output records."""
    run_dir = output_path.parent
    config = json.loads(run_dir.joinpath("config.json").read_text())
    model_name = config.get("model") or config.get("name", run_dir.name).split("_", 1)[0]
    max_points = config.get("max_points")

    files = config.get("train_files") or []
    class_from_file = [class_prefix(f) for f in files]

    records = json.loads(output_path.read_text())

    # Build label->file_id frequency to resolve class ordering.
    freq: Dict[int, Counter] = defaultdict(Counter)
    for rec in records:
        label = rec["label"]
        file_id = rec["event_id"][0]
        freq[label].update([file_id])

    label_to_class: Dict[int, str] = {}
    for label, counter in freq.items():
        file_id, _ = counter.most_common(1)[0]
        name = class_from_file[file_id] if file_id < len(class_from_file) else f"class{label}"
        label_to_class[label] = name

    class_to_label = {name: lbl for lbl, name in label_to_class.items()}
    return model_name, max_points, class_to_label, records


def gather_runs() -> List[Tuple[str, int, Dict[str, int], List[Dict]]]:
    runs: List[Tuple[str, int, Dict[str, int], List[Dict]]] = []
    for path in sorted(Path(".").glob(RUN_GLOB)):
        try:
            runs.append(load_run(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return runs


def extract_summary_metrics(run_dir: Path) -> Dict[str, float]:
    """Read history/metrics and return scalar summary values for one run."""
    out: Dict[str, float] = {}

    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"

    if history_path.exists():
        history = json.loads(history_path.read_text())
        val_records = history.get("val") or []
        if val_records:
            val_aucs = [rec.get("roc_auc") for rec in val_records if rec.get("roc_auc") is not None]
            val_losses = [rec.get("loss") for rec in val_records if rec.get("loss") is not None]
            if val_aucs:
                out["best_val_roc_auc"] = float(max(val_aucs))
            if val_losses:
                out["best_val_loss"] = float(min(val_losses))

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        test_metrics = metrics.get("test") or {}
        if test_metrics.get("roc_auc") is not None:
            out["test_roc_auc"] = float(test_metrics["roc_auc"])
        if test_metrics.get("loss") is not None:
            out["test_loss"] = float(test_metrics["loss"])

    return out


def gather_summary_data() -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """Collect summary metrics keyed by metric then model."""
    metric_model_points: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for config_path in sorted(Path(".").glob("save/*r1_t10*/config.json")):
        run_dir = config_path.parent
        try:
            config = json.loads(config_path.read_text())
            model_name = config.get("model") or config.get("name", run_dir.name).split("_", 1)[0]
            max_points = config.get("max_points")
            if max_points is None:
                continue
            values = extract_summary_metrics(run_dir)
            for metric_name, value in values.items():
                if np.isfinite(value):
                    metric_model_points[metric_name][model_name].append((int(max_points), float(value)))
        except Exception as exc:
            print(f"Skipping summary metrics for {run_dir}: {exc}")

    return metric_model_points


def compute_pair_auc(records: List[Dict], class_to_label: Dict[str, int], positive: str, negative: str) -> float:
    if positive not in class_to_label or negative not in class_to_label:
        raise KeyError(f"Classes {positive}/{negative} not found in mapping {class_to_label}")

    pos_idx = class_to_label[positive]
    neg_idx = class_to_label[negative]

    y_true: List[int] = []
    y_score: List[float] = []

    for rec in records:
        lbl = rec["label"]
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
    runs = gather_runs()
    if not runs:
        print("No runs found; nothing to plot.")
        return

    # Collect results keyed by pair and model.
    pair_model_points: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    pair_model_auc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for model_name, max_points, class_to_label, records in sorted(runs, key=lambda r: r[1]):
        for positive, negative in PAIR_SPECS:
            auc = compute_pair_auc(records, class_to_label, positive, negative)
            key = f"{positive} vs {negative}"
            pair_model_points[key][model_name].append(max_points)
            pair_model_auc[key][model_name].append(auc)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for key in pair_model_points:
        markers = cycle(["o", "s", "^", "D", "v", "x"])
        title = DISPLAY_LABELS.get(key, key)
        fig, ax = plt.subplots(figsize=(5, 4))

        for model_name, xs in pair_model_points[key].items():
            ys = pair_model_auc[key][model_name]
            if not xs:
                continue
            # Sort by x for clean lines
            sorted_pairs = sorted(zip(xs, ys))
            xs_sorted, ys_sorted = zip(*sorted_pairs)
            ax.plot(xs_sorted, ys_sorted, marker=next(markers), label=model_name)

        ax.set_xlabel("max_points (n)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Model")
        fig.tight_layout()
        out_path = PLOT_DIR / f"{key.replace(' ', '_').replace('+', 'plus').replace('-', 'minus')}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")

    summary_data = gather_summary_data()
    for metric_name, y_label, file_name in SUMMARY_METRICS:
        if metric_name not in summary_data:
            continue
        markers = cycle(["o", "s", "^", "D", "v", "x"])
        fig, ax = plt.subplots(figsize=(5, 4))
        for model_name, points in summary_data[metric_name].items():
            if not points:
                continue
            points = sorted(points, key=lambda item: item[0])
            xs = [item[0] for item in points]
            ys = [item[1] for item in points]
            ax.plot(xs, ys, marker=next(markers), label=model_name)
        ax.set_xlabel("max_points (n)")
        ax.set_ylabel(y_label)
        ax.set_title(y_label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Model")
        fig.tight_layout()
        out_path = PLOT_DIR / file_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
