"""Evaluation metrics for classification and regression tasks."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - fallback when sklearn missing
    roc_auc_score = None


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())

def roc_auc(logits: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    if roc_auc_score is None:
        return None
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    num_classes = probs.shape[1]
    try:
        if num_classes == 2:
            return float(roc_auc_score(labels_np, probs[:, 1]))
        one_hot = F.one_hot(labels, num_classes=num_classes).cpu().numpy()
        return float(roc_auc_score(one_hot, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def pairwise_roc_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_a: int,
    class_b: int,
) -> Optional[float]:
    if roc_auc_score is None:
        return None
    mask = (labels == class_a) | (labels == class_b)
    if int(mask.sum().item()) < 2:
        return None
    subset_logits = logits[mask][:, [class_a, class_b]]
    subset_labels = (labels[mask] == class_b).to(torch.long)
    labels_np = subset_labels.cpu().numpy()
    if np.unique(labels_np).size < 2:
        return None
    probs = torch.softmax(subset_logits, dim=1)[:, 1].cpu().numpy()
    try:
        return float(roc_auc_score(labels_np, probs))
    except ValueError:
        return None


def class_recall(logits: torch.Tensor, labels: torch.Tensor, class_index: int) -> Optional[float]:
    mask = labels == class_index
    positives = int(mask.sum().item())
    if positives == 0:
        return None
    preds = logits.argmax(dim=1)
    return float((preds[mask] == class_index).float().mean().item())


def macro_recall(logits: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    num_classes = logits.shape[1]
    if num_classes <= 0:
        return None
    recalls = []
    for class_index in range(num_classes):
        recall = class_recall(logits, labels, class_index)
        if recall is not None:
            recalls.append(recall)
    if not recalls:
        return None
    return float(np.mean(np.asarray(recalls, dtype=np.float64)))


def energy_resolution(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_np = pred.detach().cpu().to(torch.float32).numpy()
    target_np = target.detach().cpu().to(torch.float32).numpy()
    residual = pred_np - target_np
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(target_np) > 1e-6, residual / target_np, 0.0)
    resolution = float(np.std(rel))
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    return {"resolution": resolution, "rmse": rmse, "bias": bias}


def energy_linearity(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred_np = pred.detach().cpu().to(torch.float32).numpy()
    target_np = target.detach().cpu().to(torch.float32).numpy()
    if len(pred_np) < 2:
        return 1.0, 0.0
    slope, intercept = np.polyfit(target_np, pred_np, deg=1)
    return float(slope), float(intercept)


__all__ = [
    "accuracy",
    "mse",
    "roc_auc",
    "pairwise_roc_auc",
    "class_recall",
    "macro_recall",
    "energy_resolution",
    "energy_linearity",
]
