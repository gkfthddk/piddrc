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


__all__ = ["accuracy", "roc_auc", "energy_resolution", "energy_linearity"]
