"""Training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models.base import ModelOutputs
from . import metrics as metrics_mod


@dataclass
class TrainingConfig:
    epochs: int = 20
    log_every: int = 10
    classification_weight: float = 1.0
    regression_weight: float = 1.0
    max_grad_norm: Optional[float] = None
    use_amp: bool = True
    warmup_steps: int = 0
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    early_stopping_monitor: str = "loss"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or TrainingConfig()
        self.scaler = torch.amp.GradScaler('cuda',enabled=self.config.use_amp and device.type == "cuda")
        self._global_step = 0
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[Dict[str, float]]]:
        history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}
        best_loss = float("inf")
        epochs_without_improvement = 0
        monitor_metric = self.config.early_stopping_monitor or "loss"
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch)
            history["train"].append(train_metrics)
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch)
                history["val"].append(val_metrics)
                monitored_metrics = val_metrics
            else:
                monitored_metrics = train_metrics

            if self.config.early_stopping_patience is not None:
                monitored_value = monitored_metrics.get(monitor_metric)
                if monitored_value is None:
                    monitored_value = monitored_metrics.get("loss")
                if monitored_value is not None:
                    if monitored_value + self.config.early_stopping_min_delta < best_loss:
                        best_loss = monitored_value
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= self.config.early_stopping_patience:
                            break

            if self.scheduler is not None and (
                self.config.warmup_steps <= 0 or self._global_step >= self.config.warmup_steps
            ):
                self.scheduler.step()
        return history

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        return self._run_epoch(data_loader, training=False, epoch=None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_epoch(self, data_loader: DataLoader, *, training: bool, epoch: Optional[int]) -> Dict[str, float]:
        mode = "train" if training else "eval"
        self.model.train(mode == "train")
        iterator = tqdm(data_loader, desc=f"{mode} epoch {epoch}" if epoch else mode, leave=False)
        losses: List[float] = []
        cls_losses: List[float] = []
        reg_losses: List[float] = []
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        energy_pred: List[torch.Tensor] = []
        energy_true: List[torch.Tensor] = []
        log_sigma_list: List[torch.Tensor] = []
        sigma_values: List[float] = []

        for step, batch in enumerate(iterator, start=1):
            batch = self._move_to_device(batch)
            with torch.amp.autocast('cuda',enabled=self.scaler.is_enabled()):
                outputs = self.model(batch)
                loss_cls, loss_reg = self._compute_losses(outputs, batch)
                loss = self.config.classification_weight * loss_cls + self.config.regression_weight * loss_reg

            if training:
                self._global_step += 1
                self._apply_warmup()
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.config.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            losses.append(loss.detach().item())
            cls_losses.append(loss_cls.detach().item())
            reg_losses.append(loss_reg.detach().item())
            logits_list.append(outputs.logits.detach().cpu())
            labels_list.append(batch["labels"].detach().cpu())
            energy_pred.append(outputs.energy.detach().cpu())
            energy_true.append(batch["energy"].detach().cpu())
            if outputs.log_sigma is not None:
                log_sigma = outputs.log_sigma.detach()
                log_sigma_list.append(log_sigma.cpu())
                sigma_values.append(float(torch.exp(log_sigma).mean().item()))

            if training and step % self.config.log_every == 0:
                postfix = {
                    "loss": sum(losses) / len(losses),
                    "reg": sum(reg_losses) / len(reg_losses),
                }
                if sigma_values:
                    postfix["sigma"] = sum(sigma_values) / len(sigma_values)
                try:
                    auc = metrics_mod.roc_auc(torch.cat(logits_list), torch.cat(labels_list))
                except RuntimeError:
                    auc = None
                if auc is not None:
                    postfix["auc"] = auc
                iterator.set_postfix(postfix)

        metrics = self._compute_metrics(
            logits_list,
            labels_list,
            energy_pred,
            energy_true,
            log_sigma_list if log_sigma_list else None,
        )
        metrics.update({
            "loss": float(sum(losses) / max(len(losses), 1)),
            "loss_cls": float(sum(cls_losses) / max(len(cls_losses), 1)),
            "loss_reg": float(sum(reg_losses) / max(len(reg_losses), 1)),
        })
        metrics["classification_loss"] = metrics["loss_cls"]
        metrics["regression_loss"] = metrics["loss_reg"]
        return metrics

    def _apply_warmup(self) -> None:
        warmup_steps = self.config.warmup_steps
        if warmup_steps <= 0:
            return
        if self._global_step > warmup_steps:
            return
        progress = min(1.0, float(self._global_step) / float(warmup_steps))
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * progress

    def _compute_losses(self, outputs: ModelOutputs, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch["labels"]
        energy = batch["energy"]
        loss_cls = nn.functional.cross_entropy(outputs.logits, labels)
        if outputs.log_sigma is not None:
            log_sigma = outputs.log_sigma.clamp(min=-5.0, max=5.0)
            residual = energy - outputs.energy
            loss_reg = 0.5 * torch.exp(-2.0 * log_sigma) * (residual ** 2) + log_sigma
            loss_reg = loss_reg.mean()
        else:
            loss_reg = nn.functional.mse_loss(outputs.energy, energy)
        return loss_cls, loss_reg

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _compute_metrics(
        self,
        logits_list: Iterable[torch.Tensor],
        labels_list: Iterable[torch.Tensor],
        energy_pred: Iterable[torch.Tensor],
        energy_true: Iterable[torch.Tensor],
        log_sigma_list: Optional[Iterable[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        logits = torch.cat(list(logits_list))
        labels = torch.cat(list(labels_list))
        energy_p = torch.cat(list(energy_pred))
        energy_t = torch.cat(list(energy_true))

        metrics: Dict[str, float] = {}
        metrics["accuracy"] = metrics_mod.accuracy(logits, labels)
        auc = metrics_mod.roc_auc(logits, labels)
        if auc is not None:
            metrics["roc_auc"] = auc
            metrics["classification_auc"] = auc
        resolution = metrics_mod.energy_resolution(energy_p, energy_t)
        metrics["energy_resolution"] = resolution["resolution"]
        metrics["energy_rmse"] = resolution["rmse"]
        metrics["energy_bias"] = resolution["bias"]
        metrics["regression_mse"] = float(torch.mean((energy_p - energy_t) ** 2).item())
        if log_sigma_list is not None:
            log_sigma = torch.cat(list(log_sigma_list))
            metrics["regression_expected_sigma"] = float(torch.exp(log_sigma).mean().item())
        slope, intercept = metrics_mod.energy_linearity(energy_p, energy_t)
        metrics["linearity_slope"] = slope
        metrics["linearity_intercept"] = intercept
        return metrics


__all__ = ["Trainer", "TrainingConfig"]
