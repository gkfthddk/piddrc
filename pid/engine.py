"""Training and evaluation utilities."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
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
    label_smoothing = 0.0
    use_amp: bool = True
    warmup_steps: int = 0
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    early_stopping_monitor: str = "loss"
    show_progress: bool = True
    profile: bool = False
    profile_dir: str = "profile"


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
                            print(f"Early stopping triggered after {epoch} epochs.")
                            break

            if self.scheduler is not None:
                warmup_steps = self.config.warmup_steps
                if warmup_steps <= 0 or self._global_step >= warmup_steps:
                    self.scheduler.step()
        return history

    def evaluate(
        self, data_loader: DataLoader, *, return_outputs: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, Any]]]]:
        result = self._run_epoch(
            data_loader,
            training=False,
            epoch=None,
            collect_outputs=return_outputs,
        )
        if return_outputs:
            metrics, outputs = result  # type: ignore[misc]
            return metrics, outputs
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        data_loader: DataLoader,
        *,
        training: bool,
        epoch: Optional[int],
        collect_outputs: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, Any]]]]:
        mode = "train" if training else "eval"
        self.model.train(mode == "train")
        if self.config.show_progress:
            iterator = tqdm(
                data_loader,
                desc=f"{mode} epoch {epoch}" if epoch else mode,
                leave=True,
            )
        else:
            iterator = data_loader
        losses: List[float] = []
        cls_losses: List[float] = []
        reg_losses: List[float] = []
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        energy_pred: List[torch.Tensor] = []
        energy_true: List[torch.Tensor] = []
        log_sigma_list: List[torch.Tensor] = []
        event_id_list: List[torch.Tensor] = []
        sigma_values: List[float] = []

        autocast_enabled = self.scaler.is_enabled()

        skipped_entries = 0
        check=0
        for step, batch in enumerate(iterator, start=1):
            batch = self._move_to_device(batch)
            grad_context = nullcontext() if training else torch.no_grad()
            skip_step = False
            effective_batch = batch
            loss: Optional[torch.Tensor] = None
            loss_cls: Optional[torch.Tensor] = None
            loss_reg: Optional[torch.Tensor] = None
            check+=1
            if check==3 and self.config.profile and training:
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                with profile(activities=activities, record_shapes=True) as prof:
                    with record_function("model_inference"):
                        outputs = self.model(batch)
                print('profile results:')
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                
            with grad_context:
                with torch.amp.autocast('cuda',enabled=autocast_enabled):
                    outputs = self.model(batch)
                    valid_mask = self._valid_entry_mask(outputs, batch)
                    invalid_count = int((~valid_mask).sum().item())
                    if invalid_count:
                        skipped_entries += invalid_count
                    if valid_mask.sum().item() == 0:
                        skip_step = True
                    else:
                        if not torch.all(valid_mask):
                            outputs = self._mask_outputs(outputs, valid_mask)
                            effective_batch = self._mask_batch(batch, valid_mask)
                        loss_cls, loss_reg = self._compute_losses(outputs, effective_batch)
                        loss = self.config.classification_weight * loss_cls + self.config.regression_weight * loss_reg

            if skip_step:
                continue

            assert loss is not None and loss_cls is not None and loss_reg is not None

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
            labels_list.append(effective_batch["labels"].detach().cpu())
            energy_pred.append(outputs.energy.detach().cpu())
            energy_true.append(effective_batch["energy"].detach().cpu())
            if collect_outputs and "event_id" in effective_batch:
                event_id_list.append(effective_batch["event_id"].detach().cpu())
            if outputs.log_sigma is not None:
                log_sigma = outputs.log_sigma.detach()
                log_sigma_list.append(log_sigma.cpu())
                sigma_values.append(float(torch.exp(log_sigma).mean().item()))

            if step % self.config.log_every == 0:
                postfix = {
                    "loss": sum(losses) / len(losses),
                    "mse": metrics_mod.mse(torch.cat(energy_pred), torch.cat(energy_true)),
                }
                if sigma_values:
                    postfix["sigma"] = sum(sigma_values) / len(sigma_values)
                try:
                    auc = metrics_mod.roc_auc(torch.cat(logits_list), torch.cat(labels_list))
                except RuntimeError:
                    auc = None
                if auc is not None:
                    postfix["auc"] = auc
                if self.config.show_progress:
                    iterator.set_postfix(postfix)

        metrics = self._compute_metrics(
            logits_list,
            labels_list,
            energy_pred,
            energy_true,
            log_sigma_list if log_sigma_list else None,
        )
        metrics["invalid_entries"] = float(skipped_entries)
        metrics.update({
            "loss": float(sum(losses) / max(len(losses), 1)),
            "loss_cls": float(sum(cls_losses) / max(len(cls_losses), 1)),
            "loss_reg": float(sum(reg_losses) / max(len(reg_losses), 1)),
        })
        metrics["classification_loss"] = metrics["loss_cls"]
        metrics["regression_loss"] = metrics["loss_reg"]

        if not collect_outputs:
            return metrics

        if not logits_list:
            return metrics, []

        logits = torch.cat(list(logits_list))
        labels = torch.cat(list(labels_list))
        energy_p = torch.cat(list(energy_pred))
        energy_t = torch.cat(list(energy_true))

        output_records: List[Dict[str, Any]] = []
        event_ids = (
            torch.cat(event_id_list)
            if event_id_list
            else torch.empty(0, 2, dtype=torch.long)
        )
        log_sigma_tensor = torch.cat(list(log_sigma_list)) if log_sigma_list else None

        for idx in range(logits.shape[0]):
            record: Dict[str, Any] = {
                "event_index": idx,
                "event_id": event_ids[idx].tolist() if idx < event_ids.shape[0] else [],
                "label": int(labels[idx].item()),
                "logits": logits[idx].tolist(),
                "energy_pred": float(energy_p[idx].item()),
                "energy_true": float(energy_t[idx].item()),
            }
            if log_sigma_tensor is not None:
                record["log_sigma"] = float(log_sigma_tensor[idx].item())
            output_records.append(record)

        return metrics, output_records

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
        loss_cls = nn.functional.cross_entropy(outputs.logits, labels, label_smoothing=self.config.label_smoothing)
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
        logits_tensors = list(logits_list)
        labels_tensors = list(labels_list)
        energy_pred_tensors = list(energy_pred)
        energy_true_tensors = list(energy_true)
        log_sigma_tensors = list(log_sigma_list) if log_sigma_list is not None else None

        if not logits_tensors:
            metrics: Dict[str, float] = {
                "accuracy": float("nan"),
                "energy_resolution": float("nan"),
                "energy_rmse": float("nan"),
                "energy_bias": float("nan"),
                "regression_mse": float("nan"),
                "linearity_slope": float("nan"),
                "linearity_intercept": float("nan"),
            }
            metrics["roc_auc"] = float("nan")
            metrics["classification_auc"] = float("nan")
            if log_sigma_tensors is not None:
                metrics["regression_sigma"] = float("nan")
            return metrics

        logits = torch.cat(logits_tensors)
        labels = torch.cat(labels_tensors)
        energy_p = torch.cat(energy_pred_tensors)
        energy_t = torch.cat(energy_true_tensors)

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
        if log_sigma_tensors is not None:
            log_sigma = torch.cat(log_sigma_tensors)
            metrics["regression_sigma"] = float(torch.exp(log_sigma).mean().item())
        slope, intercept = metrics_mod.energy_linearity(energy_p, energy_t)
        metrics["linearity_slope"] = slope
        metrics["linearity_intercept"] = intercept
        return metrics

    def _valid_entry_mask(self, outputs: ModelOutputs, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = outputs.logits
        if logits.ndim > 1:
            logits_finite = torch.isfinite(logits).all(dim=-1)
        else:
            logits_finite = torch.isfinite(logits)
        energy_finite = torch.isfinite(outputs.energy) & torch.isfinite(batch["energy"])
        mask = logits_finite & energy_finite
        if outputs.log_sigma is not None:
            mask = mask & torch.isfinite(outputs.log_sigma)
        return mask

    def _mask_outputs(self, outputs: ModelOutputs, mask: torch.Tensor) -> ModelOutputs:
        mask = mask.to(dtype=torch.bool)
        extras: Dict[str, torch.Tensor] = {}
        for key, value in outputs.extras.items():
            if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == mask.shape[0]:
                extras[key] = value[mask]
            else:
                extras[key] = value
        log_sigma = outputs.log_sigma[mask] if outputs.log_sigma is not None else None
        return ModelOutputs(
            logits=outputs.logits[mask],
            energy=outputs.energy[mask],
            log_sigma=log_sigma,
            extras=extras,
        )

    def _mask_batch(self, batch: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = mask.to(dtype=torch.bool)
        filtered: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == mask.shape[0]:
                filtered[key] = value[mask]
            else:
                filtered[key] = value
        return filtered


__all__ = ["Trainer", "TrainingConfig"]
