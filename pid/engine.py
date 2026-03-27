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
    direction_weight: float = 1.0
    use_direction_regression: bool = False
    max_grad_norm: Optional[float] = None
    label_smoothing: float = 0.0
    use_amp: bool = True
    checkpoint_path: Optional[str] = None
    warmup_steps: int = 0
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    early_stopping_monitor: str = "loss"
    show_progress: bool = True
    freeze_sigma: int = 0
    profile: bool = False
    profile_dir: str = "profile"
    log_batch_lengths: bool = False
    log_cuda_memory: bool = False


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
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch)
                history["val"].append(val_metrics)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
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

            if self.config.checkpoint_path and epoch % self.config.log_every == 0:
                payload = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }
                torch.save(payload, self.config.checkpoint_path)
                print(f"Epoch {epoch}: Checkpoint saved to {self.config.checkpoint_path}")

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
        stream_metrics_only = (not training) and (not collect_outputs)
        losses: List[float] = []
        cls_losses: List[float] = []
        reg_losses: List[float] = []
        dir_losses: List[float] = []
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        energy_pred: List[torch.Tensor] = []
        energy_true: List[torch.Tensor] = []
        direction_pred: List[torch.Tensor] = []
        direction_true: List[torch.Tensor] = []
        log_sigma_list: List[torch.Tensor] = []
        direction_log_sigma_list: List[torch.Tensor] = []
        event_id_list: List[torch.Tensor] = []
        sigma_values: List[float] = []
        direction_sigma_values: List[float] = []
        stream_count = 0
        stream_correct = 0
        stream_sq_err = 0.0
        stream_bias = 0.0
        stream_rel_sum = 0.0
        stream_rel_sq_sum = 0.0
        stream_rel_count = 0
        stream_x_sum = 0.0
        stream_y_sum = 0.0
        stream_xx_sum = 0.0
        stream_xy_sum = 0.0
        stream_dir_sq_err = 0.0
        stream_dir_count = 0
        stream_reg_sigma_sum = 0.0
        stream_reg_sigma_count = 0
        stream_dir_sigma_sum = 0.0
        stream_dir_sigma_count = 0
        padded_lengths: List[int] = []
        valid_lengths_max: List[int] = []
        valid_lengths_mean: List[float] = []

        autocast_enabled = self.scaler.is_enabled()
        log_cuda_memory = self.config.log_cuda_memory and self.device.type == "cuda" and torch.cuda.is_available()
        if log_cuda_memory:
            torch.cuda.reset_peak_memory_stats(self.device)

        skipped_entries = 0
        check=0
        for step, batch in enumerate(iterator, start=1):
            batch = self._move_to_device(batch)
            if self.config.log_batch_lengths and "points" in batch and "mask" in batch:
                padded_lengths.append(int(batch["points"].shape[1]))
                per_event_lengths = batch["mask"].sum(dim=1)
                valid_lengths_max.append(int(per_event_lengths.max().item()))
                valid_lengths_mean.append(float(per_event_lengths.float().mean().item()))
            grad_context = nullcontext() if training else torch.no_grad()
            skip_step = False
            effective_batch = batch
            loss: Optional[torch.Tensor] = None
            loss_cls: Optional[torch.Tensor] = None
            loss_reg: Optional[torch.Tensor] = None
            loss_dir: Optional[torch.Tensor] = None
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
                        loss_cls, loss_reg, loss_dir = self._compute_losses(outputs, effective_batch, epoch)
                        loss = (
                            self.config.classification_weight * loss_cls
                            + self.config.regression_weight * loss_reg
                            + self.config.direction_weight * loss_dir
                        )

            if skip_step:
                continue

            assert loss is not None and loss_cls is not None and loss_reg is not None and loss_dir is not None

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
            dir_losses.append(loss_dir.detach().item())
            if stream_metrics_only:
                with torch.no_grad():
                    logits_detached = outputs.logits.detach()
                    labels_detached = effective_batch["labels"].detach()
                    energy_p_detached = outputs.energy.detach()
                    energy_t_detached = effective_batch["energy"].detach()

                    stream_count += int(labels_detached.numel())
                    stream_correct += int((logits_detached.argmax(dim=1) == labels_detached).sum().item())
                    residual = energy_p_detached - energy_t_detached
                    stream_sq_err += float((residual ** 2).sum().item())
                    stream_bias += float(residual.sum().item())

                    rel_mask = torch.abs(energy_t_detached) > 1e-6
                    if rel_mask.any():
                        rel = residual[rel_mask] / energy_t_detached[rel_mask]
                        stream_rel_sum += float(rel.sum().item())
                        stream_rel_sq_sum += float((rel ** 2).sum().item())
                        stream_rel_count += int(rel.numel())

                    x = energy_t_detached
                    y = energy_p_detached
                    stream_x_sum += float(x.sum().item())
                    stream_y_sum += float(y.sum().item())
                    stream_xx_sum += float((x * x).sum().item())
                    stream_xy_sum += float((x * y).sum().item())

                    if self.config.use_direction_regression and outputs.direction is not None and "direction" in effective_batch:
                        d_residual = outputs.direction.detach() - effective_batch["direction"].detach()
                        stream_dir_sq_err += float((d_residual ** 2).sum().item())
                        stream_dir_count += int(d_residual.numel())

                    if outputs.log_sigma is not None:
                        stream_reg_sigma_sum += float(torch.exp(outputs.log_sigma.detach()).sum().item())
                        stream_reg_sigma_count += int(outputs.log_sigma.numel())
                    if outputs.direction_log_sigma is not None:
                        direction_sigma = torch.exp(outputs.direction_log_sigma.detach())
                        stream_dir_sigma_sum += float(direction_sigma.sum().item())
                        stream_dir_sigma_count += int(direction_sigma.numel())
            else:
                logits_list.append(outputs.logits.detach().cpu())
                labels_list.append(effective_batch["labels"].detach().cpu())
                energy_pred.append(outputs.energy.detach().cpu())
                energy_true.append(effective_batch["energy"].detach().cpu())
                if outputs.direction is not None and "direction" in effective_batch:
                    direction_pred.append(outputs.direction.detach().cpu())
                if "direction" in effective_batch and (self.config.use_direction_regression or collect_outputs):
                    direction_true.append(effective_batch["direction"].detach().cpu())
                if collect_outputs and "event_id" in effective_batch:
                    event_id_list.append(effective_batch["event_id"].detach().cpu())
                if outputs.log_sigma is not None:
                    log_sigma = outputs.log_sigma.detach()
                    log_sigma_list.append(log_sigma.cpu())
                    sigma_values.append(float(torch.exp(log_sigma).mean().item()))
                if outputs.direction_log_sigma is not None:
                    direction_log_sigma = outputs.direction_log_sigma.detach()
                    direction_log_sigma_list.append(direction_log_sigma.cpu())
                    direction_sigma_values.append(float(torch.exp(direction_log_sigma).mean().item()))

            if step % self.config.log_every == 0:
                if stream_metrics_only:
                    mse_value = stream_sq_err / max(stream_count, 1)
                    postfix = {
                        "loss": sum(losses) / len(losses),
                        "mse": mse_value,
                    }
                else:
                    postfix = {
                        "loss": sum(losses) / len(losses),
                        "mse": metrics_mod.mse(torch.cat(energy_pred), torch.cat(energy_true)),
                    }
                if sigma_values:
                    postfix["sigma"] = sum(sigma_values) / len(sigma_values)
                if direction_sigma_values:
                    postfix["dir_sigma"] = sum(direction_sigma_values) / len(direction_sigma_values)
                if not stream_metrics_only:
                    try:
                        auc = metrics_mod.roc_auc(torch.cat(logits_list), torch.cat(labels_list))
                    except RuntimeError:
                        auc = None
                    if auc is not None:
                        postfix["auc"] = auc
                if self.config.show_progress:
                    iterator.set_postfix(postfix)
                if log_cuda_memory and step % self.config.log_every == 0:
                    allocated_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
                    peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    phase = "train" if training else "val"
                    epoch_text = str(epoch) if epoch is not None else "-"
                    print(
                        f"[cuda-mem] phase={phase} epoch={epoch_text} step={step} "
                        f"alloc_gb={allocated_gb:.2f} reserved_gb={reserved_gb:.2f} peak_alloc_gb={peak_gb:.2f}"
                    )

        if stream_metrics_only:
            metrics: Dict[str, float] = {}
            n = max(stream_count, 1)
            metrics["accuracy"] = float(stream_correct / n)
            metrics["regression_mse"] = float(stream_sq_err / n)
            metrics["energy_rmse"] = float((stream_sq_err / n) ** 0.5)
            metrics["energy_bias"] = float(stream_bias / n)
            if stream_rel_count > 0:
                rel_mean = stream_rel_sum / stream_rel_count
                rel_var = max(stream_rel_sq_sum / stream_rel_count - rel_mean * rel_mean, 0.0)
                metrics["energy_resolution"] = float(rel_var ** 0.5)
            else:
                metrics["energy_resolution"] = float("nan")

            denom = stream_count * stream_xx_sum - stream_x_sum * stream_x_sum
            if abs(denom) > 1e-12 and stream_count >= 2:
                slope = (stream_count * stream_xy_sum - stream_x_sum * stream_y_sum) / denom
                intercept = (stream_y_sum - slope * stream_x_sum) / stream_count
                metrics["linearity_slope"] = float(slope)
                metrics["linearity_intercept"] = float(intercept)
            else:
                metrics["linearity_slope"] = float("nan")
                metrics["linearity_intercept"] = float("nan")

            if stream_dir_count > 0:
                metrics["direction_mse"] = float(stream_dir_sq_err / stream_dir_count)
            else:
                metrics["direction_mse"] = float("nan")

            metrics["roc_auc"] = float("nan")
            metrics["classification_auc"] = float("nan")
            if stream_reg_sigma_count > 0:
                metrics["regression_sigma"] = float(stream_reg_sigma_sum / stream_reg_sigma_count)
            if stream_dir_sigma_count > 0:
                metrics["direction_sigma"] = float(stream_dir_sigma_sum / stream_dir_sigma_count)
        else:
            metrics = self._compute_metrics(
                logits_list,
                labels_list,
                energy_pred,
                energy_true,
                direction_pred,
                direction_true,
                log_sigma_list if log_sigma_list else None,
                direction_log_sigma_list if direction_log_sigma_list else None,
            )
        metrics["invalid_entries"] = float(skipped_entries)
        metrics.update({
            "loss": float(sum(losses) / max(len(losses), 1)),
            "loss_cls": float(sum(cls_losses) / max(len(cls_losses), 1)),
            "loss_reg": float(sum(reg_losses) / max(len(reg_losses), 1)),
            "loss_dir": float(sum(dir_losses) / max(len(dir_losses), 1)),
        })
        metrics["classification_loss"] = metrics["loss_cls"]
        metrics["regression_loss"] = metrics["loss_reg"]
        metrics["direction_loss"] = metrics["loss_dir"]
        if self.config.log_batch_lengths and padded_lengths:
            mean_padded = sum(padded_lengths) / len(padded_lengths)
            max_padded = max(padded_lengths)
            mean_valid_max = sum(valid_lengths_max) / len(valid_lengths_max)
            max_valid = max(valid_lengths_max)
            mean_valid = sum(valid_lengths_mean) / len(valid_lengths_mean)
            phase = "train" if training else "val"
            epoch_text = str(epoch) if epoch is not None else "-"
            print(
                f"[batch-length] phase={phase} epoch={epoch_text} "
                f"padded_mean={mean_padded:.1f} padded_max={max_padded} "
                f"valid_max_mean={mean_valid_max:.1f} valid_max={max_valid} "
                f"valid_mean={mean_valid:.1f}"
            )
        if log_cuda_memory:
            peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
            phase = "train" if training else "val"
            epoch_text = str(epoch) if epoch is not None else "-"
            print(
                f"[cuda-mem] phase={phase} epoch={epoch_text} "
                f"peak_alloc_gb={peak_gb:.2f} peak_reserved_gb={peak_reserved_gb:.2f}"
            )

        if not collect_outputs:
            return metrics

        if not logits_list:
            return metrics, []

        logits = torch.cat(list(logits_list))
        labels = torch.cat(list(labels_list))
        energy_p = torch.cat(list(energy_pred))
        energy_t = torch.cat(list(energy_true))
        direction_p = torch.cat(list(direction_pred)) if direction_pred else None
        direction_t = torch.cat(list(direction_true)) if direction_true else None

        output_records: List[Dict[str, Any]] = []
        event_ids = (
            torch.cat(event_id_list)
            if event_id_list
            else torch.empty(0, 2, dtype=torch.long)
        )
        log_sigma_tensor = torch.cat(list(log_sigma_list)) if log_sigma_list else None
        direction_log_sigma_tensor = (
            torch.cat(list(direction_log_sigma_list))
            if direction_log_sigma_list
            else None
        )

        for idx in range(logits.shape[0]):
            record: Dict[str, Any] = {
                "event_index": idx,
                "event_id": event_ids[idx].tolist() if idx < event_ids.shape[0] else [],
                "label": int(labels[idx].item()),
                "logits": logits[idx].tolist(),
                "energy_pred": float(energy_p[idx].item()),
                "energy_true": float(energy_t[idx].item()),
                "E_gen": float(energy_t[idx].item()),
            }
            if direction_t is not None and idx < direction_t.shape[0]:
                record["theta"] = float(direction_t[idx][0].item())
                if direction_t.shape[1] > 1:
                    record["phi"] = float(direction_t[idx][1].item())
                record["direction_true"] = direction_t[idx].tolist()
            if direction_p is not None and idx < direction_p.shape[0]:
                record["direction_pred"] = direction_p[idx].tolist()
            if log_sigma_tensor is not None:
                record["log_sigma"] = float(log_sigma_tensor[idx].item())
            if direction_log_sigma_tensor is not None:
                record["direction_log_sigma"] = direction_log_sigma_tensor[idx].tolist()
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

    def _compute_losses(
        self, outputs: ModelOutputs, batch: Dict[str, torch.Tensor], epoch: int=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch["labels"]
        energy = batch["energy"]
        loss_cls = nn.functional.cross_entropy(outputs.logits, labels, label_smoothing=self.config.label_smoothing)
        if outputs.log_sigma is not None:
            log_sigma = outputs.log_sigma.clamp(min=-3.0, max=5.0)
            if(epoch is not None and epoch < self.config.freeze_sigma):
                log_sigma = torch.zeros_like(log_sigma).detach()
            residual = energy - outputs.energy
            loss_reg = 0.5 * torch.exp(-2.0 * log_sigma) * (residual ** 2) + log_sigma
            loss_reg = loss_reg.mean()
        else:
            loss_reg = nn.functional.mse_loss(outputs.energy, energy)
        if self.config.use_direction_regression:
            if outputs.direction is None:
                raise RuntimeError("Direction regression enabled but model does not output direction predictions.")
            if "direction" not in batch:
                raise RuntimeError("Direction regression enabled but batch has no 'direction' target.")
            residual = batch["direction"] - outputs.direction
            if outputs.direction_log_sigma is not None:
                direction_log_sigma = outputs.direction_log_sigma.clamp(min=-3.0, max=5.0)
                if epoch is not None and epoch < self.config.freeze_sigma:
                    direction_log_sigma = torch.zeros_like(direction_log_sigma).detach()
                loss_dir = 0.5 * torch.exp(-2.0 * direction_log_sigma) * (residual ** 2) + direction_log_sigma
                loss_dir = loss_dir.mean()
            else:
                loss_dir = nn.functional.mse_loss(outputs.direction, batch["direction"])
        else:
            loss_dir = torch.zeros((), device=outputs.energy.device, dtype=outputs.energy.dtype)
        return loss_cls, loss_reg, loss_dir

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _compute_metrics(
        self,
        logits_list: Iterable[torch.Tensor],
        labels_list: Iterable[torch.Tensor],
        energy_pred: Iterable[torch.Tensor],
        energy_true: Iterable[torch.Tensor],
        direction_pred: Iterable[torch.Tensor],
        direction_true: Iterable[torch.Tensor],
        log_sigma_list: Optional[Iterable[torch.Tensor]] = None,
        direction_log_sigma_list: Optional[Iterable[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        logits_tensors = list(logits_list)
        labels_tensors = list(labels_list)
        energy_pred_tensors = list(energy_pred)
        energy_true_tensors = list(energy_true)
        direction_pred_tensors = list(direction_pred)
        direction_true_tensors = list(direction_true)
        log_sigma_tensors = list(log_sigma_list) if log_sigma_list is not None else None
        direction_log_sigma_tensors = (
            list(direction_log_sigma_list)
            if direction_log_sigma_list is not None
            else None
        )

        if not logits_tensors:
            metrics: Dict[str, float] = {
                "accuracy": float("nan"),
                "energy_resolution": float("nan"),
                "energy_rmse": float("nan"),
                "energy_bias": float("nan"),
                "regression_mse": float("nan"),
                "direction_mse": float("nan"),
                "linearity_slope": float("nan"),
                "linearity_intercept": float("nan"),
            }
            metrics["roc_auc"] = float("nan")
            metrics["classification_auc"] = float("nan")
            if log_sigma_tensors is not None:
                metrics["regression_sigma"] = float("nan")
            if direction_log_sigma_tensors is not None:
                metrics["direction_sigma"] = float("nan")
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
        if direction_pred_tensors and direction_true_tensors:
            direction_p = torch.cat(direction_pred_tensors)
            direction_t = torch.cat(direction_true_tensors)
            metrics["direction_mse"] = float(torch.mean((direction_p - direction_t) ** 2).item())
        else:
            metrics["direction_mse"] = float("nan")
        if log_sigma_tensors is not None:
            log_sigma = torch.cat(log_sigma_tensors)
            metrics["regression_sigma"] = float(torch.exp(log_sigma).mean().item())
        if direction_log_sigma_tensors is not None:
            direction_log_sigma = torch.cat(direction_log_sigma_tensors)
            metrics["direction_sigma"] = float(torch.exp(direction_log_sigma).mean().item())
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
        if self.config.use_direction_regression and outputs.direction is not None and "direction" in batch:
            direction_finite = torch.isfinite(outputs.direction).all(dim=-1) & torch.isfinite(batch["direction"]).all(dim=-1)
            mask = mask & direction_finite
        if outputs.direction_log_sigma is not None:
            mask = mask & torch.isfinite(outputs.direction_log_sigma).all(dim=-1)
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
        direction = outputs.direction[mask] if outputs.direction is not None else None
        direction_log_sigma = (
            outputs.direction_log_sigma[mask]
            if outputs.direction_log_sigma is not None
            else None
        )
        return ModelOutputs(
            logits=outputs.logits[mask],
            energy=outputs.energy[mask],
            log_sigma=log_sigma,
            direction=direction,
            direction_log_sigma=direction_log_sigma,
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
