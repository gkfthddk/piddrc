#!/usr/bin/env python
"""Training entry-point that wires together the refactored point-set models.

This script relies exclusively on the modules living under ``pid/`` for
model construction, data loading and the training loop.  It intentionally avoids
the legacy ``pm`` implementation so that running ``python run.py`` exercises the
new, shared ``PointSetAggregator`` pathway.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch
torch.set_num_threads(16)
from torch import nn
from torch.utils.data import DataLoader

from pid import DualReadoutEventDataset, Trainer, TrainingConfig, collate_events
from pid.models.base import ModelOutputs, MultiTaskHead
from pid.models.pointset_mamba import PointSetMamba
from pid.models.pointset_mlp import PointSetMLP
from pid.models.pointset_transformer import PointSetTransformer

MODEL_REGISTRY = {
    "mlp": PointSetMLP,
    "transformer": PointSetTransformer,
    "mamba": PointSetMamba,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    gpus_parser = argparse.ArgumentParser(add_help=False)
    gpus_parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of CUDA device indices to expose via "
            "CUDA_VISIBLE_DEVICES"
        ),
    )

    gpus_args, _ = gpus_parser.parse_known_args(argv)
    if gpus_args.gpus is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_args.gpus

    parser = argparse.ArgumentParser(description="Train dual-readout point-set models")

    io_group = parser.add_argument_group("Data")
    io_group.add_argument(
        "--train_files",
        type=Path,
        nargs="+",
        default=("h5s/e-_1-100GeV.h5py", "h5s/gamma_1-100GeV.h5py", "h5s/pi0_1-100GeV.h5py", "h5s/pi+_1-100GeV.h5py"),
        help="Paths to the training HDF5 files",
    )
    io_group.add_argument(
        "--val_files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional validation HDF5 files",
    )
    io_group.add_argument(
        "--hit_features",
        type=str,
        nargs="+",
        default=(
            "DRcalo3dHits.amplitude_sum",
            "DRcalo3dHits.type",
            "DRcalo3dHits.time",
            "DRcalo3dHits.time_end",
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
            ),
        help="Hit-level feature names to load from the HDF5 files",
    )
    io_group.add_argument(
        "--label_key",
        type=str,
        default="GenParticles.PDG",
        help="Dataset name containing the classification label",
    )
    io_group.add_argument(
        "--energy_key",
        type=str,
        default="E_gen",
        help="Dataset name containing the regression target (energy)",
    )
    io_group.add_argument(
        "--max_points",
        type=int,
        default=1000,
        help="Randomly down-sample each event to this many hits",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default="mamba",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Which point-set backbone to train",
    )
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension used by the backbone",
    )
    model_group.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of layers/blocks in the backbone",
    )
    model_group.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads (Transformer only)",
    )
    model_group.add_argument(
        "--mlp_ratio",
        type=float,
        default=4.0,
        help="Feed-forward expansion ratio (Transformer only)",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied throughout the model",
    )
    model_group.add_argument(
        "--head_hidden",
        type=int,
        nargs="+",
        default=(256, 128),
        help="Hidden dimensions of the multi-task prediction head",
    )
    model_group.add_argument(
        "--disable_summary",
        action="store_true",
        help="Do not fuse per-event summary features into the head",
    )
    model_group.add_argument(
        "--disable_uncertainty",
        action="store_true",
        help="Disable the log-variance regression head",
    )

    train_group = parser.add_argument_group("Optimisation")
    train_group.add_argument("--batch_size", type=int, default=64)
    train_group.add_argument("--epochs", type=int, default=100)
    train_group.add_argument("--learning_rate", type=float, default=3e-4)
    train_group.add_argument("--weight_decay", type=float, default=1e-2)
    train_group.add_argument("--log_every", type=int, default=10)
    train_group.add_argument("--max_grad_norm", type=float, default=5.0)
    train_group.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    train_group.add_argument("--num_workers", type=int, default=4)

    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--gpus",
        type=str,
        default=gpus_args.gpus,
        help=(
            "Optional comma-separated list of CUDA device indices to expose via "
            "CUDA_VISIBLE_DEVICES"
        ),
    )
    misc_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string to use for training",
    )
    misc_group.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to save the trained model weights",
    )
    misc_group.add_argument(
        "--history_json",
        type=Path,
        default=None,
        help="Optional path to dump the training/validation metrics as JSON",
    )
    misc_group.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only run evaluation using the provided checkpoint",
    )
    misc_group.add_argument(
        "--print_model_summary",
        action="store_true",
        help="Print a torchinfo overview of the model before training",
    )

    return parser.parse_args(argv)


def build_datasets(args: argparse.Namespace) -> Tuple[DualReadoutEventDataset, DualReadoutEventDataset | None]:
    train_dataset = DualReadoutEventDataset(
        [str(path) for path in args.train_files],
        hit_features=args.hit_features,
        label_key=args.label_key,
        energy_key=args.energy_key,
        max_points=args.max_points,
    )

    val_dataset = None
    if args.val_files:
        val_dataset = DualReadoutEventDataset(
            [str(path) for path in args.val_files],
            hit_features=args.hit_features,
            label_key=args.label_key,
            energy_key=args.energy_key,
            max_points=args.max_points,
            class_names=train_dataset.classes,
        )

    return train_dataset, val_dataset


def build_model(args: argparse.Namespace, dataset: DualReadoutEventDataset) -> nn.Module:
    model_name = args.model.lower()
    model_cls = MODEL_REGISTRY[model_name]
    head_hidden = tuple(int(dim) for dim in args.head_hidden)
    summary_dim = dataset[0].summary.numel()
    in_channels = len(dataset.hit_features)
    num_classes = len(dataset.classes)

    common_kwargs: Dict[str, Any] = {
        "in_channels": in_channels,
        "summary_dim": summary_dim,
        "num_classes": num_classes,
        "head_hidden": head_hidden,
        "dropout": args.dropout,
        "use_summary": not args.disable_summary,
        "use_uncertainty": not args.disable_uncertainty,
    }

    if model_name == "mlp":
        model = model_cls(**common_kwargs)
    elif model_name == "transformer":
        model = model_cls(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            **common_kwargs,
        )
    elif model_name == "mamba":
        model = model_cls(hidden_dim=args.hidden_dim, depth=args.depth, **common_kwargs)
    else:  # pragma: no cover - choices enforced by argparse
        raise ValueError(f"Unknown model type: {args.model}")

    return model


def build_dataloaders(
    train_dataset: DualReadoutEventDataset,
    val_dataset: DualReadoutEventDataset | None,
    *,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader | None]:
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_events,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def maybe_print_model_summary(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return

    try:
        from torchinfo import summary
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "The torchinfo package is required to print the model overview. "
            "Install it via 'pip install torchinfo'."
        ) from exc

    sample_batch = next(iter(train_loader))
    sample_batch = {
        key: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
        for key, tensor in sample_batch.items()
    }

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def convert_outputs(output: Any) -> Any:
        if not isinstance(output, ModelOutputs):
            return output
        tensors: list[torch.Tensor] = [output.logits, output.energy]
        if output.log_sigma is not None:
            tensors.append(output.log_sigma)
        tensors.extend(output.extras.values())
        return tuple(tensors)

    for module in model.modules():
        if isinstance(module, MultiTaskHead):
            handle = module.register_forward_hook(
                lambda _module, _inputs, output: convert_outputs(output)
            )
            hook_handles.append(handle)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            summary(model, input_data=(sample_batch,),col_names=("input_size", "output_size", "num_params"))
    finally:
        for handle in hook_handles:
            handle.remove()
    if was_training:
        model.train()


def configure_trainer(
    model: nn.Module,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    log_every: int,
    max_grad_norm: float | None,
    use_amp: bool,
) -> Tuple[Trainer, torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    config = TrainingConfig(
        epochs=epochs,
        log_every=log_every,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
    )
    trainer = Trainer(model=model, optimizer=optimizer, device=device, config=config)
    return trainer, optimizer


def maybe_save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, path)


def maybe_save_history(history: Dict[str, Iterable[Dict[str, float]]], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {split: list(records) for split, records in history.items()}
    path.write_text(json.dumps(serialisable, indent=2))


def maybe_load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path | None) -> None:
    if path is None or not path.exists():
        return
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    train_dataset, val_dataset = build_datasets(args)
    model = build_model(args, train_dataset)
    model.to(device)

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    maybe_print_model_summary(
        model,
        train_loader,
        device,
        enabled=args.print_model_summary,
    )

    trainer, optimizer = configure_trainer(
        model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_every=args.log_every,
        max_grad_norm=args.max_grad_norm,
        use_amp=not args.no_amp,
    )

    if args.checkpoint is not None and args.eval_only:
        maybe_load_checkpoint(model, optimizer, args.checkpoint)
        metrics = trainer.evaluate(val_loader or train_loader)
        print(json.dumps(metrics, indent=2))
        return

    if args.checkpoint is not None:
        maybe_load_checkpoint(model, optimizer, args.checkpoint if args.checkpoint.exists() else None)

    history = trainer.fit(train_loader, val_loader)
    maybe_save_history(history, args.history_json)
    maybe_save_checkpoint(model, optimizer, args.checkpoint)

    eval_loader = val_loader or train_loader
    metrics = trainer.evaluate(eval_loader)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
