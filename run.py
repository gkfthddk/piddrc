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
import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
torch.set_num_threads(16)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from pid import DualReadoutEventDataset, Trainer, TrainingConfig, collate_events
from pid.data import EventRecord
from pid.models.base import ModelOutputs, MultiTaskHead
from pid.models.pointset_mamba import PointSetMamba
from pid.models.pointset_mlp import PointSetMLP
from pid.models.pointset_ptv3 import PointSetTransformerV3
from pid.models.pointset_mlppp import PointSetMLPpp
from pid.models.pointset_transformer import PointSetTransformer

MODEL_REGISTRY = {
    "mlp": PointSetMLP,
    "transformer": PointSetTransformer,
    "mamba": PointSetMamba,
    "ptv3": PointSetTransformerV3,
    "mlppp": PointSetMLPpp,
    "mamba2": PointSetMamba,
}


def _write_test_outputs(records: Sequence[Dict[str, Any]], destination: Path | None) -> None:
    """Persist per-event evaluation outputs to JSON in a compact form."""

    output_path = destination or Path("test_outputs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")


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
        "--test_files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional test HDF5 files",
    )
    io_group.add_argument(
        "--stat_file",
        type=Path,
        default="h5s/stats_1-100GeV.yaml",
        help="input channel statistics yaml files",
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
        "--pool",
        type=int,
        default=1,
        help="Pooling factor for down-sampling hits (1 = no pooling)",
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
        help="Down-sample each event to this many hits",
    )
    io_group.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of the training data to reserve for validation when no validation files are provided",
    )
    io_group.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of the training data to reserve for testing when no test files are provided",
    )
    io_group.add_argument(
        "--split_seed",
        type=int,
        default=1234,
        help="Random seed used when splitting the training dataset",
    )
    io_group.add_argument(
        "--no_dataset_progress",
        dest="dataset_progress",
        action="store_false",
        help="Disable dataset initialization progress messages",
    )
    io_group.add_argument(
        "--balance_train_files",
        action="store_true",
        help="Truncate each training file to match the smallest event count",
    )
    io_group.add_argument(
        "--train_limit",
        type=int,
        default=None,
        help="Optional per-file limit on the number of training events for quick tests",
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
        default=64,
        help="Hidden dimension used by the backbone",
    )
    model_group.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Number of layers/blocks in the backbone",
    )
    model_group.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads (Transformer only)",
    )
    model_group.add_argument(
        "--mlp_expansion",
        type=float,
        default=4.0,
        help="Feed-forward expansion ratio (Transformer and MLP++ only)",
    )
    model_group.add_argument(
        "--k_neighbors",
        type=int,
        default=16,
        help="Number of neighbors for local attention (PTv3 only)",
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
        default=(512, 256, 128),
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
    train_group.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross-entropy loss (e.g., 0.1)",
    )
    train_group.add_argument("--lr_scheduler", type=str, default=None, choices=["cosine", "step", "exponential"])
    train_group.add_argument("--log_every", type=int, default=10)
    train_group.add_argument("--max_grad_norm", type=float, default=5.0)
    train_group.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision")
    train_group.add_argument("--num_workers", type=int, default=8)

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
        "--metrics_json",
        type=Path,
        default=None,
        help="Optional path to dump the evaluation metrics as JSON",
    )
    misc_group.add_argument(
        "--config_json",
        type=Path,
        default=None,
        help="Optional path to dump the resolved CLI arguments as JSON",
    )
    misc_group.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only run evaluation using the provided checkpoint",
    )
    misc_group.add_argument(
        "--model_summary",
        action="store_true",
        help="Print a torchinfo overview of the model before training",
    )
    misc_group.add_argument(
        "--no_progress_bar",
        dest="progress_bar",
        action="store_false",
        help="Disable tqdm progress bars during training and evaluation",
    )
    misc_group.add_argument(
        "--name",
        type=str,
        default="test",
        help=(
            "Optional identifier used to derive default artifact paths for the "
            "checkpoint, training history and evaluation metrics"
        ),
    )
    misc_group.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help=(
            "Optional path to save per-event model outputs when evaluating the test set. "
            "When omitted, 'test_outputs.json' is used."
        ),
    )
    misc_group.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile() for model acceleration (PyTorch 2.0+)",
    )
    misc_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable the PyTorch profiler for a few steps to analyze performance.",
    )
    misc_group.add_argument(
        "--profile_dir",
        type=Path,
        default=None,
        help="Directory to save profiler traces. Defaults to 'save/<name>/profile'.",
    )
    misc_group.add_argument(
        "--freeze_sigma",
        type=int,
        default=0,
        help="Freeze the uncertainty head (log_sigma) to zero for the first N epochs.",
    )
    parser.set_defaults(dataset_progress=True, progress_bar=True)

    args = parser.parse_args(argv)

    setattr(args, "instance_name", args.name)

    if args.name:
        base_dir = Path("save") / args.name
        if args.history_json is None:
            args.history_json = base_dir / "history.json"
        if args.checkpoint is None:
            args.checkpoint = base_dir / "checkpoint.pt"
        if args.metrics_json is None:
            args.metrics_json = base_dir / "metrics.json"
        if args.config_json is None:
            args.config_json = base_dir / "config.json"
        if args.output_json is None:
            args.output_json = base_dir / "output.json"
        if args.profile_dir is None:
            args.profile_dir = base_dir / "profile"
    if args.pool > 1:
        args.hit_features = [feat.replace("DRcalo3dHits",f"DRcalo3dHits{args.pool}") for feat in args.hit_features]

    return args


def _split_dataset(
    dataset: DualReadoutEventDataset,
    *,
    val_fraction: float,
    test_fraction: float,
    need_val: bool,
    need_test: bool,
    seed: int,
) -> Tuple[Dataset[EventRecord], Dataset[EventRecord] | None, Dataset[EventRecord] | None]:
    if not need_val and not need_test:
        return dataset, None, None

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the [0, 1) range")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in the [0, 1) range")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("The sum of val_fraction and test_fraction must be < 1")

    total = len(dataset)
    if total == 0:
        raise ValueError("Cannot split an empty dataset")

    val_len = 0
    if need_val and val_fraction > 0.0:
        val_len = max(int(total * val_fraction), 1)

    test_len = 0
    if need_test and test_fraction > 0.0:
        test_len = max(int(total * test_fraction), 1)

    train_len = total - val_len - test_len
    if train_len <= 0:
        raise ValueError(
            "Not enough events to perform the requested split. "
            "Consider lowering val_fraction/test_fraction."
        )

    lengths: List[int] = [train_len]
    include_val = val_len > 0
    include_test = test_len > 0
    if include_val:
        lengths.append(val_len)
    if include_test:
        lengths.append(test_len)

    if len(lengths) == 1:
        return dataset, None, None

    generator = torch.Generator()
    generator.manual_seed(seed)
    subsets = random_split(dataset, lengths, generator=generator)

    subset_iter = iter(subsets)
    train_subset = next(subset_iter)
    val_subset = next(subset_iter) if include_val else None
    test_subset = next(subset_iter) if include_test else None

    def _indices_to_set(subset: Dataset[EventRecord] | None) -> set[int]:
        if subset is None:
            return set()
        indices = getattr(subset, "indices", None)
        if indices is None:
            return set()
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return {int(idx) for idx in indices}

    train_indices = _indices_to_set(train_subset)
    val_indices = _indices_to_set(val_subset)
    test_indices = _indices_to_set(test_subset)

    if train_indices & val_indices:
        raise RuntimeError("Training and validation splits share overlapping events")
    if train_indices & test_indices:
        raise RuntimeError("Training and test splits share overlapping events")
    if val_indices & test_indices:
        raise RuntimeError("Validation and test splits share overlapping events")

    return train_subset, val_subset, test_subset


def build_datasets(
    args: argparse.Namespace,
) -> Tuple[
    DualReadoutEventDataset,
    Dataset[EventRecord],
    Dataset[EventRecord] | None,
    Dataset[EventRecord] | None,
]:
    balance_train_files = getattr(args, "balance_train_files", False)
    train_limit = getattr(args, "train_limit", None)
    progress = getattr(args, "dataset_progress", True)

    print("Preparing datasets...", flush=True)
    print("  Loading training dataset", flush=True)
    pool = getattr(args, "pool", 1)

    base_dataset = DualReadoutEventDataset(
        [str(path) for path in args.train_files],
        hit_features=args.hit_features,
        label_key=args.label_key,
        energy_key=args.energy_key,
        stat_file=args.stat_file,
        max_points=args.max_points,
        pool=pool,
        balance_files=balance_train_files,
        max_events=train_limit,
        progress=progress,
    )
    print("  Training dataset ready", flush=True)

    val_dataset = None
    if args.val_files:
        print("  Loading validation dataset", flush=True)
        val_dataset = DualReadoutEventDataset(
            [str(path) for path in args.val_files],
            hit_features=args.hit_features,
            label_key=args.label_key,
            energy_key=args.energy_key,
            stat_file=args.stat_file,
            max_points=args.max_points,
            pool=pool,
            class_names=base_dataset.classes,
            progress=progress,
        )
        print("  Validation dataset ready", flush=True)

    test_dataset = None
    if args.test_files:
        print("  Loading test dataset", flush=True)
        test_dataset = DualReadoutEventDataset(
            [str(path) for path in args.test_files],
            hit_features=args.hit_features,
            label_key=args.label_key,
            energy_key=args.energy_key,
            stat_file=args.stat_file,
            max_points=args.max_points,
            pool=pool,
            class_names=base_dataset.classes,
            progress=progress,
        )
        print("  Test dataset ready", flush=True)

    train_dataset: Dataset[EventRecord] = base_dataset
    need_val_split = val_dataset is None
    need_test_split = test_dataset is None

    if need_val_split or need_test_split:
        train_dataset, split_val_dataset, split_test_dataset = _split_dataset(
            base_dataset,
            val_fraction=args.val_fraction if need_val_split else 0.0,
            test_fraction=args.test_fraction if need_test_split else 0.0,
            need_val=need_val_split,
            need_test=need_test_split,
            seed=args.split_seed,
        )
        if need_val_split:
            val_dataset = split_val_dataset
        if need_test_split:
            test_dataset = split_test_dataset

    return base_dataset, train_dataset, val_dataset, test_dataset


def print_dataset_summary(
    base_dataset: DualReadoutEventDataset,
    train_dataset: Dataset[EventRecord],
    val_dataset: Dataset[EventRecord] | None,
    test_dataset: Dataset[EventRecord] | None,
) -> None:
    """Log a concise overview of the configured datasets."""

    base_event_count = len(base_dataset)
    hit_features = ", ".join(base_dataset.hit_features)
    class_names = ", ".join([str(c) for c in base_dataset.classes])
    summary_dim = base_dataset[0].summary.numel() if base_event_count > 0 else 0

    print("Dataset summary:")
    print(f"  Training files: {len(base_dataset.files)}")
    print(f"  Base events: {base_event_count:,}")
    print(f"  Hit features ({len(base_dataset.hit_features)}): {hit_features}")
    print(f"  Summary dimension: {summary_dim}")
    print(f"  Classes ({len(base_dataset.classes)}): {class_names}")
    max_points = getattr(base_dataset, "max_points", None)
    print("  Max points per event: {}".format(max_points if max_points is not None else "unbounded"))

    def _describe_split(name: str, dataset: Dataset[EventRecord] | None) -> None:
        if dataset is None:
            print(f"    {name:<5}: not provided")
            return
        size = len(dataset)
        details = [f"{size:,} events"]
        if isinstance(dataset, Subset) and base_event_count > 0:
            fraction = size / base_event_count
            details.append(f"{fraction:.1%} of base")
        print(f"    {name:<5}: {', '.join(details)}")

    print("  Splits:")
    _describe_split("train", train_dataset)
    _describe_split("val", val_dataset)
    _describe_split("test", test_dataset)


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
            mlp_ratio=args.mlp_expansion,
            **common_kwargs,
        )
    elif model_name in {"mamba", "mamba2"}:
        backend = "mamba2" if model_name == "mamba2" else "mamba"
        model = model_cls(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            backend=backend,
            **common_kwargs,
        )
    elif model_name == "ptv3":
        model = model_cls(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            k_neighbors=args.k_neighbors,
            **common_kwargs,
        )
    elif model_name == "mlppp":
        model = model_cls(
            embed_dim=args.hidden_dim,
            depth=args.depth,
            expansion=args.mlp_expansion,
            **common_kwargs,
            )
    else:  # pragma: no cover - choices enforced by argparse
        raise ValueError(f"Unknown model type: {args.model}")

    return model


def build_dataloaders(
    train_dataset: Dataset[EventRecord],
    val_dataset: Dataset[EventRecord] | None,
    test_dataset: Dataset[EventRecord] | None,
    *,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None]:
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
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def print_model_summary(
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
            summary(model, input_data=(sample_batch,), depth=6, col_names=("input_size", "output_size", "num_params"))
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
    label_smoothing: float,
    weight_decay: float,
    epochs: int,
    log_every: int,
    max_grad_norm: float | None,
    use_amp: bool,
    show_progress: bool,
    lr_scheduler_name: str | None,
    profile: bool,
    profile_dir: Path | None,
) -> Tuple[Trainer, torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    config = TrainingConfig(
        epochs=epochs,
        log_every=log_every,
        max_grad_norm=max_grad_norm,
        label_smoothing=label_smoothing, 
        use_amp=use_amp,
        show_progress=show_progress,
        early_stopping_patience=5,
        profile=profile,
        profile_dir=str(profile_dir) if profile_dir else "profile",
    )

    scheduler = None
    if lr_scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif lr_scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif lr_scheduler_name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device, config=config
    )
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


def maybe_save_metrics(metrics: Dict[str, Dict[str, float]], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def _serialise_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialise_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialise_value(val) for key, val in value.items()}
    return value


def maybe_save_config(args: argparse.Namespace, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: _serialise_value(value) for key, value in vars(args).items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def maybe_load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch < 2.1 compatibility
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])


def main() -> None:
    start_dt=datetime.datetime.now()
    print(f"{os.path.basename(__file__)} started at {start_dt.isoformat(sep=' ', timespec='seconds')}")
    args = parse_args()

    maybe_save_config(args, args.config_json)

    device = torch.device(args.device)

    base_dataset, train_dataset, val_dataset, test_dataset = build_datasets(args)
    print_dataset_summary(base_dataset, train_dataset, val_dataset, test_dataset)
    model = build_model(args, base_dataset)
    model.to(device)
    
    # Add torch.compile for a potential speed-up on PyTorch 2.0+
    if hasattr(torch, "compile") and args.compile:
        compile_mode = "reduce-overhead"
        if args.profile:
            compile_mode = "default"
        print(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        model = torch.compile(model, mode=compile_mode)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print_model_summary(
        model,
        train_loader,
        device,
        enabled=args.model_summary,
    )

    trainer, optimizer = configure_trainer(
        model,
        device=device,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_every=args.log_every,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        show_progress=args.progress_bar,
        lr_scheduler_name=args.lr_scheduler,
        profile=args.profile,
        profile_dir=args.profile_dir,
    )

    if args.checkpoint is not None and args.eval_only:
        maybe_load_checkpoint(model, optimizer, args.checkpoint)
        eval_targets: Dict[str, DataLoader] = {}
        if test_loader is not None:
            eval_targets["test"] = test_loader
        if val_loader is not None:
            eval_targets.setdefault("val", val_loader)
        if not eval_targets:
            eval_targets["train"] = train_loader
        metrics: Dict[str, Any] = {}
        test_outputs = None
        for split, loader in eval_targets.items():
            if split == "test":
                split_metrics, test_outputs = trainer.evaluate(
                    loader, return_outputs=True
                )
                metrics[split] = split_metrics
            else:
                metrics[split] = trainer.evaluate(loader)
        print(json.dumps(metrics, indent=2))
        maybe_save_metrics(metrics, args.metrics_json)
        if test_outputs is not None:
            _write_test_outputs(test_outputs, args.output_json)
        return

    if args.checkpoint is not None:
        maybe_load_checkpoint(model, optimizer, args.checkpoint if args.checkpoint.exists() else None)

    history = trainer.fit(train_loader, val_loader)
    maybe_save_history(history, args.history_json)
    maybe_save_checkpoint(model, optimizer, args.checkpoint)

    evaluation_loaders: Dict[str, DataLoader] = {}
    if val_loader is not None:
        evaluation_loaders["val"] = val_loader
    if test_loader is not None:
        evaluation_loaders["test"] = test_loader
    if not evaluation_loaders:
        evaluation_loaders["train"] = train_loader
  
    metrics: Dict[str, Any] = {}
    test_outputs = None
    for split, loader in evaluation_loaders.items():
        if split == "test":
            split_metrics, test_outputs = trainer.evaluate(loader, return_outputs=True)
            metrics[split] = split_metrics
        else:
            metrics[split] = trainer.evaluate(loader)
    print(json.dumps(metrics, indent=2))
    maybe_save_metrics(metrics, args.metrics_json)
    if test_outputs is not None:
        _write_test_outputs(test_outputs, args.output_json)
    end_dt=datetime.datetime.now()
    print(f"{os.path.basename(__file__)} ended at {end_dt.isoformat(sep=' ', timespec='seconds')}")
    print(f"Total running time: {end_dt - start_dt}")

if __name__ == "__main__":
    main()
