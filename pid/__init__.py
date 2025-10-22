"""High-level utilities for dual-readout calorimeter deep learning experiments."""

from .data import DualReadoutEventDataset, collate_events
from .engine import Trainer, TrainingConfig

__all__ = [
    "DualReadoutEventDataset",
    "collate_events",
    "Trainer",
    "TrainingConfig",
]
