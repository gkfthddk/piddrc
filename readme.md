# Dual-Readout Calorimeter PID Toolkit

A PyTorch research toolkit for **particle identification (PID)** and **energy regression** on dual-readout calorimeter events stored in HDF5.

It provides:
- A reusable dataset layer for variable-length hit clouds.
- Multiple point-set backbones (MLP, Transformer, Mamba, graph, Point Transformer v3, MLP++).
- A multi-task training engine for classification + regression (with optional uncertainty and direction heads).
- A CLI (`run.py`) for training, evaluation, checkpointing, and metrics export.

---

## Repository layout

```text
.
├── pid/
│   ├── data.py                 # Dataset + collate utilities
│   ├── engine.py               # Trainer + TrainingConfig
│   ├── metrics.py              # Task metrics
│   └── models/                 # Model backbones and shared heads
├── run.py                      # Main train/eval CLI
├── compute_stats.py            # Compute feature statistics from HDF5
├── toh5.py                     # Convert ROOT-style inputs to HDF5
├── submit.py                   # Batch/sweep launcher
├── plot_*.py                   # Plotting/evaluation helper scripts
└── tests/                      # Pytest suite
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data expectations

The training pipeline expects HDF5 files containing:
- **Per-hit features** (e.g., amplitude, type, timing, xyz coordinates).
- **Event-level labels** for class targets (`--label_key`, default `GenParticles.PDG`).
- **Event-level regression targets** for energy (`--energy_key`, default `E_gen`).

Default hit features in `run.py` are:
- `DRcalo3dHits.amplitude_sum`
- `DRcalo3dHits.type`
- `DRcalo3dHits.time`
- `DRcalo3dHits.time_end`
- `DRcalo3dHits.position.x`
- `DRcalo3dHits.position.y`
- `DRcalo3dHits.position.z`

If your schema differs, pass custom `--hit_features`, `--pos_keys`, `--label_key`, and `--energy_key`.

---

## Quick start (CLI)

### 1) Compute normalization statistics

```bash
python compute_stats.py \
  --data_dir /path/to/h5s \
  --files train_electron train_pion \
  --output stats.yaml
```

### 2) Train a model

```bash
python run.py \
  --train_files train_electron.h5 train_pion.h5 \
  --val_files val_electron.h5 val_pion.h5 \
  --stat_file stats.yaml \
  --model mamba \
  --epochs 50 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --name exp_mamba
```

Artifacts are written under `save/<name>/` by default:
- `checkpoint.pt`
- `history.json`
- `metrics.json`
- `config.json`
- `output.json` (test-set per-event outputs)

### 3) Evaluate only

```bash
python run.py \
  --eval_only \
  --checkpoint save/exp_mamba/checkpoint.pt \
  --test_files test_electron.h5 test_pion.h5 \
  --stat_file stats.yaml \
  --name exp_mamba_eval
```

---

## Supported models

Select with `--model`:
- `mlp`
- `transformer`
- `mamba`
- `graph`
- `ptv3`
- `mlppp`
- `mamba2` (alias to current Mamba implementation)

---

## Python API usage

```python
from torch.utils.data import DataLoader
from pid import DualReadoutEventDataset, Trainer, TrainingConfig, collate_events
from pid.models.pointset_mamba import PointSetMamba
import torch

dataset = DualReadoutEventDataset(
    files=["train_electron.h5", "train_pion.h5"],
    hit_features=(
        "DRcalo3dHits.amplitude_sum",
        "DRcalo3dHits.type",
        "DRcalo3dHits.time",
        "DRcalo3dHits.time_end",
        "DRcalo3dHits.position.x",
        "DRcalo3dHits.position.y",
        "DRcalo3dHits.position.z",
    ),
    label_key="GenParticles.PDG",
    energy_key="E_gen",
    stat_file="stats.yaml",
)

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_events)

sample = dataset[0]
model = PointSetMamba(
    in_channels=sample.points.shape[-1],
    summary_dim=sample.summary.numel(),
    num_classes=len(dataset.classes),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
trainer = Trainer(model=model, optimizer=optimizer, device=torch.device("cuda"))
trainer.fit(loader)
```

---

## Analysis and plotting helpers

In addition to the training/evaluation CLI, this repository ships helper scripts for post-training inspection and report figures:

- `plot_performance.py`: plot model-level metrics such as PID and energy-regression performance from saved outputs/checkpoints.
- `plot_eval_across_n.py`: compare evaluation metrics across different event-count or sampling configurations (`n`).
- `eval_checkpoint_across_n.py`: run checkpoint evaluation repeatedly across multiple `n` settings and save aggregated results.
- `plot_shower_properties.py`: visualize hit/shower-level distributions (spatial, timing, and amplitude-derived views).
- `analyze_cutoff_readouts.py`: inspect readout truncation/cutoff behavior and its impact on retained signal content.
- `check_point_order.py`: quick sanity check utility for ordering/consistency of point-wise arrays in prepared files.

A typical workflow is:
1. Train with `run.py` and save artifacts under `save/<name>/`.
2. Run evaluation scripts to produce aggregate JSON/CSV summaries.
3. Use plotting scripts to generate figures for debugging, comparisons, and reports.

---

## Testing

Run the test suite:

```bash
pytest
```

Or run a subset while iterating:

```bash
pytest tests/test_data.py -q
```

---

## Notes

- Use `--pool > 1` to switch feature paths from `DRcalo3dHits.*` to pooled variants (`DRcalo3dHits{pool}.*`).
- Use `--enable_direction_regression` with `--direction_keys` to add angular target regression.
- Use `--compile` (PyTorch 2+) and `--use_amp` for performance-oriented runs.
