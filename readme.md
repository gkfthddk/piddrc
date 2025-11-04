# Dual-Readout Calorimeter Deep Learning Toolkit

This repository provides reusable building blocks to study particle
identification (PID) and energy reconstruction with dual-readout
calorimeter simulations.  The code is intentionally modular so that you
can experiment with different model families (MLP, masked point-set and
Mamba-inspired networks) on both simulated and test-beam data.

## Key Components

| Module | Description |
| --- | --- |
| `pid.data` | HDF5 dataset loader with on-the-fly feature engineering and collate function for variable-length showers. |
| `pid.models.mlp.SummaryMLP` | Simple baseline operating on engineered S/C summary statistics. |
| `pid.models.pointset_mlp.PointSetMLP` | Lightweight masked point-set MLP with multi-task (PID + energy) head. |
| `pid.models.pointset_transformer.PointSetTransformer` | Transformer encoder for masked point sets with global pooling head. |
| `pid.models.pointset_mamba.PointSetMamba` | Sequence model using gated residual mixing blocks inspired by Mamba. |
| `pid.engine.Trainer` | End-to-end training loop with mixed-precision support and research-grade metrics (accuracy, ROC-AUC, energy resolution/linearity). |

All models produce three outputs:

1. Classification logits for PID tasks.
2. Energy regression prediction.
3. Optional log-variance head to capture aleatoric uncertainty.

## Quick Start

### 1. Install dependencies

Install the core Python dependencies into your environment using the
provided requirements file:

```bash
pip install -r requirements.txt
```

### 2. Prepare input datasets

The toolkit expects HDF5 files where each entry represents a calorimeter
event.  You can use the helper script `toh5.py` to convert raw ROOT files
into the expected format:

```bash
python toh5.py --input path/to/tree.root --output electrons.h5
```

Repeat this conversion for every particle type that you would like to
include in the training sample.  When working with large data volumes it
is often convenient to shard the converted output into multiple files so
that they can be streamed efficiently from disk.

### 3. Create a dataset object

```python
from torch.utils.data import DataLoader
from pid.data import DualReadoutEventDataset, collate_events

dataset = DualReadoutEventDataset(
    files=["/path/to/electrons.h5", "/path/to/pions.h5"],
    hit_features=("x", "y", "z", "S", "C", "t"),
    label_key="particle_type",
    energy_key="true_energy",
    stat_file="/path/to/stats.yaml",
    max_points=2048,
)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_events, shuffle=True)
```

### 4. Instantiate a model and trainer

```python
import torch
from pid.engine import Trainer, TrainingConfig
from pid.models.pointset_mlp import PointSetMLP

sample = dataset[0]
model = PointSetMLP(
    in_channels=len(dataset.hit_features),
    summary_dim=sample.summary.numel(),
    num_classes=len(dataset.classes),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
trainer = Trainer(model, optimizer, device=torch.device("cuda"))
trainer.fit(loader)
```

### 5. Train and evaluate from the command line

For longer training runs you can use the high level CLI.  The entry point
`run.py` mirrors the steps above while adding logging, checkpointing and
TensorBoard integration:

```bash
python run.py \
  --train electron.h5 pion.h5 \
  --val val_electron.h5 val_pion.h5 \
  --model pointset_mlp \
  --epochs 100 \
  --batch_size 64 \
  --lr 3e-4
```

Checkpoints are written to the directory specified via `--output_dir` and
can later be restored with the `--resume` flag.  After training finishes
you can run the evaluation loop only by appending `--eval_only`.

### 6. Evaluate in Python

Use `trainer.evaluate(test_loader)` to obtain a summary of PID and energy
metrics in interactive notebooks or scripts.

### 7. Reproduce paper results

To replicate the configurations used in the accompanying paper or note,
inspect the experiment YAML files under `pid/config/experiments/`.  Each
file specifies the model family, optimizer schedule and data sources used
in our benchmarks.  The provided `submit.py` script reads these configs
and orchestrates multi-job sweeps across different random seeds.

### Inspecting model architectures

The training script can print a [`torchinfo`](https://github.com/tyleryep/torchinfo)
overview of the selected architecture using a representative batch drawn
from the training data. Enable it via the `--print_model_summary` flag:

```bash
python run.py --print_model_summary --eval_only --checkpoint path/to/checkpoint.pt
```

This is a convenient way to verify tensor shapes and parameter counts
before launching long training runs.

## Testing

A lightweight unit test suite is available to validate the full training
loop on synthetic data:

```bash
pytest
```

This exercises the data loader, models and trainer to guard against
regressions.

## Project structure

```
.
├── compute_channel_stats.py   # Utility for aggregating per-channel hit statistics
├── pid/                       # Core library code: data, models and engine modules
├── run.py                     # Training/evaluation CLI entry point
├── submit.py                  # Experiment launcher for sweeps across configs and seeds
├── tests/                     # Pytest-based regression suite
└── toh5.py                    # ROOT → HDF5 converter used during data preparation
```

## License

See `LICENSE` for licensing information.
