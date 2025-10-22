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

1. **Install dependencies** using the provided requirements file:

   ```bash
   pip install -r requirements.txt
   ```
2. **Create a dataset** by pointing the loader to your HDF5 files:

   ```python
   from torch.utils.data import DataLoader
   from pid.data import DualReadoutEventDataset, collate_events

   dataset = DualReadoutEventDataset(
       files=["/path/to/electrons.h5", "/path/to/pions.h5"],
       hit_features=("x", "y", "z", "S", "C", "t"),
       label_key="particle_type",
       energy_key="true_energy",
       max_points=2048,
   )

   loader = DataLoader(dataset, batch_size=32, collate_fn=collate_events, shuffle=True)
   ```

3. **Instantiate a model** and trainer:

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

4. **Evaluate** using `trainer.evaluate(test_loader)` to obtain a summary
   of PID and energy metrics.

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

## License

See `LICENSE` for licensing information.
