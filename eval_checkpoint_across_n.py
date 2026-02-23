"""Evaluate one checkpoint across multiple max_points values.

Example:
  python eval_checkpoint_across_n.py \
    --run-dir save/mamba_r1_t10_n1000 \
    --n-values 999 1000 1001 1002 2000 3000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


LIST_KEYS = [
    "train_files",
    "val_files",
    "test_files",
    "hit_features",
    "pos_keys",
    "head_hidden",
]

SCALAR_KEYS = [
    "model",
    "hidden_dim",
    "depth",
    "num_heads",
    "mlp_expansion",
    "k_neighbors",
    "dropout",
    "pool",
    "label_key",
    "energy_key",
    "batch_size",
    "num_workers",
    "stat_file",
    "device",
]

TRUE_FLAGS = [
    "dynamic_knn",
    "disable_summary",
    "disable_uncertainty",
    "use_amp",
    "compile",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed checkpoint with different max_points.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing config.json/checkpoint.pt.")
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        required=True,
        help="max_points values to evaluate.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="eval_same_ckpt_diff_n",
        help="Subdirectory under run-dir for evaluation outputs.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to call run.py.",
    )
    parser.add_argument(
        "--run-script",
        type=Path,
        default=Path("run.py"),
        help="Path to training/eval script.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Optional CUDA device list forwarded to run.py (e.g. '0' or '0,1').",
    )
    return parser.parse_args()


def append_cfg_args(cmd: List[str], cfg: Dict[str, Any]) -> None:
    for key in LIST_KEYS:
        value = cfg.get(key)
        if value:
            cmd.extend([f"--{key}", *[str(v) for v in value]])

    for key in SCALAR_KEYS:
        value = cfg.get(key)
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    for key in TRUE_FLAGS:
        if bool(cfg.get(key, False)):
            cmd.append(f"--{key}")

    # Flags that are store_false in run.py defaults.
    if cfg.get("dataset_progress") is False:
        cmd.append("--no_dataset_progress")
    if cfg.get("progress_bar") is False:
        cmd.append("--no_progress_bar")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoint.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = json.loads(cfg_path.read_text())

    out_root = run_dir / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    n_values = sorted(set(int(n) for n in args.n_values))
    for n in n_values:
        out_dir = out_root / f"n{n}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [args.python_bin, str(args.run_script)]
        if args.gpu is not None:
            cmd.extend(["--gpu", args.gpu])
        cmd.extend(["--eval_only", "--checkpoint", str(ckpt_path)])
        append_cfg_args(cmd, cfg)
        cmd.extend(
            [
                "--max_points",
                str(n),
                "--name",
                f"{run_dir.name}_{args.output_subdir}_n{n}",
                "--metrics_json",
                str(out_dir / "metrics.json"),
                "--output_json",
                str(out_dir / "output.json"),
                "--config_json",
                str(out_dir / "config.json"),
            ]
        )

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Done. Outputs written under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
