"""Plot run performance as a function of E_gen, theta, and phi.

This script compares multiple run directories (e.g. test2/test3/test4/test5)
using:
- pairwise ROC-AUC in bins of E_gen/theta/phi
- overall accuracy in bins of E_gen/theta/phi

Legends include differing config fields so runs with different settings are
visually identified.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from utils.plot_helpers import (
    PAIR_LABELS,
    PAIR_SPECS,
    config_label,
    differing_config_keys,
    infer_class_to_label,
    pair_key,
    resolve_run_dir,
    safe_name,
)


KIN_VARS = {
    "E_gen": "E_gen",
    "theta": "GenParticles.momentum.theta",
    "phi": "GenParticles.momentum.phi",
}
CONFIG_DIFF_EXCLUDE = {
    "cache_size",
    "max_cache_chunks",
    "eval_batch_size",
    "num_workers",
    "log_batch_lengths",
    "log_cuda_memory",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot performance vs E_gen/theta/phi across runs.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run names or directories.")
    parser.add_argument("--base-dir", default="save", help="Base dir when run names are provided.")
    parser.add_argument("--out-dir", default="plots", help="Output directory.")
    parser.add_argument("--bins", type=int, default=37, help="Number of bins.")
    parser.add_argument(
        "--binning",
        choices=["quantile", "uniform"],
        default="quantile",
        help="Binning strategy for x variables.",
    )
    parser.add_argument(
        "--barrel-theta-min",
        type=float,
        default=0.55,
        help="Barrel lower theta bound in radians.",
    )
    parser.add_argument(
        "--barrel-theta-max",
        type=float,
        default=np.pi/2+0.55,
        help="Barrel upper theta bound in radians.",
    )
    return parser.parse_args()


def _resolve_h5_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        Path(".") / raw,
        Path("h5s") / raw,
        Path("/store/ml/dual-readout/h5s") / raw,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


def _extract_event_ids(records: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    file_ids = np.full(len(records), -1, dtype=np.int64)
    event_ids = np.full(len(records), -1, dtype=np.int64)
    for i, rec in enumerate(records):
        eid = rec.get("event_id", [])
        if isinstance(eid, list) and len(eid) >= 2:
            file_ids[i] = int(eid[0])
            event_ids[i] = int(eid[1])
    return file_ids, event_ids


def _load_var_from_h5(
    records: Sequence[Mapping[str, object]],
    files: Sequence[str],
    dataset_key: str,
    cache: Dict[Tuple[str, Tuple[str, ...], bytes, bytes], np.ndarray] | None = None,
) -> np.ndarray:
    file_ids, event_ids = _extract_event_ids(records)
    cache_key = (
        dataset_key,
        tuple(str(f) for f in files),
        file_ids.tobytes(),
        event_ids.tobytes(),
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key].copy()

    n = len(records)
    out = np.full(n, np.nan, dtype=np.float64)

    idx_by_file: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        if file_ids[i] >= 0 and event_ids[i] >= 0:
            idx_by_file[int(file_ids[i])].append(i)

    for fidx, rec_indices in idx_by_file.items():
        if not (0 <= fidx < len(files)):
            continue
        h5_path = _resolve_h5_path(str(files[fidx]))
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h5:
            if dataset_key not in h5:
                continue
            ds = h5[dataset_key]
            ev = np.asarray([event_ids[i] for i in rec_indices], dtype=np.int64)
            # h5py requires monotonically increasing advanced indices.
            order = np.argsort(ev, kind="stable")
            ev_sorted = ev[order]
            vals_sorted = np.asarray(ds[ev_sorted], dtype=np.float64).reshape(-1)
            vals = np.empty_like(vals_sorted)
            vals[order] = vals_sorted
            out[np.asarray(rec_indices, dtype=np.int64)] = vals
    if cache is not None:
        cache[cache_key] = out.copy()
    return out


def make_bins(values: np.ndarray, bins: int, mode: str) -> np.ndarray:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([], dtype=np.float64)
    if mode == "uniform":
        lo, hi = float(np.min(v)), float(np.max(v))
        if lo == hi:
            return np.array([lo, hi + 1e-12], dtype=np.float64)
        return np.linspace(lo, hi, bins + 1)
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(v, q)
    edges = np.unique(edges)
    if edges.size < 2:
        x = float(v[0])
        return np.array([x, x + 1e-12], dtype=np.float64)
    return edges


def binned_accuracy(labels: np.ndarray, logits: np.ndarray, x: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers: List[float] = []
    vals: List[float] = []
    preds = np.argmax(logits, axis=1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (x >= lo) & (x < hi if i < len(edges) - 2 else x <= hi)
        m = m & np.isfinite(x)
        if np.count_nonzero(m) == 0:
            continue
        centers.append(0.5 * (lo + hi))
        vals.append(float(np.mean(preds[m] == labels[m])))
    return np.asarray(centers), np.asarray(vals)


def binned_pair_auc(
    labels: np.ndarray,
    probs: np.ndarray,
    x: np.ndarray,
    edges: np.ndarray,
    class_to_label: Mapping[str, int],
    pos: str,
    neg: str,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if pos not in class_to_label or neg not in class_to_label:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    pos_idx = int(class_to_label[pos])
    neg_idx = int(class_to_label[neg])
    score = probs[:, pos_idx]
    pair_mask = np.isin(labels, [pos_idx, neg_idx])
    centers: List[float] = []
    vals: List[float] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (x >= lo) & (x < hi if i < len(edges) - 2 else x <= hi)
        m = m & np.isfinite(x) & pair_mask
        if extra_mask is not None:
            m = m & extra_mask
        if np.count_nonzero(m) < 2:
            continue
        y = (labels[m] == pos_idx).astype(np.int32)
        if np.unique(y).size < 2:
            continue
        auc = roc_auc_score(y, score[m])
        centers.append(0.5 * (lo + hi))
        vals.append(float(auc))
    return np.asarray(centers), np.asarray(vals)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)

    run_names: List[str] = []
    cfgs: List[Dict[str, object]] = []
    run_payloads: List[Dict[str, object]] = []
    var_cache: Dict[Tuple[str, Tuple[str, ...], bytes, bytes], np.ndarray] = {}

    for raw in args.run_dirs:
        t0 = time.perf_counter()
        run_dir = resolve_run_dir(raw, base_dir)
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        output_path = run_dir / "output.json"
        if not (metrics_path.exists() and config_path.exists() and output_path.exists()):
            raise FileNotFoundError(f"{run_dir} must contain metrics.json/config.json/output.json")

        cfg = json.loads(config_path.read_text())
        records = json.loads(output_path.read_text())
        labels = np.asarray([int(r["label"]) for r in records], dtype=np.int64)
        logits = np.asarray([r["logits"] for r in records], dtype=np.float64)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        class_to_label = infer_class_to_label(records, cfg.get("train_files") or [])

        # Prefer test_files if explicitly provided; fallback to train_files mapping.
        source_files = cfg.get("test_files") or cfg.get("train_files") or []
        print(f"Loading {run_dir.name}: {len(records)} records")
        e_gen = _load_var_from_h5(records, source_files, KIN_VARS["E_gen"], var_cache)
        theta = _load_var_from_h5(records, source_files, KIN_VARS["theta"], var_cache)
        phi = _load_var_from_h5(records, source_files, KIN_VARS["phi"], var_cache)

        run_names.append(run_dir.name)
        cfgs.append(cfg)
        run_payloads.append(
            {
                "labels": labels,
                "logits": logits,
                "probs": probs,
                "class_to_label": class_to_label,
                "E_gen": e_gen,
                "theta": theta,
                "phi": phi,
            }
        )
        print(f"Loaded {run_dir.name} in {time.perf_counter() - t0:.1f}s")

    diff_keys = differing_config_keys(cfgs, CONFIG_DIFF_EXCLUDE)
    labels_for_legend = [
        config_label(name, cfg, diff_keys, width=44) for name, cfg in zip(run_names, cfgs)
    ]

    # Barrel/endcap split in theta (radians).
    theta_min = float(args.barrel_theta_min)
    theta_max = float(args.barrel_theta_max)
    if not (0.0 <= theta_min < theta_max <= math.pi):
        raise ValueError(
            f"Invalid barrel theta range: [{theta_min}, {theta_max}] must satisfy 0 <= min < max <= pi."
        )

    # Plot overall accuracy vs each kinematic variable.
    for kin_name in ("E_gen", "theta", "phi"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, payload in zip(labels_for_legend, run_payloads):
            x = payload[kin_name]
            edges = make_bins(x, args.bins, args.binning)
            if edges.size < 2:
                continue
            xc, yc = binned_accuracy(payload["labels"], payload["logits"], x, edges)
            if xc.size == 0:
                continue
            ax.plot(xc, yc, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel(kin_name)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs {kin_name}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, labelspacing=0.9)
        fig.tight_layout()
        out_path = out_dir / f"perf_vs_{kin_name}_accuracy.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")

    # Plot pairwise ROC-AUC vs each variable.
    for kin_name in ("E_gen", "theta", "phi"):
        for pos, neg in PAIR_SPECS:
            key = pair_key(pos, neg)
            fig, ax = plt.subplots(figsize=(8, 5))
            for label, payload in zip(labels_for_legend, run_payloads):
                x = payload[kin_name]
                edges = make_bins(x, args.bins, args.binning)
                if edges.size < 2:
                    continue
                xc, yc = binned_pair_auc(
                    payload["labels"],
                    payload["probs"],
                    x,
                    edges,
                    payload["class_to_label"],
                    pos,
                    neg,
                )
                if xc.size == 0:
                    continue
                ax.plot(xc, yc, marker="o", linewidth=1.5, label=label)
            ax.set_xlabel(kin_name)
            ax.set_ylabel("ROC-AUC")
            ax.set_title(f"{PAIR_LABELS.get(key, key)} AUC vs {kin_name}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=8, labelspacing=0.9)
            fig.tight_layout()
            safe_pair = safe_name(key)
            out_path = out_dir / f"perf_vs_{kin_name}_{safe_pair}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Wrote {out_path}")

    # Dedicated diagnostic: pi0 vs gamma AUC vs E_gen split by barrel/endcap.
    pair_pos, pair_neg = "pi0", "gamma"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    region_specs = [
        ("barrel", rf"Barrel: ${theta_min:.3f} \leq \theta \leq {theta_max:.3f}$ rad"),
        ("endcap", rf"Endcap: $\theta < {theta_min:.3f}$ or $\theta > {theta_max:.3f}$ rad"),
    ]
    for ax, (region_key, region_title) in zip(axes, region_specs):
        for label, payload in zip(labels_for_legend, run_payloads):
            x = payload["E_gen"]
            theta = payload["theta"]
            if x.size == 0 or theta.size == 0:
                continue
            if region_key == "barrel":
                region_mask = np.isfinite(theta) & (theta >= theta_min) & (theta <= theta_max)
            else:
                region_mask = np.isfinite(theta) & ((theta < theta_min) | (theta > theta_max))
            edges = make_bins(x[region_mask], args.bins, args.binning)
            if edges.size < 2:
                continue
            xc, yc = binned_pair_auc(
                payload["labels"],
                payload["probs"],
                x,
                edges,
                payload["class_to_label"],
                pair_pos,
                pair_neg,
                extra_mask=region_mask,
            )
            if xc.size == 0:
                continue
            ax.plot(xc, yc, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("E_gen")
        ax.set_title(region_title)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(fontsize=8, labelspacing=0.9)
    fig.suptitle(f"{PAIR_LABELS['pi0 vs gamma']} AUC vs E_gen by region")
    fig.tight_layout()
    out_region = out_dir / "perf_vs_E_gen_pi0_vs_gamma_barrel_endcap.png"
    fig.savefig(out_region, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_region}")

    # Save summary metadata.
    summary = {
        "runs": run_names,
        "config_diff_keys": diff_keys,
        "labels": labels_for_legend,
        "variables": list(KIN_VARS.keys()),
        "binning": args.binning,
        "bins": args.bins,
        "barrel_theta_min": theta_min,
        "barrel_theta_max": theta_max,
    }
    out_json = out_dir / "perf_by_kinematics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
