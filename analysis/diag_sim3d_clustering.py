#!/usr/bin/env python
"""Grid-search optimization for Sim3dCalorimeterHits clustering hyperparameters.

Evaluates BIRCH (threshold, branching_factor, n_clusters) on digi ROOT files using
both efficiency and information-loss metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot
from tqdm.auto import tqdm

try:
    from sklearn.cluster import Birch
except Exception as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "scikit-learn is required for clustering optimization. Install scikit-learn."
    ) from exc


SIM3D_E = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.energy"
SIM3D_X = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.position.x"
SIM3D_Y = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.position.y"
SIM3D_Z = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.position.z"


@dataclass
class EventData:
    xyz: np.ndarray  # (N,3)
    energy: np.ndarray  # (N,)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize Sim3dCalorimeterHit clustering hyperparameters.")
    p.add_argument("--input-dir", type=Path, default=Path("/hdfs/ml/dualreadout/v001/pi+_1-120GeV"))
    p.add_argument("--pattern", type=str, default="digi_*.root")
    p.add_argument("--tree", type=str, default="events")
    p.add_argument("--max-files", type=int, default=20)
    p.add_argument("--max-events", type=int, default=50)
    p.add_argument(
        "--max-hits-per-event",
        type=int,
        default=0,
        help="Per-event hit cap before clustering (<=0 means no cap).",
    )
    p.add_argument(
        "--hit-downsample-mode",
        type=str,
        choices=["top_energy", "random", "stratified"],
        default="stratified",
        help="How to downsample hits when --max-hits-per-event is exceeded.",
    )
    p.add_argument("--stratify-energy-bins", type=int, default=4, help="Energy-quantile bins for stratified downsampling.")
    p.add_argument("--stratify-radius-bins", type=int, default=6, help="Radius-quantile bins for stratified downsampling.")
    p.add_argument(
        "--stratify-min-per-bin",
        type=int,
        default=4,
        help="Minimum requested samples per non-empty stratified bin (soft target).",
    )
    p.add_argument(
        "--events-per-config",
        type=int,
        default=0,
        help="Evaluate each clustering config on only this many events (<=0 means all collected events).",
    )

    p.add_argument("--threshold-grid", type=str, default="5,10")
    p.add_argument("--branching-factor-grid", type=str, default="64,128,256")
    p.add_argument("--n-clusters-grid", type=str, default="none,1000,3000,5000")

    p.add_argument("--radial-bins", type=int, default=80)
    p.add_argument("--r-min", type=float, default=1800.0)
    p.add_argument("--r-max", type=float, default=3800.0)

    p.add_argument("--alpha-time", type=float, default=0.35, help="Weight for runtime term in weighted score.")
    p.add_argument("--alpha-compression", type=float, default=0.35, help="Weight for compression term in weighted score.")
    p.add_argument("--alpha-loss", type=float, default=0.30, help="Weight for information loss term in weighted score.")

    p.add_argument("--out-json", type=Path, default=Path("analysis/sim3d_clustering_optimization/results.json"))
    p.add_argument("--out-plot", type=Path, default=Path("analysis/sim3d_clustering_optimization/pareto.png"))
    p.add_argument(
        "--out-hitcount-plot",
        type=Path,
        default=Path("analysis/sim3d_clustering_optimization/hit_count_distribution.png"),
    )
    return p.parse_args()


def _parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_grid(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_n_clusters_grid(text: str) -> list[int | None]:
    out: list[int | None] = []
    for token in text.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(t))
    return out


def _stratified_indices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    e: np.ndarray,
    n_take: int,
    rng: np.random.Generator,
    n_e_bins: int,
    n_r_bins: int,
    min_per_bin: int,
) -> np.ndarray:
    if n_take >= e.size:
        return np.arange(e.size, dtype=np.int64)

    r = np.sqrt(x * x + y * y + z * z)
    # Quantile bins make sparse low-amplitude regions retain representation.
    e_q = np.quantile(e, np.linspace(0.0, 1.0, max(2, n_e_bins + 1)))
    r_q = np.quantile(r, np.linspace(0.0, 1.0, max(2, n_r_bins + 1)))
    e_q = np.unique(e_q)
    r_q = np.unique(r_q)
    if e_q.size < 2 or r_q.size < 2:
        return rng.choice(e.size, size=n_take, replace=False)

    e_bin = np.clip(np.digitize(e, e_q[1:-1], right=True), 0, e_q.size - 2)
    r_bin = np.clip(np.digitize(r, r_q[1:-1], right=True), 0, r_q.size - 2)
    joint = e_bin.astype(np.int64) * (r_q.size - 1) + r_bin.astype(np.int64)

    uniq, counts = np.unique(joint, return_counts=True)
    # Proportional allocation with soft minimum per populated bin.
    alloc = np.floor(n_take * (counts / np.sum(counts))).astype(int)
    if min_per_bin > 0:
        alloc = np.maximum(alloc, np.minimum(counts, min_per_bin))

    # Normalize to exact n_take.
    total = int(np.sum(alloc))
    if total > n_take:
        order = np.argsort(alloc)[::-1]
        k = 0
        while total > n_take and k < order.size * 4:
            j = order[k % order.size]
            if alloc[j] > 1:
                alloc[j] -= 1
                total -= 1
            k += 1
    elif total < n_take:
        short = n_take - total
        frac = n_take * (counts / np.sum(counts)) - np.floor(n_take * (counts / np.sum(counts)))
        order = np.argsort(frac)[::-1]
        k = 0
        while short > 0 and k < order.size * 6:
            j = order[k % order.size]
            if alloc[j] < counts[j]:
                alloc[j] += 1
                short -= 1
            k += 1

    picks: list[np.ndarray] = []
    for key, n_sel in zip(uniq, alloc):
        if n_sel <= 0:
            continue
        idx = np.where(joint == key)[0]
        if idx.size <= n_sel:
            picks.append(idx)
        else:
            picks.append(rng.choice(idx, size=n_sel, replace=False))

    if not picks:
        return rng.choice(e.size, size=n_take, replace=False)
    out = np.concatenate(picks)
    if out.size > n_take:
        out = rng.choice(out, size=n_take, replace=False)
    elif out.size < n_take:
        remain = np.setdiff1d(np.arange(e.size, dtype=np.int64), out, assume_unique=False)
        if remain.size > 0:
            extra = rng.choice(remain, size=min(n_take - out.size, remain.size), replace=False)
            out = np.concatenate([out, extra])
    return out


def _collect_events(args: argparse.Namespace) -> list[EventData]:
    files = sorted(args.input_dir.glob(args.pattern))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files found: {args.input_dir / args.pattern}")

    events: list[EventData] = []
    remaining = args.max_events if args.max_events > 0 else math.inf

    rng = np.random.default_rng(12345)
    for path in tqdm(files, desc="Loading files", unit="file"):
        if remaining <= 0:
            break
        with uproot.open(path) as f:
            if args.tree not in f:
                continue
            t = f[args.tree]
            arr = t.arrays([SIM3D_E, SIM3D_X, SIM3D_Y, SIM3D_Z], library="ak")

        n_evt = len(arr[SIM3D_E])
        take = int(min(n_evt, remaining)) if remaining != math.inf else n_evt

        for i in range(take):
            e = np.asarray(arr[SIM3D_E][i], dtype=np.float64)
            x = np.asarray(arr[SIM3D_X][i], dtype=np.float64)
            y = np.asarray(arr[SIM3D_Y][i], dtype=np.float64)
            z = np.asarray(arr[SIM3D_Z][i], dtype=np.float64)

            m = np.isfinite(e) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (e > 0)
            if not np.any(m):
                continue
            e = e[m]
            x = x[m]
            y = y[m]
            z = z[m]

            if args.max_hits_per_event > 0 and e.size > args.max_hits_per_event:
                if args.hit_downsample_mode == "random":
                    idx = rng.choice(e.size, size=args.max_hits_per_event, replace=False)
                elif args.hit_downsample_mode == "stratified":
                    idx = _stratified_indices(
                        x=x,
                        y=y,
                        z=z,
                        e=e,
                        n_take=args.max_hits_per_event,
                        rng=rng,
                        n_e_bins=args.stratify_energy_bins,
                        n_r_bins=args.stratify_radius_bins,
                        min_per_bin=args.stratify_min_per_bin,
                    )
                else:
                    idx = np.argpartition(e, -args.max_hits_per_event)[-args.max_hits_per_event :]
                e = e[idx]
                x = x[idx]
                y = y[idx]
                z = z[idx]

            xyz = np.column_stack([x, y, z])
            events.append(EventData(xyz=xyz, energy=e))
            if remaining != math.inf:
                remaining -= 1
                if remaining <= 0:
                    break

    if not events:
        raise RuntimeError("No valid Sim3d events were collected.")
    return events


def _weighted_centroid(xyz: np.ndarray, w: np.ndarray) -> np.ndarray:
    ws = np.sum(w)
    if ws <= 0:
        return np.full(3, np.nan, dtype=np.float64)
    return np.sum(xyz * w[:, None], axis=0) / ws


def _radial_hist(xyz: np.ndarray, e: np.ndarray, r_edges: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(xyz * xyz, axis=1))
    return np.histogram(r, bins=r_edges, weights=e)[0].astype(np.float64)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    ps = p / max(float(np.sum(p)), eps)
    qs = q / max(float(np.sum(q)), eps)
    m = 0.5 * (ps + qs)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * _kl(ps, m) + 0.5 * _kl(qs, m)


def _cluster_event(
    event: EventData,
    threshold: float,
    branching_factor: int,
    n_clusters: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
    )
    t0 = time.perf_counter()
    labels = model.fit_predict(event.xyz)
    dt_ms = (time.perf_counter() - t0) * 1e3

    uniq = np.unique(labels)
    c_xyz = np.empty((uniq.size, 3), dtype=np.float64)
    c_e = np.empty(uniq.size, dtype=np.float64)

    for j, lab in enumerate(uniq):
        m = labels == lab
        w = event.energy[m]
        pts = event.xyz[m]
        c_e[j] = float(np.sum(w))
        c_xyz[j] = _weighted_centroid(pts, w)

    return c_xyz, c_e, dt_ms


def _evaluate_one(
    events: list[EventData],
    threshold: float,
    branching_factor: int,
    n_clusters: int | None,
    r_edges: np.ndarray,
    events_per_config: int = 0,
) -> dict[str, Any]:
    times: list[float] = []
    comp_ratios: list[float] = []
    centroid_errs: list[float] = []
    width_rel_errs: list[float] = []
    js_losses: list[float] = []
    cluster_counts: list[int] = []

    if events_per_config > 0 and len(events) > events_per_config:
        idx = np.linspace(0, len(events) - 1, events_per_config, dtype=int)
        events_eval = [events[i] for i in idx]
    else:
        events_eval = events

    cfg_label = f"ev thr={threshold:g} bf={branching_factor} nc={n_clusters}"
    for ev in tqdm(events_eval, desc=cfg_label, unit="evt", leave=False):
        c_xyz, c_e, dt_ms = _cluster_event(ev, threshold, branching_factor, n_clusters)
        times.append(dt_ms)
        cluster_counts.append(int(c_e.size))
        comp_ratios.append(float(c_e.size / max(ev.energy.size, 1)))

        c0 = _weighted_centroid(ev.xyz, ev.energy)
        c1 = _weighted_centroid(c_xyz, c_e)
        centroid_errs.append(float(np.linalg.norm(c1 - c0)))

        r0 = np.sqrt(np.sum((ev.xyz - c0[None, :]) ** 2, axis=1))
        r1 = np.sqrt(np.sum((c_xyz - c1[None, :]) ** 2, axis=1))
        w0 = float(np.sqrt(np.sum(ev.energy * r0 * r0) / max(np.sum(ev.energy), 1e-15)))
        w1 = float(np.sqrt(np.sum(c_e * r1 * r1) / max(np.sum(c_e), 1e-15)))
        width_rel_errs.append(float(abs(w1 - w0) / max(w0, 1e-12)))

        h0 = _radial_hist(ev.xyz, ev.energy, r_edges)
        h1 = _radial_hist(c_xyz, c_e, r_edges)
        js_losses.append(_js_divergence(h0, h1))

    cc = np.asarray(cluster_counts, dtype=np.int64)
    return {
        "threshold": float(threshold),
        "branching_factor": int(branching_factor),
        "n_clusters": (None if n_clusters is None else int(n_clusters)),
        "mean_cluster_time_ms": float(np.mean(times)),
        "mean_cluster_count": float(np.mean(cc)),
        "min_cluster_count": int(np.min(cc)),
        "max_cluster_count": int(np.max(cc)),
        "p50_cluster_count": float(np.quantile(cc, 0.50)),
        "p90_cluster_count": float(np.quantile(cc, 0.90)),
        "p99_cluster_count": float(np.quantile(cc, 0.99)),
        "mean_cluster_ratio": float(np.mean(comp_ratios)),
        "mean_compression": float(1.0 - np.mean(comp_ratios)),
        "mean_centroid_error": float(np.mean(centroid_errs)),
        "mean_width_rel_error": float(np.mean(width_rel_errs)),
        "mean_js_loss": float(np.mean(js_losses)),
        "composite_loss": float(np.mean(js_losses) + np.mean(width_rel_errs)),
        "events_evaluated": int(len(events_eval)),
    }


def _minmax_norm(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    n = costs.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        efficient[dominated] = False
    return efficient


def _make_plot(results: list[dict[str, Any]], out_plot: Path) -> None:
    import matplotlib.pyplot as plt

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    comp = np.array([r["mean_compression"] for r in results], dtype=np.float64)
    loss = np.array([r["composite_loss"] for r in results], dtype=np.float64)
    tms = np.array([r["mean_cluster_time_ms"] for r in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(comp, loss, c=tms, cmap="viridis", s=45, alpha=0.9)
    ax.set_xlabel("mean compression (1 - cluster_ratio)")
    ax.set_ylabel("composite loss (JS + width_rel_error)")
    ax.set_title("Sim3d clustering optimization tradeoff")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="mean cluster time [ms/event]")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=170)
    plt.close(fig)


def _summarize_preclustering(events: list[EventData], out_plot: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    hit_counts = np.asarray([ev.energy.size for ev in events], dtype=np.int64)
    if hit_counts.size == 0:
        return {
            "num_events": 0,
            "hit_count": {
                "min": None,
                "max": None,
                "mean": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "p99": float("nan"),
            },
            "histogram_plot": None,
        }

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hit_counts, bins=80, histtype="step", linewidth=1.6)
    ax.set_title("Pre-clustering Sim3d hit-count distribution")
    ax.set_xlabel("number of hits per event")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=170)
    plt.close(fig)

    return {
        "num_events": int(hit_counts.size),
        "hit_count": {
            "min": int(np.min(hit_counts)),
            "max": int(np.max(hit_counts)),
            "mean": float(np.mean(hit_counts)),
            "p50": float(np.quantile(hit_counts, 0.50)),
            "p90": float(np.quantile(hit_counts, 0.90)),
            "p99": float(np.quantile(hit_counts, 0.99)),
        },
        "histogram_plot": str(out_plot),
    }


def main() -> int:
    args = parse_args()

    thr_grid = _parse_float_grid(args.threshold_grid)
    bf_grid = _parse_int_grid(args.branching_factor_grid)
    nc_grid = _parse_n_clusters_grid(args.n_clusters_grid)

    events = _collect_events(args)
    pre_summary = _summarize_preclustering(events, args.out_hitcount_plot)
    r_edges = np.linspace(args.r_min, args.r_max, args.radial_bins + 1)

    results: list[dict[str, Any]] = []
    grid = [(t, b, n) for t in thr_grid for b in bf_grid for n in nc_grid]

    for t, b, n in tqdm(grid, desc="Evaluating grid", unit="cfg"):
        try:
            res = _evaluate_one(events, t, b, n, r_edges, events_per_config=args.events_per_config)
            results.append(res)
        except Exception as exc:
            results.append(
                {
                    "threshold": float(t),
                    "branching_factor": int(b),
                    "n_clusters": (None if n is None else int(n)),
                    "error": str(exc),
                }
            )

    valid = [r for r in results if "error" not in r]
    if not valid:
        raise RuntimeError("All grid evaluations failed.")

    tms = np.array([r["mean_cluster_time_ms"] for r in valid], dtype=np.float64)
    ratio = np.array([r["mean_cluster_ratio"] for r in valid], dtype=np.float64)
    loss = np.array([r["composite_loss"] for r in valid], dtype=np.float64)

    nt = _minmax_norm(tms)
    nr = _minmax_norm(ratio)  # lower ratio is better
    nl = _minmax_norm(loss)

    for i, r in enumerate(valid):
        weighted = args.alpha_time * nt[i] + args.alpha_compression * nr[i] + args.alpha_loss * nl[i]
        r["weighted_score"] = float(weighted)

    costs = np.column_stack([tms, ratio, loss])
    pareto_mask = _is_pareto_efficient(costs)
    pareto = [valid[i] for i, ok in enumerate(pareto_mask) if ok]

    valid_sorted = sorted(valid, key=lambda x: x["weighted_score"])

    payload = {
        "config": {
            "input_dir": str(args.input_dir),
            "pattern": args.pattern,
            "tree": args.tree,
            "num_events": len(events),
            "threshold_grid": thr_grid,
            "branching_factor_grid": bf_grid,
            "n_clusters_grid": [None if n is None else int(n) for n in nc_grid],
            "max_hits_per_event": int(args.max_hits_per_event),
            "hit_downsample_mode": args.hit_downsample_mode,
            "stratify_energy_bins": int(args.stratify_energy_bins),
            "stratify_radius_bins": int(args.stratify_radius_bins),
            "stratify_min_per_bin": int(args.stratify_min_per_bin),
            "events_per_config": int(args.events_per_config),
            "alpha_time": float(args.alpha_time),
            "alpha_compression": float(args.alpha_compression),
            "alpha_loss": float(args.alpha_loss),
        },
        "preclustering_summary": pre_summary,
        "best_by_weighted_score": valid_sorted[:10],
        "pareto_front": pareto,
        "all_results": results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _make_plot(valid, args.out_plot)

    print(json.dumps(
        {
            "num_events": len(events),
            "hit_count_min": pre_summary["hit_count"]["min"],
            "hit_count_max": pre_summary["hit_count"]["max"],
            "num_configs": len(grid),
            "num_valid_configs": len(valid),
            "best": valid_sorted[0],
            "out_json": str(args.out_json),
            "out_plot": str(args.out_plot),
            "out_hitcount_plot": str(args.out_hitcount_plot),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
