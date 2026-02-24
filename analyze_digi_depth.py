#!/usr/bin/env python
"""Analyze Sim3dCalorimeterHits shower-shape observables for kaon+/pi+ discrimination."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from tqdm.auto import tqdm


SIM3D_ENERGY_BRANCH = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.energy"
SIM3D_X_BRANCH = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.position.x"
SIM3D_Y_BRANCH = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.position.y"
SIM3D_Z_BRANCH = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.position.z"
MOM_X_BRANCH = "GenParticles/GenParticles.momentum.x"
MOM_Y_BRANCH = "GenParticles/GenParticles.momentum.y"
MOM_Z_BRANCH = "GenParticles/GenParticles.momentum.z"


@dataclass
class SampleResult:
    summary: dict[str, Any]
    features: dict[str, np.ndarray]
    r_hist_energy: np.ndarray
    rperp_hist_energy: np.ndarray
    xyz: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Sim3dCalorimeterHits longitudinal/lateral shower distributions."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/hdfs/ml/dualreadout/v001/kaon+_10GeV"),
        help="Directory containing digi_*.root files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="digi_kaon+_*.root",
        help="Glob pattern for ROOT files under --input-dir.",
    )
    parser.add_argument("--label", type=str, default="kaon+", help="Display label for primary sample.")
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=Path("/hdfs/ml/dualreadout/v001/pi+_10GeV"),
        help="Second sample directory for comparison.",
    )
    parser.add_argument("--compare-pattern", type=str, default="digi_pi+_*.root", help="Glob pattern for --compare-dir.")
    parser.add_argument("--compare-label", type=str, default="pi+", help="Display label for comparison sample.")
    parser.add_argument("--no-compare", action="store_true", help="Analyze only primary sample.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files per sample (<=0 means all).")
    parser.add_argument("--max-events", type=int, default=2000, help="Limit events per sample (<=0 means all).")
    parser.add_argument("--tree", type=str, default="events", help="TTree name.")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/kaon_digi_depth"), help="Output directory.")

    parser.add_argument("--r-min", type=float, default=1800.0, help="Minimum r bin edge.")
    parser.add_argument("--r-max", type=float, default=3800.0, help="Maximum r bin edge.")
    parser.add_argument("--r-bins", type=int, default=120, help="Number of r bins.")
    parser.add_argument("--r-leak-cut", type=float, default=3200.0, help="Leakage threshold in r.")
    parser.add_argument("--r-core-cut", type=float, default=2300.0, help="Longitudinal core threshold in r.")

    parser.add_argument("--rperp-max", type=float, default=500.0, help="Maximum lateral-distance bin edge.")
    parser.add_argument("--rperp-bins", type=int, default=100, help="Number of lateral-distance bins.")
    parser.add_argument("--rperp-core-cut", type=float, default=80.0, help="Lateral core threshold in r_perp.")

    parser.add_argument(
        "--momentum-bins",
        type=str,
        default="1,3,5,10,20,40,60,80,100",
        help="Comma-separated momentum bin edges for discrimination metrics.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic downsampling.")
    parser.add_argument(
        "--shape-scatter-points",
        type=int,
        default=20000,
        help="Maximum event points in depth-vs-width scatter plots.",
    )
    parser.add_argument(
        "--xyz-scatter",
        action="store_true",
        help="Enable optional x-y-z Sim3dHit scatter plot.",
    )
    parser.add_argument(
        "--scatter-points",
        type=int,
        default=15000,
        help="Maximum points in optional x-y-z scatter plot.",
    )
    parser.add_argument(
        "--axis-quality-cut",
        type=float,
        default=1.5,
        help="Minimum lambda1/lambda2 ratio to consider PCA axis well-defined.",
    )
    return parser.parse_args()


def _safe_stats(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "min": None,
            "max": None,
        }
    is_int = np.issubdtype(finite.dtype, np.integer)
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "min": int(np.min(finite)) if is_int else float(np.min(finite)),
        "max": int(np.max(finite)) if is_int else float(np.max(finite)),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return float("nan")
    q = float(np.clip(quantile, 0.0, 1.0))
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    wsum = np.sum(w)
    if wsum <= 0:
        return float("nan")
    cdf = np.cumsum(w) / wsum
    return float(v[np.searchsorted(cdf, q, side="left")])


def _first_momentum_norm(arr: ak.Array) -> np.ndarray:
    px0 = ak.fill_none(ak.firsts(arr[MOM_X_BRANCH]), np.nan)
    py0 = ak.fill_none(ak.firsts(arr[MOM_Y_BRANCH]), np.nan)
    pz0 = ak.fill_none(ak.firsts(arr[MOM_Z_BRANCH]), np.nan)
    px = np.asarray(px0, dtype=np.float64)
    py = np.asarray(py0, dtype=np.float64)
    pz = np.asarray(pz0, dtype=np.float64)
    return np.sqrt(px * px + py * py + pz * pz)


def _event_pca_axis_and_distances(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> tuple[float, float, np.ndarray]:
    pos = np.column_stack([x, y, z])
    wsum = np.sum(w)
    if pos.shape[0] < 2 or wsum <= 0:
        return float("nan"), float("nan"), np.full(pos.shape[0], np.nan, dtype=np.float64)

    centroid = np.average(pos, axis=0, weights=w)
    centered = pos - centroid
    cov = (centered * w[:, None]).T @ centered / wsum
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axis = eigvecs[:, order[0]]
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        return float("nan"), float("nan"), np.full(pos.shape[0], np.nan, dtype=np.float64)

    axis = axis / axis_norm
    proj = centered @ axis
    parallel = np.outer(proj, axis)
    perp_vec = centered - parallel
    r_perp = np.linalg.norm(perp_vec, axis=1)

    l1 = float(eigvals[0]) if eigvals.size > 0 else float("nan")
    l2 = float(eigvals[1]) if eigvals.size > 1 else float("nan")
    ratio = float(l1 / l2) if np.isfinite(l1) and np.isfinite(l2) and l2 > 0 else float("nan")
    return l1, ratio, r_perp


def _collect_sample_events(
    *,
    sample_label: str,
    input_dir: Path,
    pattern: str,
    max_files: int,
    max_events: int,
    tree_name: str,
    r_edges: np.ndarray,
    rperp_edges: np.ndarray,
    r_leak_cut: float,
    r_core_cut: float,
    rperp_core_cut: float,
    axis_quality_cut: float,
    xyz_scatter: bool,
    seed: int,
) -> SampleResult:
    files = sorted(input_dir.glob(pattern))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files found for {sample_label}: {input_dir / pattern}")

    r_hist_energy = np.zeros(len(r_edges) - 1, dtype=np.float64)
    rperp_hist_energy = np.zeros(len(rperp_edges) - 1, dtype=np.float64)

    features: dict[str, list[float]] = {
        "sim3dhit_count": [],
        "event_energy": [],
        "gen_momentum_norm": [],
        "depth_centroid_r": [],
        "width_r": [],
        "r50": [],
        "r80": [],
        "r90": [],
        "leakage_fraction_r": [],
        "core_fraction_r": [],
        "lateral_width": [],
        "rperp50": [],
        "rperp80": [],
        "rperp90": [],
        "core_fraction_rperp": [],
        "pca_lambda1": [],
        "pca_axis_ratio": [],
    }

    xyz_samples: list[np.ndarray] = []
    total_events_processed = 0
    files_processed = 0

    events_target = max_events if max_events > 0 else None
    file_iter = tqdm(files, desc=f"Files [{sample_label}]", unit="file")
    event_pbar = tqdm(total=events_target, desc=f"Events [{sample_label}]", unit="evt") if events_target else None

    rng = np.random.default_rng(seed)

    for file_path in file_iter:
        if max_events > 0 and total_events_processed >= max_events:
            break

        with uproot.open(file_path) as f:
            if tree_name not in f:
                continue
            tree = f[tree_name]
            arr = tree.arrays(
                [
                    SIM3D_ENERGY_BRANCH,
                    SIM3D_X_BRANCH,
                    SIM3D_Y_BRANCH,
                    SIM3D_Z_BRANCH,
                    MOM_X_BRANCH,
                    MOM_Y_BRANCH,
                    MOM_Z_BRANCH,
                ],
                library="ak",
            )

        files_processed += 1

        n_file_events = len(arr[SIM3D_ENERGY_BRANCH])
        if max_events > 0:
            remaining = max_events - total_events_processed
            if remaining <= 0:
                break
            if n_file_events > remaining:
                arr = arr[:remaining]
                n_file_events = remaining

        total_events_processed += n_file_events
        if event_pbar is not None:
            event_pbar.update(n_file_events)
        file_iter.set_postfix(events=total_events_processed)

        mom_norm = _first_momentum_norm(arr)

        sim_e = arr[SIM3D_ENERGY_BRANCH]
        sim_x = arr[SIM3D_X_BRANCH]
        sim_y = arr[SIM3D_Y_BRANCH]
        sim_z = arr[SIM3D_Z_BRANCH]

        for iev in range(n_file_events):
            e = np.asarray(sim_e[iev], dtype=np.float64)
            x = np.asarray(sim_x[iev], dtype=np.float64)
            y = np.asarray(sim_y[iev], dtype=np.float64)
            z = np.asarray(sim_z[iev], dtype=np.float64)

            valid = np.isfinite(e) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (e > 0)
            e = e[valid]
            x = x[valid]
            y = y[valid]
            z = z[valid]

            features["sim3dhit_count"].append(float(e.size))
            features["gen_momentum_norm"].append(float(mom_norm[iev]) if iev < mom_norm.size else float("nan"))

            if e.size == 0:
                features["event_energy"].append(0.0)
                features["depth_centroid_r"].append(float("nan"))
                features["width_r"].append(float("nan"))
                features["r50"].append(float("nan"))
                features["r80"].append(float("nan"))
                features["r90"].append(float("nan"))
                features["leakage_fraction_r"].append(float("nan"))
                features["core_fraction_r"].append(float("nan"))
                features["lateral_width"].append(float("nan"))
                features["rperp50"].append(float("nan"))
                features["rperp80"].append(float("nan"))
                features["rperp90"].append(float("nan"))
                features["core_fraction_rperp"].append(float("nan"))
                features["pca_lambda1"].append(float("nan"))
                features["pca_axis_ratio"].append(float("nan"))
                continue

            r = np.sqrt(x * x + y * y + z * z)
            etot = float(np.sum(e))
            features["event_energy"].append(etot)

            r_hist_energy += np.histogram(r, bins=r_edges, weights=e)[0]

            r_cent = float(np.sum(e * r) / etot)
            r_var = float(np.sum(e * (r - r_cent) ** 2) / etot)
            r_width = float(np.sqrt(max(r_var, 0.0)))
            features["depth_centroid_r"].append(r_cent)
            features["width_r"].append(r_width)

            features["r50"].append(_weighted_quantile(r, e, 0.50))
            features["r80"].append(_weighted_quantile(r, e, 0.80))
            features["r90"].append(_weighted_quantile(r, e, 0.90))
            features["leakage_fraction_r"].append(float(np.sum(e[r > r_leak_cut]) / etot))
            features["core_fraction_r"].append(float(np.sum(e[r < r_core_cut]) / etot))

            l1, ratio, r_perp = _event_pca_axis_and_distances(x, y, z, e)
            features["pca_lambda1"].append(l1)
            features["pca_axis_ratio"].append(ratio)

            if np.any(np.isfinite(r_perp)):
                rperp_hist_energy += np.histogram(r_perp, bins=rperp_edges, weights=e)[0]

                rp_cent = float(np.sum(e * r_perp) / etot)
                rp_var = float(np.sum(e * (r_perp - rp_cent) ** 2) / etot)
                rp_width = float(np.sqrt(max(rp_var, 0.0)))
                features["lateral_width"].append(rp_width)
                features["rperp50"].append(_weighted_quantile(r_perp, e, 0.50))
                features["rperp80"].append(_weighted_quantile(r_perp, e, 0.80))
                features["rperp90"].append(_weighted_quantile(r_perp, e, 0.90))
                features["core_fraction_rperp"].append(float(np.sum(e[r_perp < rperp_core_cut]) / etot))
            else:
                features["lateral_width"].append(float("nan"))
                features["rperp50"].append(float("nan"))
                features["rperp80"].append(float("nan"))
                features["rperp90"].append(float("nan"))
                features["core_fraction_rperp"].append(float("nan"))

            if xyz_scatter and len(xyz_samples) < 200:
                pts = np.column_stack([x, y, z])
                if pts.shape[0] > 0:
                    if pts.shape[0] > 1500:
                        pick = rng.choice(pts.shape[0], size=1500, replace=False)
                        pts = pts[pick]
                    xyz_samples.append(pts)

    if event_pbar is not None:
        event_pbar.close()
    file_iter.close()

    if total_events_processed == 0:
        raise RuntimeError(
            f"No events were processed for sample '{sample_label}'. "
            "Check tree name/pattern or increase --max-files."
        )

    features_np = {k: np.asarray(v, dtype=np.float64) for k, v in features.items()}

    axis_ratio = features_np["pca_axis_ratio"]
    finite_ratio = np.isfinite(axis_ratio)
    low_quality = finite_ratio & (axis_ratio < axis_quality_cut)

    summary = {
        "label": sample_label,
        "input_dir": str(input_dir),
        "pattern": pattern,
        "files_processed": int(files_processed),
        "max_events_requested": int(max_events),
        "events_total": int(total_events_processed),
        "features": {k: _safe_stats(v) for k, v in features_np.items()},
        "axis_quality": {
            "ratio_cut": float(axis_quality_cut),
            "finite_ratio_events": int(np.sum(finite_ratio)),
            "low_quality_events": int(np.sum(low_quality)),
            "low_quality_fraction": (
                float(np.sum(low_quality) / np.sum(finite_ratio)) if np.sum(finite_ratio) > 0 else float("nan")
            ),
        },
    }

    xyz_all = np.concatenate(xyz_samples, axis=0) if xyz_samples else np.empty((0, 3), dtype=np.float64)
    return SampleResult(
        summary=summary,
        features=features_np,
        r_hist_energy=r_hist_energy,
        rperp_hist_energy=rperp_hist_energy,
        xyz=xyz_all,
    )


def _std_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    fa = a[np.isfinite(a)]
    fb = b[np.isfinite(b)]
    if fa.size < 2 or fb.size < 2:
        return float("nan")
    ma, mb = np.mean(fa), np.mean(fb)
    va, vb = np.var(fa), np.var(fb)
    pooled = 0.5 * (va + vb)
    if pooled <= 0:
        return float("nan")
    return float((ma - mb) / np.sqrt(pooled))


def _balanced_threshold_score(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    fa = a[np.isfinite(a)]
    fb = b[np.isfinite(b)]
    if fa.size < 2 or fb.size < 2:
        return float("nan"), float("nan")

    vals = np.concatenate([fa, fb])
    lo, hi = np.min(vals), np.max(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float("nan"), float("nan")

    thresholds = np.linspace(lo, hi, 200)
    best_score = -1.0
    best_thr = float("nan")

    for thr in thresholds:
        tpr_gt = float(np.mean(fa > thr))
        fpr_gt = float(np.mean(fb > thr))
        bal_gt = 0.5 * (tpr_gt + (1.0 - fpr_gt))

        tpr_lt = float(np.mean(fa < thr))
        fpr_lt = float(np.mean(fb < thr))
        bal_lt = 0.5 * (tpr_lt + (1.0 - fpr_lt))

        if bal_gt > best_score:
            best_score = bal_gt
            best_thr = float(thr)
        if bal_lt > best_score:
            best_score = bal_lt
            best_thr = float(thr)

    return float(best_score), best_thr


def _compute_separation_metrics(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    momentum_bins: np.ndarray,
) -> dict[str, Any]:
    feature_names = [
        "event_energy",
        "depth_centroid_r",
        "width_r",
        "r50",
        "r80",
        "r90",
        "leakage_fraction_r",
        "core_fraction_r",
        "lateral_width",
        "rperp50",
        "rperp80",
        "rperp90",
        "core_fraction_rperp",
        "pca_axis_ratio",
    ]

    global_metrics: dict[str, Any] = {}
    for name in feature_names:
        d = _std_effect_size(a[name], b[name])
        score, thr = _balanced_threshold_score(a[name], b[name])
        global_metrics[name] = {
            "effect_size": d,
            "best_balanced_score": score,
            "best_threshold": thr,
        }

    ranked = sorted(
        (
            {
                "feature": k,
                "abs_effect_size": abs(v["effect_size"]) if np.isfinite(v["effect_size"]) else float("nan"),
                "best_balanced_score": v["best_balanced_score"],
            }
            for k, v in global_metrics.items()
        ),
        key=lambda x: (-np.nan_to_num(x["abs_effect_size"], nan=-1.0), -np.nan_to_num(x["best_balanced_score"], nan=-1.0)),
    )

    pa = a["gen_momentum_norm"]
    pb = b["gen_momentum_norm"]
    by_bin: list[dict[str, Any]] = []

    for i in range(len(momentum_bins) - 1):
        lo = float(momentum_bins[i])
        hi = float(momentum_bins[i + 1])
        ma = np.isfinite(pa) & (pa >= lo) & (pa < hi)
        mb = np.isfinite(pb) & (pb >= lo) & (pb < hi)

        entry: dict[str, Any] = {
            "momentum_min": lo,
            "momentum_max": hi,
            "n_primary": int(np.sum(ma)),
            "n_comparison": int(np.sum(mb)),
            "features": {},
        }

        for name in feature_names:
            d = _std_effect_size(a[name][ma], b[name][mb])
            score, thr = _balanced_threshold_score(a[name][ma], b[name][mb])
            entry["features"][name] = {
                "effect_size": d,
                "best_balanced_score": score,
                "best_threshold": thr,
            }
        by_bin.append(entry)

    return {
        "global": global_metrics,
        "ranked_features": ranked,
        "by_momentum_bin": by_bin,
    }


def _overlay_hist(ax: Any, a: np.ndarray, b: np.ndarray | None, label_a: str, label_b: str | None, *, bins: int, title: str, xlabel: str) -> None:
    fa = a[np.isfinite(a)]
    if fa.size > 0:
        ax.hist(fa, bins=bins, density=True, alpha=0.65, label=label_a)
    if b is not None and label_b is not None:
        fb = b[np.isfinite(b)]
        if fb.size > 0:
            ax.hist(fb, bins=bins, density=True, alpha=0.65, label=label_b)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)
    ax.legend()


def _plot_profiles(
    *,
    out_dir: Path,
    primary: SampleResult,
    comparison: SampleResult | None,
    args: argparse.Namespace,
) -> None:
    r_edges = np.linspace(args.r_min, args.r_max, args.r_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    rperp_edges = np.linspace(0.0, args.rperp_max, args.rperp_bins + 1)
    rperp_centers = 0.5 * (rperp_edges[:-1] + rperp_edges[1:])

    n_primary = max(1, int(primary.summary["events_total"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_centers, primary.r_hist_energy / n_primary, label=f"{args.label} energy/event")
    if comparison is not None:
        n_comp = max(1, int(comparison.summary["events_total"]))
        ax.plot(r_centers, comparison.r_hist_energy / n_comp, linestyle="--", label=f"{args.compare_label} energy/event")
    ax.set_title("Radial Energy Profile (Sim3dHits)")
    ax.set_xlabel("r = sqrt(x^2+y^2+z^2)")
    ax.set_ylabel("absolute energy per event")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "radial_energy_profile_r.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if primary.r_hist_energy.sum() > 0:
        ax.plot(r_centers, np.cumsum(primary.r_hist_energy) / np.sum(primary.r_hist_energy), label=args.label)
    if comparison is not None and comparison.r_hist_energy.sum() > 0:
        ax.plot(
            r_centers,
            np.cumsum(comparison.r_hist_energy) / np.sum(comparison.r_hist_energy),
            linestyle="--",
            label=args.compare_label,
        )
    ax.set_title("Radial Energy Containment (Sim3dHits)")
    ax.set_xlabel("r = sqrt(x^2+y^2+z^2)")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "radial_energy_containment_r.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rperp_centers, primary.rperp_hist_energy / n_primary, label=f"{args.label} energy/event")
    if comparison is not None:
        n_comp = max(1, int(comparison.summary["events_total"]))
        ax.plot(
            rperp_centers,
            comparison.rperp_hist_energy / n_comp,
            linestyle="--",
            label=f"{args.compare_label} energy/event",
        )
    ax.set_title("Lateral Energy Profile (distance to PCA axis)")
    ax.set_xlabel("r_perp")
    ax.set_ylabel("absolute energy per event")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lateral_energy_profile.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if primary.rperp_hist_energy.sum() > 0:
        ax.plot(
            rperp_centers,
            np.cumsum(primary.rperp_hist_energy) / np.sum(primary.rperp_hist_energy),
            label=args.label,
        )
    if comparison is not None and comparison.rperp_hist_energy.sum() > 0:
        ax.plot(
            rperp_centers,
            np.cumsum(comparison.rperp_hist_energy) / np.sum(comparison.rperp_hist_energy),
            linestyle="--",
            label=args.compare_label,
        )
    ax.set_title("Lateral Energy Containment (distance to PCA axis)")
    ax.set_xlabel("r_perp")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lateral_energy_containment.png", dpi=160)
    plt.close(fig)


def _plot_feature_distributions(
    *,
    out_dir: Path,
    primary: SampleResult,
    comparison: SampleResult | None,
    args: argparse.Namespace,
) -> None:
    cmp_feat = comparison.features if comparison is not None else None
    cmp_label = args.compare_label if comparison is not None else None

    fig, ax = plt.subplots(figsize=(7, 5))
    _overlay_hist(
        ax,
        primary.features["depth_centroid_r"],
        cmp_feat["depth_centroid_r"] if cmp_feat else None,
        args.label,
        cmp_label,
        bins=120,
        title="Energy-Weighted Depth Centroid (r)",
        xlabel="r centroid",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "depth_centroid_r.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _overlay_hist(
        ax,
        primary.features["width_r"],
        cmp_feat["width_r"] if cmp_feat else None,
        args.label,
        cmp_label,
        bins=120,
        title="Longitudinal Shower Width (r RMS)",
        xlabel="width_r",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "shower_width_r.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _overlay_hist(
        ax,
        primary.features["lateral_width"],
        cmp_feat["lateral_width"] if cmp_feat else None,
        args.label,
        cmp_label,
        bins=120,
        title="Lateral Shower Width (r_perp RMS)",
        xlabel="lateral_width",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "lateral_width.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _overlay_hist(
        ax,
        primary.features["leakage_fraction_r"],
        cmp_feat["leakage_fraction_r"] if cmp_feat else None,
        args.label,
        cmp_label,
        bins=80,
        title="Leakage Fraction (E(r > r_leak_cut) / E_total)",
        xlabel="leakage_fraction_r",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "leakage_fraction_r.png", dpi=160)
    plt.close(fig)


def _scatter_downsample(x: np.ndarray, y: np.ndarray, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    xx = x[mask]
    yy = y[mask]
    if xx.size > max_points > 0:
        pick = rng.choice(xx.size, size=max_points, replace=False)
        return xx[pick], yy[pick]
    return xx, yy


def _plot_scatter(
    *,
    out_dir: Path,
    primary: SampleResult,
    comparison: SampleResult | None,
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(args.seed)
    max_scatter = max(1, int(args.shape_scatter_points))

    fig, ax = plt.subplots(figsize=(7, 5))
    x, y = _scatter_downsample(primary.features["depth_centroid_r"], primary.features["width_r"], max_scatter, rng)
    if x.size > 0:
        ax.scatter(x, y, s=6, alpha=0.35, label=args.label)

    if comparison is not None:
        x, y = _scatter_downsample(
            comparison.features["depth_centroid_r"],
            comparison.features["width_r"],
            max_scatter,
            rng,
        )
        if x.size > 0:
            ax.scatter(x, y, s=6, alpha=0.35, label=args.compare_label)

    ax.set_title("Shower Depth vs Longitudinal Width")
    ax.set_xlabel("depth centroid r")
    ax.set_ylabel("width_r")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "depth_vs_width_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    x, y = _scatter_downsample(primary.features["width_r"], primary.features["lateral_width"], max_scatter, rng)
    if x.size > 0:
        ax.scatter(x, y, s=6, alpha=0.35, label=args.label)

    if comparison is not None:
        x, y = _scatter_downsample(comparison.features["width_r"], comparison.features["lateral_width"], max_scatter, rng)
        if x.size > 0:
            ax.scatter(x, y, s=6, alpha=0.35, label=args.compare_label)

    ax.set_title("Longitudinal vs Lateral Width")
    ax.set_xlabel("width_r")
    ax.set_ylabel("lateral_width")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "longitudinal_vs_lateral_width_scatter.png", dpi=160)
    plt.close(fig)

    if args.xyz_scatter and primary.xyz.shape[0] > 0:
        n_keep = min(int(args.scatter_points), primary.xyz.shape[0]) if args.scatter_points > 0 else primary.xyz.shape[0]
        if n_keep < primary.xyz.shape[0]:
            pick = rng.choice(primary.xyz.shape[0], size=n_keep, replace=False)
            pts = primary.xyz[pick]
        else:
            pts = primary.xyz
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.35)
        ax.set_title(f"Example Sim3dHit Scatter (x, y, z) - {args.label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.tight_layout()
        fig.savefig(out_dir / "xyz_scatter_example.png", dpi=160)
        plt.close(fig)


def _plot_separation_vs_momentum(
    out_dir: Path,
    separation: dict[str, Any],
) -> None:
    bins = separation["by_momentum_bin"]
    if not bins:
        return

    top_features = [entry["feature"] for entry in separation["ranked_features"][:6]]
    if not top_features:
        return

    centers = [0.5 * (b["momentum_min"] + b["momentum_max"]) for b in bins]

    fig, ax = plt.subplots(figsize=(8, 5))
    for feat in top_features:
        y = [b["features"][feat]["best_balanced_score"] for b in bins]
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(y)
        if np.any(mask):
            ax.plot(np.asarray(centers)[mask], y[mask], marker="o", linewidth=1.4, label=feat)

    ax.set_title("Feature Separation vs Momentum")
    ax.set_xlabel("|p_gen| bin center")
    ax.set_ylabel("best balanced score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_separation_vs_energy.png", dpi=160)
    plt.close(fig)


def _parse_momentum_bins(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("--momentum-bins must provide at least two edges")
    arr = np.asarray(vals, dtype=np.float64)
    if not np.all(np.diff(arr) > 0):
        raise ValueError("--momentum-bins edges must be strictly increasing")
    return arr


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_compare:
        args.compare_dir = None

    momentum_bins = _parse_momentum_bins(args.momentum_bins)
    r_edges = np.linspace(args.r_min, args.r_max, args.r_bins + 1)
    rperp_edges = np.linspace(0.0, args.rperp_max, args.rperp_bins + 1)

    primary = _collect_sample_events(
        sample_label=args.label,
        input_dir=args.input_dir,
        pattern=args.pattern,
        max_files=args.max_files,
        max_events=args.max_events,
        tree_name=args.tree,
        r_edges=r_edges,
        rperp_edges=rperp_edges,
        r_leak_cut=args.r_leak_cut,
        r_core_cut=args.r_core_cut,
        rperp_core_cut=args.rperp_core_cut,
        axis_quality_cut=args.axis_quality_cut,
        xyz_scatter=args.xyz_scatter,
        seed=args.seed,
    )

    comparison = None
    separation = None
    if args.compare_dir is not None:
        comparison = _collect_sample_events(
            sample_label=args.compare_label,
            input_dir=args.compare_dir,
            pattern=args.compare_pattern,
            max_files=args.max_files,
            max_events=args.max_events,
            tree_name=args.tree,
            r_edges=r_edges,
            rperp_edges=rperp_edges,
            r_leak_cut=args.r_leak_cut,
            r_core_cut=args.r_core_cut,
            rperp_core_cut=args.rperp_core_cut,
            axis_quality_cut=args.axis_quality_cut,
            xyz_scatter=False,
            seed=args.seed + 1,
        )
        separation = _compute_separation_metrics(primary.features, comparison.features, momentum_bins)

    _plot_profiles(out_dir=out_dir, primary=primary, comparison=comparison, args=args)
    _plot_feature_distributions(out_dir=out_dir, primary=primary, comparison=comparison, args=args)
    _plot_scatter(out_dir=out_dir, primary=primary, comparison=comparison, args=args)
    if separation is not None:
        _plot_separation_vs_momentum(out_dir, separation)

    payload: dict[str, Any] = {
        "config": {
            "r_min": float(args.r_min),
            "r_max": float(args.r_max),
            "r_bins": int(args.r_bins),
            "r_leak_cut": float(args.r_leak_cut),
            "r_core_cut": float(args.r_core_cut),
            "rperp_max": float(args.rperp_max),
            "rperp_bins": int(args.rperp_bins),
            "rperp_core_cut": float(args.rperp_core_cut),
            "momentum_bins": [float(x) for x in momentum_bins],
            "seed": int(args.seed),
            "axis_quality_cut": float(args.axis_quality_cut),
            "xyz_scatter": bool(args.xyz_scatter),
        },
        "primary": primary.summary,
    }

    if comparison is not None:
        payload["comparison"] = comparison.summary
    if separation is not None:
        payload["separation"] = separation

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Saved outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
