"""Plot shower-property distributions from HDF5 files.

Example
-------
python plot_shower_properties.py --files e-_1-100GeV_1.h5py gamma_1-100GeV_1.h5py \
  --properties theta phi e_gen e_dep S_amp C_amp

python plot_shower_properties.py \
  --files "h5s/*_1-100GeV_1.h5py" \
  --properties theta phi e_gen e_dep S_amp C_amp \
  --bins 150 --density --out-dir plots/shower_properties
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ALIASES: Dict[str, str] = {
    "theta": "GenParticles.momentum.theta",
    "phi": "GenParticles.momentum.phi",
    "e_gen": "E_gen",
    "e_dep": "E_dep",
    "e_leak": "E_leak",
    "s_amp": "S_amp",
    "c_amp": "C_amp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot shower-property histograms for selected HDF5 files.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Input files or globs (e.g. e-_1-100GeV_1.h5py or h5s/*.h5py).",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["theta", "phi", "e_gen", "e_dep", "e_leak", "S_amp", "C_amp"],
        help="Property names or raw HDF5 keys.",
    )
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument("--max-events", type=int, default=20000, help="Per-file event cap (<=0 means all).")
    parser.add_argument("--out-dir", default="plots/shower_properties", help="Output directory.")
    parser.add_argument("--density", action="store_true", help="Normalize histograms to density.")
    parser.add_argument(
        "--skip-scatter",
        action="store_true",
        help="Skip theta-vs-energy scatter plots (faster for large inputs).",
    )
    parser.add_argument(
        "--mask-feature",
        default="DRcalo3dHits.amplitude_sum",
        help="Feature used to define effective points (finite and non-zero).",
    )
    parser.add_argument(
        "--max-point-sweep",
        nargs="+",
        type=int,
        default=[1000, 3000, 5000, 10000],
        help="Cutoffs used for unique-readout vs max-point curve.",
    )
    parser.add_argument(
        "--energy-feature",
        default="DRcalo3dHits.amplitude_sum",
        help="Per-hit energy-like feature used for retained-energy fraction vs max point length.",
    )
    return parser.parse_args()


def _canonical_key(name: str) -> str:
    lowered = name.strip().lower()
    return ALIASES.get(lowered, name)


def _resolve_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    roots = [Path("."), Path("h5s"), Path("/store/ml/dual-readout/h5s")]
    for pattern in patterns:
        found: List[Path] = []
        for root in roots:
            found.extend(sorted(root.glob(pattern)))
        if found:
            paths.extend(found)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
            else:
                for root in roots[1:]:
                    alt = root / pattern
                    if alt.exists():
                        paths.append(alt)
                        break
    dedup: List[Path] = []
    seen = set()
    for p in paths:
        try:
            key = str(p.resolve())
        except FileNotFoundError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def _read_1d(handle: h5py.File, key: str, max_events: int) -> np.ndarray:
    ds = handle[key]
    if ds.ndim == 0:
        arr = np.asarray(ds[()], dtype=np.float64).reshape(1)
    else:
        n = ds.shape[0]
        take = n if max_events <= 0 else min(n, max_events)
        arr = np.asarray(ds[:take], dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _read_effective_readout_metrics(
    handle: h5py.File,
    *,
    cell_key: str,
    mask_key: str,
    energy_key: str,
    max_events: int,
    cutoffs: List[int],
) -> Dict[str, np.ndarray]:
    if cell_key not in handle:
        return {}
    cell_ds = handle[cell_key]
    if cell_ds.ndim == 0:
        return {}
    n_events = cell_ds.shape[0]
    take = n_events if max_events <= 0 else min(n_events, max_events)
    cells = np.asarray(cell_ds[:take])
    if cells.ndim == 1:
        cells = cells.reshape(-1, 1)

    mask_arr = None
    if mask_key in handle:
        mask_ds = handle[mask_key]
        if mask_ds.ndim >= 1 and mask_ds.shape[0] >= take:
            mask_arr = np.asarray(mask_ds[:take])
            if mask_arr.ndim == 1:
                mask_arr = mask_arr.reshape(-1, 1)

    energy_arr = None
    if energy_key in handle:
        energy_ds = handle[energy_key]
        if energy_ds.ndim >= 1 and energy_ds.shape[0] >= take:
            energy_arr = np.asarray(energy_ds[:take])
            if energy_arr.ndim == 1:
                energy_arr = energy_arr.reshape(-1, 1)

    n_cut = len(cutoffs)
    effective_points = np.zeros(take, dtype=np.float64)
    unique_raw = np.zeros(take, dtype=np.float64)
    unique_full = np.zeros(take, dtype=np.float64)
    ratio = np.full(take, np.nan, dtype=np.float64)
    unique_vs_cutoff = np.zeros((take, n_cut), dtype=np.float64)
    energy_frac_vs_cutoff = np.full((take, n_cut), np.nan, dtype=np.float64)

    for i in range(take):
        row_cell = np.asarray(cells[i]).reshape(-1)
        raw_valid = np.isfinite(row_cell) & (row_cell != 0)
        unique_raw[i] = float(np.unique(row_cell[raw_valid]).size) if np.any(raw_valid) else 0.0
        valid = raw_valid
        if mask_arr is not None and mask_arr.shape[1] == row_cell.shape[0]:
            row_mask = np.asarray(mask_arr[i]).reshape(-1)
            valid = valid & np.isfinite(row_mask) & (row_mask != 0)
        eff_cells = row_cell[valid]
        n_eff = int(eff_cells.size)
        effective_points[i] = float(n_eff)
        if n_eff == 0:
            continue
        n_unique = int(np.unique(eff_cells).size)
        unique_full[i] = float(n_unique)
        if n_unique > 0:
            ratio[i] = float(n_eff / n_unique)
        for j, n_cutoff in enumerate(cutoffs):
            kept = eff_cells[: min(n_eff, n_cutoff)]
            unique_vs_cutoff[i, j] = float(np.unique(kept).size) if kept.size > 0 else 0.0

        if energy_arr is not None and energy_arr.shape[1] == row_cell.shape[0]:
            row_energy = np.asarray(energy_arr[i]).reshape(-1)
            row_energy = np.where(np.isfinite(row_energy), row_energy, 0.0)
            eff_energy = row_energy[valid]
            total_energy = float(np.sum(eff_energy))
            if total_energy > 0.0:
                for j, n_cutoff in enumerate(cutoffs):
                    kept_e = eff_energy[: min(n_eff, n_cutoff)]
                    energy_frac_vs_cutoff[i, j] = float(np.sum(kept_e) / total_energy)

    out: Dict[str, np.ndarray] = {
        "num_unique_readout": unique_raw,
        "effective_points": effective_points,
        "num_unique_readout_effective": unique_full,
        "effective_points_per_unique_readout": ratio[np.isfinite(ratio)],
    }
    for j, n_cutoff in enumerate(cutoffs):
        out[f"num_unique_readout_at_n{n_cutoff}"] = unique_vs_cutoff[:, j]
        frac_vals = energy_frac_vs_cutoff[:, j]
        out[f"energy_fraction_at_n{n_cutoff}"] = frac_vals[np.isfinite(frac_vals)]
    return out


def _event_count(handle: h5py.File, preferred_keys: List[str]) -> int:
    for key in preferred_keys:
        if key in handle and handle[key].ndim >= 1:
            return int(handle[key].shape[0])
    for key in handle.keys():
        ds = handle[key]
        if ds.ndim >= 1:
            return int(ds.shape[0])
    return 0


def _collect(
    files: List[Path],
    keys: List[str],
    max_events: int,
    mask_feature: str,
    energy_feature: str,
    max_point_sweep: List[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    by_key_total: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    by_file: Dict[str, Dict[str, np.ndarray]] = {}

    for path in files:
        with h5py.File(path, "r") as handle:
            total_events = _event_count(handle, keys)
        if max_events > 0:
            print(f"{path.name}: total_events={total_events} (using up to {min(total_events, max_events)})")
        else:
            print(f"{path.name}: total_events={total_events}")

    for path in tqdm(files, desc="Reading files"):
        file_data: Dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            for key in tqdm(keys, desc=f"  {path.name} vars", leave=False):
                # Cell-level diagnostics are handled in one dedicated pass below.
                if key == "DRcalo3dHits.cellID":
                    continue
                if key not in handle:
                    continue
                values = _read_1d(handle, key, max_events)
                if values.size == 0:
                    continue
                file_data[key] = values
                by_key_total.setdefault(key, []).append(values)

            if "E_dep" in file_data and "E_gen" in file_data:
                n = min(file_data["E_dep"].size, file_data["E_gen"].size)
                if n > 0:
                    e_dep = file_data["E_dep"][:n]
                    e_gen = file_data["E_gen"][:n]
                    valid = np.isfinite(e_dep) & np.isfinite(e_gen) & (e_gen != 0)
                    ratio = np.array([], dtype=np.float64)
                    if np.any(valid):
                        ratio = e_dep[valid] / e_gen[valid]
                    file_data["E_dep_over_E_gen"] = ratio
                    by_key_total.setdefault("E_dep_over_E_gen", []).append(ratio)

            extra = _read_effective_readout_metrics(
                handle,
                cell_key="DRcalo3dHits.cellID",
                mask_key=mask_feature,
                energy_key=energy_feature,
                max_events=max_events,
                cutoffs=max_point_sweep,
            )
            for extra_key, extra_values in extra.items():
                if extra_values.size == 0:
                    continue
                file_data[extra_key] = extra_values
                by_key_total.setdefault(extra_key, []).append(extra_values)
        by_file[path.name] = file_data

    total: Dict[str, np.ndarray] = {}
    for key, chunks in by_key_total.items():
        if chunks:
            total[key] = np.concatenate(chunks)
        else:
            total[key] = np.array([], dtype=np.float64)
    return total, by_file


def _plot_theta_energy_scatter(by_file: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> None:
    theta_key = "GenParticles.momentum.theta"
    y_keys = ["E_dep", "E_gen", "E_leak", "E_dep_over_E_gen"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for y_key in y_keys:
        all_theta: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        fig, ax = plt.subplots(figsize=(8, 6))

        for file_name, file_map in by_file.items():
            theta = file_map.get(theta_key)
            y_val = file_map.get(y_key)
            if theta is None or y_val is None or theta.size == 0 or y_val.size == 0:
                continue
            n = min(theta.size, y_val.size)
            if n == 0:
                continue
            theta_n = theta[:n]
            y_n = y_val[:n]
            all_theta.append(theta_n)
            all_y.append(y_n)
            ax.scatter(theta_n, y_n, s=5, alpha=0.2, label=file_name)

        if not all_theta:
            plt.close(fig)
            print(f"Skipping scatter theta vs {y_key}: missing data")
            continue

        theta_cat = np.concatenate(all_theta)
        y_cat = np.concatenate(all_y)

        ax.set_xlabel(theta_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"Scatter: theta vs {y_key}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, markerscale=2)
        fig.tight_layout()

        out_path = out_dir / f"scatter_theta_vs_{y_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"theta vs {y_key}: pairs={theta_cat.size}")


def _plot(total: Dict[str, np.ndarray], by_file: Dict[str, Dict[str, np.ndarray]], out_dir: Path, bins: int, density: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, values in total.items():
        if values.size == 0:
            print(f"Skipping {key}: no data found in inputs")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=bins, histtype="step", linewidth=1.6, color="black", density=density, label="all files")

        for file_name, file_map in by_file.items():
            if key not in file_map:
                continue
            ax.hist(file_map[key], bins=bins, histtype="step", linewidth=1.0, alpha=0.7, density=density, label=file_name)

        ax.set_xlabel(key)
        ax.set_ylabel("density" if density else "count")
        ax.set_title(f"Distribution: {key}")
        if key == "E_leak":
            ax.set_yscale("log", nonpositive="clip")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()

        safe_key = key.replace("/", "_").replace(".", "_")
        out_path = out_dir / f"dist_{safe_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        q = np.quantile(values, [0.01, 0.5, 0.99])
        print(
            f"{key}: n={values.size} min={values.min():.6g} p01={q[0]:.6g} median={q[1]:.6g} "
            f"p99={q[2]:.6g} max={values.max():.6g}"
        )


def _plot_effective_points_per_unique_readout_avg(by_file: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> None:
    key = "effective_points_per_unique_readout"
    labels: List[str] = []
    means: List[float] = []
    all_values: List[np.ndarray] = []
    for file_name in sorted(by_file.keys()):
        vals = by_file[file_name].get(key)
        if vals is None or vals.size == 0:
            continue
        labels.append(file_name)
        means.append(float(np.mean(vals)))
        all_values.append(vals)
    if not labels:
        return
    if all_values:
        labels.append("all")
        means.append(float(np.mean(np.concatenate(all_values))))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("avg effective points / unique readout")
    ax.set_title("Average effective points per unique readout")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "avg_effective_points_per_unique_readout.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_unique_readout_vs_max_points(
    by_file: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    max_point_sweep: List[int],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    used_any = False
    per_n_all: Dict[int, List[np.ndarray]] = {n: [] for n in max_point_sweep}

    for file_name in sorted(by_file.keys()):
        xs: List[int] = []
        ys: List[float] = []
        for n_cutoff in max_point_sweep:
            key = f"num_unique_readout_at_n{n_cutoff}"
            vals = by_file[file_name].get(key)
            if vals is None or vals.size == 0:
                continue
            xs.append(n_cutoff)
            ys.append(float(np.mean(vals)))
            per_n_all[n_cutoff].append(vals)
        if xs:
            used_any = True
            ax.plot(xs, ys, marker="o", label=file_name, alpha=0.85)

    xs_all: List[int] = []
    ys_all: List[float] = []
    for n_cutoff in max_point_sweep:
        chunks = per_n_all[n_cutoff]
        if not chunks:
            continue
        xs_all.append(n_cutoff)
        ys_all.append(float(np.mean(np.concatenate(chunks))))
    if xs_all:
        used_any = True
        ax.plot(xs_all, ys_all, marker="o", linewidth=2.2, color="black", label="all files")

    if not used_any:
        plt.close(fig)
        return
    ax.set_xlabel("max point length (n)")
    ax.set_ylabel("avg unique readout count (effective prefix)")
    ax.set_title("Unique readout count vs max point length")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "unique_readout_vs_max_point_length.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_energy_fraction_vs_max_points(
    by_file: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    max_point_sweep: List[int],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    used_any = False
    per_n_all: Dict[int, List[np.ndarray]] = {n: [] for n in max_point_sweep}

    for file_name in sorted(by_file.keys()):
        xs: List[int] = []
        ys: List[float] = []
        for n_cutoff in max_point_sweep:
            key = f"energy_fraction_at_n{n_cutoff}"
            vals = by_file[file_name].get(key)
            if vals is None or vals.size == 0:
                continue
            xs.append(n_cutoff)
            ys.append(float(np.mean(vals)))
            per_n_all[n_cutoff].append(vals)
        if xs:
            used_any = True
            ax.plot(xs, ys, marker="o", label=file_name, alpha=0.85)

    xs_all: List[int] = []
    ys_all: List[float] = []
    for n_cutoff in max_point_sweep:
        chunks = per_n_all[n_cutoff]
        if not chunks:
            continue
        xs_all.append(n_cutoff)
        ys_all.append(float(np.mean(np.concatenate(chunks))))
    if xs_all:
        used_any = True
        ax.plot(xs_all, ys_all, marker="o", linewidth=2.2, color="black", label="all files")

    if not used_any:
        plt.close(fig)
        return
    ax.set_xlabel("max point length (n)")
    ax.set_ylabel("avg retained energy fraction")
    ax.set_title("Retained energy fraction vs max point length")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "energy_fraction_vs_max_point_length.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    files = _resolve_paths(args.files)
    if not files:
        raise SystemExit("No input files matched.")

    keys = [_canonical_key(p) for p in args.properties]
    for required in ("GenParticles.momentum.theta", "E_dep", "E_gen", "E_leak"):
        if required not in keys:
            keys.append(required)
    sweep = sorted({n for n in args.max_point_sweep if n > 0})
    total, by_file = _collect(files, keys, args.max_events, args.mask_feature, args.energy_feature, sweep)
    out_dir = Path(args.out_dir)
    _plot(total, by_file, out_dir, args.bins, args.density)
    if not args.skip_scatter:
        _plot_theta_energy_scatter(by_file, out_dir)
    _plot_effective_points_per_unique_readout_avg(by_file, out_dir)
    _plot_unique_readout_vs_max_points(by_file, out_dir, sweep)
    _plot_energy_fraction_vs_max_points(by_file, out_dir, sweep)
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
