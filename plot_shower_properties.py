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
        default=["theta", "phi", "e_gen", "e_dep", "S_amp", "C_amp"],
        help="Property names or raw HDF5 keys.",
    )
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument("--max-events", type=int, default=10000, help="Per-file event cap (<=0 means all).")
    parser.add_argument("--out-dir", default="plots/shower_properties", help="Output directory.")
    parser.add_argument("--density", action="store_true", help="Normalize histograms to density.")
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


def _read_unique_readout_counts(handle: h5py.File, cell_key: str, max_events: int) -> np.ndarray:
    if cell_key not in handle:
        return np.array([], dtype=np.float64)
    ds = handle[cell_key]
    if ds.ndim == 0:
        return np.array([], dtype=np.float64)
    n_events = ds.shape[0]
    take = n_events if max_events <= 0 else min(n_events, max_events)
    cells = np.asarray(ds[:take])
    if cells.ndim == 1:
        cells = cells.reshape(-1, 1)

    counts = np.zeros(cells.shape[0], dtype=np.float64)
    for i in range(cells.shape[0]):
        row = np.asarray(cells[i]).reshape(-1)
        valid = np.isfinite(row) & (row != 0)
        if not np.any(valid):
            counts[i] = 0.0
        else:
            counts[i] = float(np.unique(row[valid]).size)
    return counts


def _event_count(handle: h5py.File, preferred_keys: List[str]) -> int:
    for key in preferred_keys:
        if key in handle and handle[key].ndim >= 1:
            return int(handle[key].shape[0])
    for key in handle.keys():
        ds = handle[key]
        if ds.ndim >= 1:
            return int(ds.shape[0])
    return 0


def _collect(files: List[Path], keys: List[str], max_events: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
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

            unique_readout = _read_unique_readout_counts(handle, "DRcalo3dHits.cellID", max_events)
            if unique_readout.size > 0:
                file_data["num_unique_readout"] = unique_readout
                by_key_total.setdefault("num_unique_readout", []).append(unique_readout)
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


def main() -> None:
    args = parse_args()
    files = _resolve_paths(args.files)
    if not files:
        raise SystemExit("No input files matched.")

    keys = [_canonical_key(p) for p in args.properties]
    for required in ("GenParticles.momentum.theta", "E_dep", "E_gen", "E_leak", "DRcalo3dHits.cellID"):
        if required not in keys:
            keys.append(required)
    total, by_file = _collect(files, keys, args.max_events)
    _plot(total, by_file, Path(args.out_dir), args.bins, args.density)
    _plot_theta_energy_scatter(by_file, Path(args.out_dir))
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
