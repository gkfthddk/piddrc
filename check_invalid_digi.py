#!/usr/bin/env python
"""Read invalid-file lists from toh5.py and validate corresponding digi ROOT files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from tqdm.auto import tqdm

SIM3D_E = "Sim3dCalorimeterHits/Sim3dCalorimeterHits.energy"
DIGI_E = "DigiCalorimeterHits/DigiCalorimeterHits.energy"
DIGI_T = "DigiCalorimeterHits/DigiCalorimeterHits.type"
GPX = "GenParticles/GenParticles.momentum.x"
GPY = "GenParticles/GenParticles.momentum.y"
GPZ = "GenParticles/GenParticles.momentum.z"
C_AMP = "C_amp"
S_AMP = "S_amp"
E_DEP = "E_dep"
E_GEN = "E_gen"
THETA = "GenParticles/GenParticles.momentum.theta"
PHI = "GenParticles/GenParticles.momentum.phi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate corresponding digi ROOT files from invalid reco/digi lists.")
    parser.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help=(
            "Path to *_invalid_reco.txt or *_invalid_digi.txt produced by toh5.py. "
            "If omitted, run all discovered list files."
        ),
    )
    parser.add_argument(
        "--digi-base",
        type=Path,
        default=Path("/hdfs/ml/dualreadout/v001"),
        help="Base digi directory used when list-file contains reco .h5py paths.",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default=None,
        help="Sample name override (normally inferred from list file name).",
    )
    parser.add_argument("--tree", type=str, default="events", help="TTree name to check.")
    parser.add_argument(
        "--check-branches",
        type=str,
        default="RawCalorimeterHits/RawCalorimeterHits.amplitude,Sim3dCalorimeterHits/Sim3dCalorimeterHits.energy",
        help="Comma-separated branch names to verify in tree.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Only check first N paths from list (<=0 means all).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path (default: next to list-file with *_check.json).",
    )
    parser.add_argument(
        "--summary-plots",
        action="store_true",
        help="Generate summary plots for invalid-file digi distributions.",
    )
    parser.add_argument(
        "--no-summary-plots",
        dest="summary_plots",
        action="store_false",
        help="Disable summary plot generation.",
    )
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=2000,
        help="Max events per ROOT file to sample for summary plots (<=0 means all).",
    )
    parser.add_argument(
        "--plot-out-dir",
        type=Path,
        default=None,
        help="Directory for summary plots (default: <out_json stem>_plots).",
    )
    parser.add_argument(
        "--cs-type-map",
        type=str,
        default="0,1",
        help="Explicit DigiCalorimeterHits.type mapping 'Ctype,Stype' used for C/S proxies (default: 0,1).",
    )
    parser.add_argument(
        "--auto-cs-type-map",
        action="store_true",
        help="Infer C/S DigiCalorimeterHits.type mapping from data (slower than --cs-type-map).",
    )
    parser.set_defaults(summary_plots=True)
    return parser.parse_args()


def _pick_default_list_files() -> list[Path]:
    """
    Auto-pick newest reco-invalid list with current preferred layout first.
    Search order:
    1) h5s/validation_reports/*_readable_but_invalid_reco.txt
    2) h5s/validation_reports/*_readable_mixed_reco.txt
    3) h5s/validation_reports/*_invalid_reco.txt
    4) h5s/invalid_lists/*_invalid_reco.txt (legacy)
    5) h5s/invalid_lists/*_invalid_digi.txt (legacy)
    """
    candidates: list[Path] = []
    for pattern in (
        "h5s/validation_reports/*_readable_but_invalid_reco.txt",
        "h5s/validation_reports/*_readable_mixed_reco.txt",
        "h5s/validation_reports/*_invalid_reco.txt",
        "h5s/invalid_lists/*_invalid_reco.txt",
        "h5s/invalid_lists/*_invalid_digi.txt",
    ):
        candidates.extend(Path(".").glob(pattern))

    if candidates:
        # De-duplicate and process oldest->newest for stable progression.
        unique = sorted(set(candidates), key=lambda p: p.stat().st_mtime)
        return unique

    raise FileNotFoundError(
        "No invalid list files found. Searched:\n"
        "- h5s/validation_reports/*_readable_but_invalid_reco.txt\n"
        "- h5s/validation_reports/*_readable_mixed_reco.txt\n"
        "- h5s/validation_reports/*_invalid_reco.txt\n"
        "- h5s/invalid_lists/*_invalid_reco.txt\n"
        "- h5s/invalid_lists/*_invalid_digi.txt\n"
        "Run toh5.py --check_only first or pass --list-file explicitly."
    )


def _infer_sample_name(list_file: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = list_file.name
    for suffix in (
        "_readable_but_invalid_reco.txt",
        "_readable_mixed_reco.txt",
        "_readable_valid_reco.txt",
        "_unreadable_reco.txt",
        "_invalid_reco.txt",
        "_invalid_digi.txt",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(
        "Could not infer sample name from list file name. "
        "Use --sample-name when file does not follow '*_invalid_reco.txt' or '*_invalid_digi.txt'."
    )


def _parse_line_to_path(line: str) -> Path | None:
    s = line.strip()
    if not s:
        return None
    if s.startswith("["):
        end = s.find("]")
        if end >= 0:
            s = s[end + 1 :].strip()
    return Path(s)


def _to_digi_path(src_path: Path, sample_name: str, digi_base: Path) -> Path:
    if src_path.suffix == ".root":
        return src_path
    if src_path.suffix == ".h5py":
        return digi_base / sample_name / f"digi_{src_path.stem}.root"
    return src_path


def _read_list_paths(list_file: Path, sample_name: str, digi_base: Path) -> list[Path]:
    lines = list_file.read_text(encoding="utf-8").splitlines()
    out: list[Path] = []
    for line in lines:
        p = _parse_line_to_path(line)
        if p is None:
            continue
        out.append(_to_digi_path(p, sample_name, digi_base))
    return out


def _check_one(path: Path, tree_name: str, branches: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "open_ok": False,
        "tree_exists": False,
        "num_entries": None,
        "missing_branches": [],
        "status": "missing_file",
        "error": None,
    }
    if not record["exists"]:
        return record

    try:
        with uproot.open(path) as handle:
            record["open_ok"] = True
            if tree_name not in handle:
                record["status"] = "missing_tree"
                return record
            record["tree_exists"] = True
            tree = handle[tree_name]
            record["num_entries"] = int(tree.num_entries)

            missing = [b for b in branches if b and b not in tree]
            record["missing_branches"] = missing
            if missing:
                record["status"] = "missing_branches"
            else:
                record["status"] = "ok"
            return record
    except Exception as exc:
        record["status"] = "read_error"
        record["error"] = str(exc)
        return record


def _safe_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
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
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _infer_cs_types(paths: list[Path], tree_name: str, max_events_per_file: int, explicit_map: str | None) -> tuple[int | None, int | None, dict[str, Any]]:
    if explicit_map:
        tokens = [x.strip() for x in explicit_map.split(",") if x.strip()]
        if len(tokens) == 2:
            return int(tokens[0]), int(tokens[1]), {"source": "explicit_map", tokens[0]: 0, tokens[1]: 0}

    counts: dict[str, Any] = {"source": "auto_inferred"}
    for p in tqdm(paths, desc="Inferring C/S type map", unit="file", leave=False):
        try:
            with uproot.open(p) as h:
                if tree_name not in h:
                    continue
                t = h[tree_name]
                if DIGI_T not in t:
                    continue
                entry_stop = None if max_events_per_file <= 0 else max_events_per_file
                arr = t.arrays([DIGI_T], entry_stop=entry_stop, library="ak")
                vals = np.asarray(ak.flatten(arr[DIGI_T], axis=None), dtype=np.int64)
                if vals.size == 0:
                    continue
                u, c = np.unique(vals, return_counts=True)
                for ui, ci in zip(u, c):
                    key = str(int(ui))
                    counts[key] = int(counts.get(key, 0)) + int(ci)
        except Exception:
            continue

    numeric_items = [(k, v) for k, v in counts.items() if k != "source"]
    if not numeric_items:
        return None, None, counts

    sorted_types = sorted(numeric_items, key=lambda kv: kv[1], reverse=True)
    c_type = int(sorted_types[0][0])
    s_type = int(sorted_types[1][0]) if len(sorted_types) > 1 else None
    return c_type, s_type, counts


def _collect_summary_features(
    paths: list[Path],
    tree_name: str,
    max_events_per_file: int,
    c_type: int | None,
    s_type: int | None,
) -> dict[str, np.ndarray]:
    feat: dict[str, list[float]] = {
        "C_amp": [],
        "S_amp": [],
        "E_dep": [],
        "E_gen": [],
        "gen_theta": [],
        "gen_phi": [],
    }

    for p in tqdm(paths, desc="Reading summary features", unit="file"):
        try:
            with uproot.open(p) as h:
                if tree_name not in h:
                    continue
                t = h[tree_name]
                wanted = [
                    b
                    for b in (C_AMP, S_AMP, E_DEP, E_GEN, THETA, PHI, SIM3D_E, DIGI_E, DIGI_T, GPX, GPY, GPZ)
                    if b in t
                ]
                if not wanted:
                    continue
                entry_stop = None if max_events_per_file <= 0 else max_events_per_file
                arr = t.arrays(wanted, entry_stop=entry_stop, library="ak")
        except Exception:
            continue

        n_events = len(arr[wanted[0]]) if wanted else 0

        if E_DEP in arr.fields:
            v = np.asarray(arr[E_DEP], dtype=np.float64).reshape(-1)
            feat["E_dep"].extend(v[np.isfinite(v)].tolist())
        elif SIM3D_E in arr.fields:
            e_dep = np.asarray(ak.sum(arr[SIM3D_E], axis=1), dtype=np.float64)
            feat["E_dep"].extend(e_dep[np.isfinite(e_dep)].tolist())

        if E_GEN in arr.fields:
            v = np.asarray(arr[E_GEN], dtype=np.float64).reshape(-1)
            feat["E_gen"].extend(v[np.isfinite(v)].tolist())
        elif GPX in arr.fields and GPY in arr.fields and GPZ in arr.fields:
            px = np.asarray(ak.fill_none(ak.firsts(arr[GPX]), np.nan), dtype=np.float64)
            py = np.asarray(ak.fill_none(ak.firsts(arr[GPY]), np.nan), dtype=np.float64)
            pz = np.asarray(ak.fill_none(ak.firsts(arr[GPZ]), np.nan), dtype=np.float64)
            pnorm = np.sqrt(px * px + py * py + pz * pz)
            feat["E_gen"].extend(pnorm[np.isfinite(pnorm)].tolist())

        if THETA in arr.fields and PHI in arr.fields:
            theta = np.asarray(ak.fill_none(ak.firsts(arr[THETA]), np.nan), dtype=np.float64)
            phi = np.asarray(ak.fill_none(ak.firsts(arr[PHI]), np.nan), dtype=np.float64)
            feat["gen_theta"].extend(theta[np.isfinite(theta)].tolist())
            feat["gen_phi"].extend(phi[np.isfinite(phi)].tolist())
        elif GPX in arr.fields and GPY in arr.fields and GPZ in arr.fields:
            px = np.asarray(ak.fill_none(ak.firsts(arr[GPX]), np.nan), dtype=np.float64)
            py = np.asarray(ak.fill_none(ak.firsts(arr[GPY]), np.nan), dtype=np.float64)
            pz = np.asarray(ak.fill_none(ak.firsts(arr[GPZ]), np.nan), dtype=np.float64)
            theta = np.arctan2(np.sqrt(px * px + py * py), pz)
            phi = np.arctan2(py, px)
            feat["gen_theta"].extend(theta[np.isfinite(theta)].tolist())
            feat["gen_phi"].extend(phi[np.isfinite(phi)].tolist())

        if C_AMP in arr.fields:
            v = np.asarray(arr[C_AMP], dtype=np.float64).reshape(-1)
            feat["C_amp"].extend(v[np.isfinite(v)].tolist())
        if S_AMP in arr.fields:
            v = np.asarray(arr[S_AMP], dtype=np.float64).reshape(-1)
            feat["S_amp"].extend(v[np.isfinite(v)].tolist())

        if (C_AMP not in arr.fields or S_AMP not in arr.fields) and DIGI_E in arr.fields and DIGI_T in arr.fields and n_events > 0 and c_type is not None:
            en = arr[DIGI_E]
            ty = arr[DIGI_T]
            c_evt = ak.sum(ak.where(ty == c_type, en, 0.0), axis=1)
            c_np = np.asarray(c_evt, dtype=np.float64)
            feat["C_amp"].extend(c_np[np.isfinite(c_np)].tolist())
            if s_type is not None:
                s_evt = ak.sum(ak.where(ty == s_type, en, 0.0), axis=1)
                s_np = np.asarray(s_evt, dtype=np.float64)
                feat["S_amp"].extend(s_np[np.isfinite(s_np)].tolist())

    return {k: np.asarray(v, dtype=np.float64) for k, v in feat.items()}


def _plot_1d(values: np.ndarray, title: str, xlabel: str, out_path: Path) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(finite, bins=120, density=True, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _make_summary_plots(features: dict[str, np.ndarray], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    if _plot_1d(features["C_amp"], "C_amp distribution", "C_amp", out_dir / "C_amp.png"):
        made.append("C_amp.png")
    if _plot_1d(features["S_amp"], "S_amp distribution", "S_amp", out_dir / "S_amp.png"):
        made.append("S_amp.png")
    if _plot_1d(features["E_dep"], "E_dep distribution", "E_dep", out_dir / "E_dep.png"):
        made.append("E_dep.png")
    if _plot_1d(features["E_gen"], "E_gen distribution", "E_gen", out_dir / "E_gen.png"):
        made.append("E_gen.png")

    th = features["gen_theta"]
    ph = features["gen_phi"]
    mask = np.isfinite(th) & np.isfinite(ph)
    if np.any(mask):
        fig, ax = plt.subplots(figsize=(7, 5))
        h = ax.hist2d(ph[mask], th[mask], bins=100, cmap="viridis")
        ax.set_title("Gen particle direction (phi vs theta)")
        ax.set_xlabel("phi [rad]")
        ax.set_ylabel("theta [rad]")
        fig.colorbar(h[3], ax=ax, label="counts")
        fig.tight_layout()
        fig.savefig(out_dir / "gen_direction_theta_phi.png", dpi=160)
        plt.close(fig)
        made.append("gen_direction_theta_phi.png")

    return made


def main() -> int:
    args = parse_args()
    if args.list_file is None:
        list_files = _pick_default_list_files()
        print(f"Auto-selected {len(list_files)} list files")
    else:
        list_files = [args.list_file]

    branches = [b.strip() for b in args.check_branches.split(",") if b.strip()]
    multi = len(list_files) > 1
    all_run_summaries: list[dict[str, Any]] = []

    for list_file in list_files:
        if not list_file.is_file():
            raise FileNotFoundError(f"List file not found: {list_file}")

        sample_name = _infer_sample_name(list_file, args.sample_name)
        digi_paths = _read_list_paths(list_file, sample_name, args.digi_base)
        if args.max_files > 0:
            digi_paths = digi_paths[: args.max_files]
        if not digi_paths:
            raise RuntimeError(f"No paths found in list file: {list_file}")

        results: list[dict[str, Any]] = []
        for p in tqdm(digi_paths, desc=f"Checking digi ROOT [{sample_name}]", unit="file"):
            results.append(_check_one(p, args.tree, branches))

        counts: dict[str, int] = {}
        for row in results:
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + 1

        payload: dict[str, Any] = {
            "list_file": str(list_file),
            "sample_name": sample_name,
            "tree": args.tree,
            "checked_branches": branches,
            "digi_base": str(args.digi_base),
            "num_files": len(results),
            "status_counts": counts,
            "files": results,
        }

        out_json: Path
        if args.out_json is None:
            out_json = list_file.with_name(list_file.stem + "_check.json")
        elif not multi:
            out_json = args.out_json
        else:
            if args.out_json.suffix == ".json":
                out_json = args.out_json.with_name(f"{args.out_json.stem}__{list_file.stem}.json")
            else:
                args.out_json.mkdir(parents=True, exist_ok=True)
                out_json = args.out_json / f"{list_file.stem}_check.json"

        if args.summary_plots:
            ok_paths = [Path(r["path"]) for r in results if r.get("status") in {"ok", "missing_branches"}]

            if args.auto_cs_type_map:
                c_type, s_type, type_counts = _infer_cs_types(
                    ok_paths,
                    tree_name=args.tree,
                    max_events_per_file=args.max_events_per_file,
                    explicit_map=None,
                )
            else:
                c_type, s_type, type_counts = _infer_cs_types(
                    [],
                    tree_name=args.tree,
                    max_events_per_file=args.max_events_per_file,
                    explicit_map=args.cs_type_map,
                )

            features = _collect_summary_features(
                ok_paths,
                tree_name=args.tree,
                max_events_per_file=args.max_events_per_file,
                c_type=c_type,
                s_type=s_type,
            )
            feature_summary = {k: _safe_stats(v) for k, v in features.items()}

            plot_out_dir = args.plot_out_dir
            if plot_out_dir is None:
                plot_out_dir = out_json.with_suffix("")
                plot_out_dir = plot_out_dir.parent / f"{plot_out_dir.name}_plots"
            elif multi:
                plot_out_dir = plot_out_dir / list_file.stem
            made = _make_summary_plots(features, plot_out_dir)

            payload["feature_summary"] = feature_summary
            payload["cs_proxy_info"] = {
                "c_type": c_type,
                "s_type": s_type,
                "type_counts": type_counts,
                "note": "If C_amp/S_amp are absent, script uses proxies from DigiCalorimeterHits.energy grouped by DigiCalorimeterHits.type.",
            }
            payload["summary_plot_dir"] = str(plot_out_dir)
            payload["summary_plots"] = made

        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run_summary = {
            "list_file": str(list_file),
            "sample_name": sample_name,
            "status_counts": counts,
            "num_files": len(results),
            "out_json": str(out_json),
            "summary_plots": payload.get("summary_plots", []),
        }
        all_run_summaries.append(run_summary)
        print(json.dumps(run_summary, indent=2))

    if multi:
        print(json.dumps({"processed_lists": len(all_run_summaries), "runs": all_run_summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
