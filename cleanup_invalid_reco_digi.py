#!/usr/bin/env python
"""Delete unreadable/invalid reco files and corresponding digi ROOT files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete unreadable/invalid reco files and their corresponding digi ROOT files."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("h5s/validation_reports"),
        help="Directory containing *_unreadable_reco.txt and *_readable_but_invalid_reco.txt.",
    )
    parser.add_argument(
        "--digi-base",
        type=Path,
        default=Path("/hdfs/ml/dualreadout/v001"),
        help="Base directory for digi ROOT samples.",
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Also delete files listed in *_readable_mixed_reco.txt.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Restrict cleanup to one or more sample names (repeatable).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of reco files to process (<=0 means all).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, script runs in dry-run mode.",
    )
    return parser.parse_args()


def _sample_from_report_name(report_path: Path) -> str:
    name = report_path.name
    suffixes = [
        "_unreadable_reco.txt",
        "_readable_but_invalid_reco.txt",
        "_readable_mixed_reco.txt",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unexpected report filename: {report_path}")


def _iter_reports(reports_dir: Path, include_mixed: bool) -> list[Path]:
    patterns = ["*_unreadable_reco.txt", "*_readable_but_invalid_reco.txt"]
    if include_mixed:
        patterns.append("*_readable_mixed_reco.txt")

    reports: list[Path] = []
    for pat in patterns:
        reports.extend(sorted(reports_dir.glob(pat)))
    return reports


def _parse_reco_paths(report_paths: Iterable[Path], selected_samples: set[str] | None) -> list[Path]:
    reco_set: set[Path] = set()
    for report in report_paths:
        sample = _sample_from_report_name(report)
        if selected_samples is not None and sample not in selected_samples:
            continue
        for raw in report.read_text(encoding="utf-8").splitlines():
            path_s = raw.strip()
            if not path_s:
                continue
            reco_set.add(Path(path_s))
    return sorted(reco_set)


def _to_digi_path(reco_path: Path, digi_base: Path) -> Path:
    sample = reco_path.parent.name
    return digi_base / sample / f"digi_{reco_path.stem}.root"


def _delete_path(path: Path, execute: bool) -> str:
    if not path.exists():
        return "missing"
    if not execute:
        return "would_delete"
    try:
        path.unlink()
        return "deleted"
    except Exception:
        return "error"


def main() -> int:
    args = parse_args()

    if not args.reports_dir.is_dir():
        raise FileNotFoundError(f"Reports directory not found: {args.reports_dir}")

    reports = _iter_reports(args.reports_dir, include_mixed=args.include_mixed)
    if not reports:
        raise RuntimeError(f"No matching report files in {args.reports_dir}")

    selected_samples = set(args.sample) if args.sample else None
    reco_paths = _parse_reco_paths(reports, selected_samples)
    if args.max_files > 0:
        reco_paths = reco_paths[: args.max_files]

    if not reco_paths:
        raise RuntimeError("No reco files selected for cleanup.")

    digi_paths = [_to_digi_path(p, args.digi_base) for p in reco_paths]

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Reports scanned: {len(reports)}")
    print(f"Reco targets: {len(reco_paths)}")

    reco_stats = {"deleted": 0, "would_delete": 0, "missing": 0, "error": 0}
    digi_stats = {"deleted": 0, "would_delete": 0, "missing": 0, "error": 0}

    for reco_path, digi_path in tqdm(list(zip(reco_paths, digi_paths)), desc="Cleanup", unit="file"):
        reco_status = _delete_path(reco_path, args.execute)
        #digi_status = _delete_path(digi_path, args.execute)
        reco_stats[reco_status] += 1
        #digi_stats[digi_status] += 1

    print("Reco results:", reco_stats)
    print("Digi results:", digi_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
