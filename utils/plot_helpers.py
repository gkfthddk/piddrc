"""Shared helpers for plotting and analysis scripts."""

from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np


PAIR_SPECS: List[Tuple[str, str]] = [("pi0", "gamma"), ("pi+", "e-")]
PAIR_LABELS = {
    "pi0 vs gamma": r"$\pi^{0}$ vs $\gamma$",
    "pi+ vs e-": r"$\pi^{+}$ vs $e^{-}$",
}
DEFAULT_CONFIG_DIFF_EXCLUDE: Set[str] = {
    "checkpoint",
    "history_json",
    "metrics_json",
    "output_json",
    "config_json",
    "profile_dir",
    "name",
    "instance_name",
}


def pair_key(pos: str, neg: str) -> str:
    return f"{pos} vs {neg}"


def safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("+", "plus").replace("-", "minus")


def resolve_run_dir(raw: str, base_dir: Path) -> Path:
    direct = Path(raw)
    if direct.exists():
        return direct
    fallback = base_dir / raw
    if fallback.exists():
        return fallback
    return direct


def softmax(logits: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(logits), dtype=np.float64)
    arr = arr - arr.max()
    np.exp(arr, out=arr)
    arr /= arr.sum()
    return arr


def class_prefix(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("_", 1)[0]


def infer_class_to_label(
    records: Sequence[Mapping[str, object]],
    train_files: Sequence[str],
) -> Dict[str, int]:
    names = [class_prefix(f) for f in train_files]
    freq: Dict[int, Counter] = defaultdict(Counter)
    for rec in records:
        label = int(rec["label"])
        event_id = rec.get("event_id", [])
        if isinstance(event_id, list) and len(event_id) >= 1:
            freq[label].update([int(event_id[0])])

    out: Dict[str, int] = {}
    for label, cnt in freq.items():
        file_id, _ = cnt.most_common(1)[0]
        if 0 <= file_id < len(names):
            out[names[file_id]] = label
    return out


def is_simple_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def is_simple_value(value: object) -> bool:
    if is_simple_scalar(value):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_simple_scalar(item) for item in value)
    return False


def format_config_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def differing_config_keys(
    cfg_items: Sequence[Mapping[str, object]],
    exclude_keys: Iterable[str] | None = None,
) -> List[str]:
    exclude = set(DEFAULT_CONFIG_DIFF_EXCLUDE)
    if exclude_keys is not None:
        exclude.update(exclude_keys)

    keys = sorted(set().union(*[set(cfg.keys()) for cfg in cfg_items]))
    changed: List[str] = []
    for key in keys:
        if key in exclude:
            continue
        present_values = [cfg[key] for cfg in cfg_items if key in cfg]
        if not present_values:
            continue
        if not all(is_simple_value(v) for v in present_values):
            continue
        values = [cfg[key] if key in cfg else "__missing__" for cfg in cfg_items]
        if any(v != values[0] for v in values[1:]):
            changed.append(key)
    return changed


def wrap_label_text(text: str, width: int = 52) -> str:
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else text


def config_label(
    run_name: str,
    cfg: Mapping[str, object],
    diff_keys: Sequence[str],
    *,
    max_items: int = 4,
    width: int = 52,
) -> str:
    if not diff_keys:
        return run_name
    parts = [
        f"{key}={format_config_value(cfg[key]) if key in cfg else 'missing'}"
        for key in diff_keys[:max_items]
    ]
    if not parts:
        return run_name
    details = wrap_label_text(", ".join(parts), width=width)
    return f"{run_name}\n{details}"
