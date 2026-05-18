"""Shared helpers for plotting and analysis scripts."""

from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import matplotlib as mpl
import numpy as np


PAIR_SPECS: List[Tuple[str, str]] = [("pi0", "gamma"), ("pi+", "e-"), ("e-", "gamma")]
PAIR_LABELS = {
    "pi0 vs gamma": r"$\pi^{0}$ vs $\gamma$",
    "pi+ vs e-": r"$\pi^{+}$ vs $e^{-}$",
    "e- vs gamma": r"$e^{-}$ vs $\gamma$",
}

# Classes that share single-EM shower topology and should be grouped
# when computing pairwise discrimination scores against π⁰.
EM_GROUP_CLASSES: Tuple[str, ...] = ("gamma", "e-")


def pairwise_score(
    probs: np.ndarray,
    class_to_label: Mapping[str, int],
    pos: str,
    neg: str,
) -> np.ndarray:
    """Compute a pairwise discrimination score for ROC analysis.

    For π⁰ vs γ (or any pair where one side overlaps with another class
    in the EM group), the score is the conditional probability of the
    positive class given the relevant class pool:

        score = P(pos) / (P(pos) + Σ P(neg_group))

    where neg_group = EM_GROUP_CLASSES when neg ∈ EM_GROUP_CLASSES,
    otherwise neg_group = {neg}.  This prevents the γ/e⁻ confusion
    from leaking into the π⁰ vs γ ROC.

    For pairs where neither side is in the EM group (or when the pair
    is purely EM, e.g. γ vs e⁻), the raw softmax P(pos) is returned.
    """
    pos_idx = int(class_to_label[pos])
    neg_idx = int(class_to_label[neg])

    # Case 1: Grouped background (e.g. pi0 vs gamma where gamma group includes e-)
    if neg in EM_GROUP_CLASSES and pos not in EM_GROUP_CLASSES:
        neg_indices = [int(class_to_label[c]) for c in EM_GROUP_CLASSES if c in class_to_label]
        score_pos = probs[:, pos_idx]
        score_neg = np.sum(probs[:, neg_indices], axis=1)
    # Case 2: Grouped signal (e.g. gamma vs pi0 where gamma group includes e-)
    elif pos in EM_GROUP_CLASSES and neg not in EM_GROUP_CLASSES:
        pos_indices = [int(class_to_label[c]) for c in EM_GROUP_CLASSES if c in class_to_label]
        score_pos = np.sum(probs[:, pos_indices], axis=1)
        score_neg = probs[:, neg_idx]
    # Case 3: Pure pairwise (no grouping, e.g. e- vs gamma, or pi+ vs e-)
    else:
        score_pos = probs[:, pos_idx]
        score_neg = probs[:, neg_idx]

    denom = score_pos + score_neg
    denom = np.where(denom > 0, denom, 1.0)  # avoid division by zero
    return score_pos / denom
DEFAULT_CONFIG_DIFF_EXCLUDE: Set[str] = {
    "checkpoint",
    "history_json",
    "metrics_json",
    "output_json",
    "config_json",
    "profile_dir",
    "name",
    "instance_name",
    "progress_bar",
    "device",
    "log_file",
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
    source_files: Sequence[str],
) -> Dict[str, int]:
    names = [class_prefix(f) for f in source_files]
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


def _short_config_scalar(value: object) -> str:
    if not isinstance(value, str):
        return str(value)
    if "/" in value:
        value = Path(value).name
    if value.startswith("DRcalo3dHits."):
        value = value.replace("DRcalo3dHits.", "")
    if value.startswith("GenParticles."):
        value = value.replace("GenParticles.", "")
    if len(value) > 36:
        value = value[:33] + "..."
    return value


def format_config_value(value: object, *, max_list_items: int = 3) -> str:
    if isinstance(value, (list, tuple)):
        shown = [_short_config_scalar(item) for item in value[:max_list_items]]
        if len(value) > max_list_items:
            shown.append(f"...+{len(value) - max_list_items}")
        return "[" + ",".join(shown) + "]"
    return _short_config_scalar(value)


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
    shown_keys = list(diff_keys[:max_items])
    parts = [
        f"{key}={format_config_value(cfg[key]) if key in cfg else 'missing'}"
        for key in shown_keys
    ]
    if len(diff_keys) > len(shown_keys):
        parts.append(f"... +{len(diff_keys) - len(shown_keys)} more")
    if not parts:
        return run_name
    details = wrap_label_text(", ".join(parts), width=width)
    return f"{run_name}\n{details}"


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 15,
            "savefig.dpi": 300,
            "figure.dpi": 100,
            "axes.grid": False,
        }
    )
