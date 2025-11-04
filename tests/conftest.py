"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

HIT_FEATURE_NAMES: Iterable[str] = (
    "DRcalo3dHits.amplitude_sum",
    "DRcalo3dHits.type",
    "DRcalo3dHits.time",
    "DRcalo3dHits.time_end",
    "DRcalo3dHits.position.x",
    "DRcalo3dHits.position.y",
    "DRcalo3dHits.position.z",
)


@pytest.fixture()
def stats_file(tmp_path: Path) -> Path:
    """Write a lightweight statistics file compatible with the dataset loader."""

    stats = {
        name: {"min": -1.0, "max": 1.0} for name in HIT_FEATURE_NAMES
    }
    # Provide generous amplitude limits so synthetic tests keep their events.
    stats["DRcalo3dHits.amplitude_sum"] = {"min": 0.0, "max": 100.0}
    # Include the amplitude summary channel consumed for overflow filtering.
    stats["S_amp"] = {"min": 0.0, "max": 100.0}

    output = tmp_path / "stats.json"
    output.write_text(json.dumps(stats))
    return output
