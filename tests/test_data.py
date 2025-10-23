"""Tests for data loading utilities."""

import numpy as np
import pytest

try:
    import h5py
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise pytest.UsageError(
        "The data loader tests require h5py and torch. Install via 'pip install -r requirements.txt'"
    ) from exc

from pid.data import DualReadoutEventDataset


@pytest.fixture()
def overflow_h5(tmp_path):
    file_path = tmp_path / "overflow.h5"
    with h5py.File(file_path, "w") as handle:
        # Build a tiny dataset with one obvious outlier.
        amplitudes = np.full((4, 8), 10.0, dtype=np.float32)
        amplitudes[3, :] = 1e9
        types = np.zeros((4, 8), dtype=np.float32)
        types[:, 4:] = 1.0  # mark last four hits as Cherenkov
        time = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        time = np.tile(time, (4, 1))
        zeros = np.zeros_like(amplitudes)

        handle.create_dataset("DRcalo3dHits.amplitude_sum", data=amplitudes)
        handle.create_dataset("DRcalo3dHits.type", data=types)
        handle.create_dataset("DRcalo3dHits.time", data=time)
        handle.create_dataset("DRcalo3dHits.time_end", data=time + 0.1)
        handle.create_dataset("DRcalo3dHits.position.x", data=zeros)
        handle.create_dataset("DRcalo3dHits.position.y", data=zeros)
        handle.create_dataset("DRcalo3dHits.position.z", data=zeros)

        labels = np.array(["11", "211", "11", "211"], dtype="S")
        energies = np.full(4, 42.0, dtype=np.float32)
        handle.create_dataset("GenParticles.PDG", data=labels)
        handle.create_dataset("E_gen", data=energies)

    return file_path


def test_amplitude_sum_masking(overflow_h5):
    dataset = DualReadoutEventDataset(
        [str(overflow_h5)],
        hit_features=(
            "DRcalo3dHits.amplitude_sum",
            "DRcalo3dHits.type",
            "DRcalo3dHits.time",
            "DRcalo3dHits.time_end",
            "DRcalo3dHits.position.x",
            "DRcalo3dHits.position.y",
            "DRcalo3dHits.position.z",
        ),
        label_key="GenParticles.PDG",
        energy_key="E_gen",
        max_points=None,
        amp_sum_clip_percentile=0.75,
        amp_sum_clip_multiplier=1.0,
        amp_sum_clip_sample_size=16,
    )

    # The dataset should drop the overflowing event entirely.
    assert len(dataset) == 3

    amp_index = dataset.feature_to_index["DRcalo3dHits.amplitude_sum"]
    seen_event_ids = []
    for i in range(len(dataset)):
        event = dataset[i]
        seen_event_ids.append(event.event_id[1])
        amp_total = float(torch.sum(event.points[:, amp_index]))
        assert amp_total > 0.0
        assert np.isfinite(amp_total)
        assert torch.all(torch.isfinite(event.summary))

    assert sorted(seen_event_ids) == [0, 1, 2]

