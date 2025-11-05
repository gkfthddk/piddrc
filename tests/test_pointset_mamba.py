import importlib

import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def ensure_mamba_stub(monkeypatch):
    module = importlib.import_module("pid.models.pointset_mamba")

    class IdentityMamba(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return x

    monkeypatch.setattr(module, "Mamba", IdentityMamba, raising=False)
    monkeypatch.setattr(module, "Mamba2", IdentityMamba, raising=False)
    monkeypatch.setattr(
        module,
        "_AVAILABLE_MAMBA_BACKENDS",
        module._AVAILABLE_MAMBA_BACKENDS.copy(),
        raising=False,
    )
    module._AVAILABLE_MAMBA_BACKENDS["mamba"] = IdentityMamba
    module._AVAILABLE_MAMBA_BACKENDS["mamba2"] = IdentityMamba
    yield


def test_mamba_block_masks_padding():
    module = importlib.import_module("pid.models.pointset_mamba")
    block = module.MambaBlock(dim=4, dropout=0.0)
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)

    out = block(x, mask)

    assert torch.allclose(out[:, 2:], torch.zeros_like(out[:, 2:]))


def test_pointset_mamba_masks_projected_inputs():
    module = importlib.import_module("pid.models.pointset_mamba")
    model = module.PointSetMamba(
        in_channels=3,
        hidden_dim=4,
        depth=1,
        summary_dim=1,
        num_classes=2,
        head_hidden=(4,),
        dropout=0.0,
        use_summary=False,
        use_uncertainty=False,
    )

    points = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ])
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    batch = {"points": points, "mask": mask}

    captured = {}

    def _pre_hook(_, inputs):
        captured["block_input"] = inputs[0].detach().clone()

    hook_handle = model.blocks[0].register_forward_pre_hook(_pre_hook)
    try:
        model(batch)
    finally:
        hook_handle.remove()

    assert "block_input" in captured
    assert torch.allclose(captured["block_input"][:, 2:], torch.zeros_like(captured["block_input"][:, 2:]))


def test_pointset_mamba2_masks_projected_inputs():
    module = importlib.import_module("pid.models.pointset_mamba")
    model = module.PointSetMamba(
        in_channels=3,
        hidden_dim=4,
        depth=1,
        summary_dim=1,
        num_classes=2,
        head_hidden=(4,),
        dropout=0.0,
        use_summary=False,
        use_uncertainty=False,
        backend="mamba2",
    )

    points = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ])
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    batch = {"points": points, "mask": mask}

    captured = {}

    def _pre_hook(_, inputs):
        captured["block_input"] = inputs[0].detach().clone()

    hook_handle = model.blocks[0].register_forward_pre_hook(_pre_hook)
    try:
        model(batch)
    finally:
        hook_handle.remove()

    assert "block_input" in captured
    assert torch.allclose(captured["block_input"][:, 2:], torch.zeros_like(captured["block_input"][:, 2:]))


def test_pointset_mamba_backend_kwargs_forwarded(monkeypatch):
    module = importlib.import_module("pid.models.pointset_mamba")
    recorded = {}

    class RecordingIdentity(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            recorded["kwargs"] = kwargs

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return x

    monkeypatch.setattr(
        module,
        "_AVAILABLE_MAMBA_BACKENDS",
        module._AVAILABLE_MAMBA_BACKENDS.copy(),
        raising=False,
    )
    module._AVAILABLE_MAMBA_BACKENDS["mamba2"] = RecordingIdentity

    module.PointSetMamba(
        in_channels=3,
        hidden_dim=4,
        depth=1,
        summary_dim=1,
        num_classes=2,
        head_hidden=(4,),
        dropout=0.0,
        use_summary=False,
        use_uncertainty=False,
        backend="mamba2",
        backend_kwargs={"headdim": 4},
    )

    assert "kwargs" in recorded
    assert recorded["kwargs"]["headdim"] == 4


def test_pointset_mamba_rejects_unknown_backend():
    module = importlib.import_module("pid.models.pointset_mamba")
    with pytest.raises(ValueError):
        module.PointSetMamba(
            in_channels=3,
            hidden_dim=4,
            depth=1,
            summary_dim=1,
            num_classes=2,
            head_hidden=(4,),
            dropout=0.0,
            use_summary=False,
            use_uncertainty=False,
            backend="not-a-backend",  # type: ignore[arg-type]
        )
