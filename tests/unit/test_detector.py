"""Unit tests for device-specific detector inference behavior."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from surgint.model.detector import Detector


class StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values=None, labels=None):
        return SimpleNamespace(
            logits=torch.zeros(1, 1, 1),
            pred_boxes=torch.zeros(1, 1, 4),
        )


class DeviceDetector(Detector):
    def __init__(self, device_type: str) -> None:
        super().__init__(StubModel())
        self.device_type = device_type

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)


@pytest.mark.parametrize(("device_type", "mixed_precision"), [("cpu", False), ("cuda", True)])
def test_predict_uses_mixed_precision_only_on_cuda(monkeypatch, device_type, mixed_precision):
    calls = []

    @contextmanager
    def fake_autocast(*, device_type, dtype, enabled):
        calls.append((device_type, dtype, enabled))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)

    logits, pred_boxes = DeviceDetector(device_type).predict(torch.zeros(1, 3, 4, 4))

    assert calls == [("cuda", torch.float16, mixed_precision)]
    assert logits.device.type == "cpu"
    assert pred_boxes.device.type == "cpu"
