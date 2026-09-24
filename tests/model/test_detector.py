"""Integration tests for loading, running, and persisting the detector."""

import tempfile
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from surgint.model.detector import Detector

# these load a real checkpoint from the hub
pytestmark = pytest.mark.integration

CHECKPOINT = "PekingU/rtdetr_r18vd_coco_o365"
ID2LABEL = {0: "scalpel", 1: "scissors", 2: "forceps"}
HEIGHT, WIDTH = 256, 320                       # small canvas; only shapes are under test
QUERIES = 300


@lru_cache(maxsize=1)
def build_detector() -> Detector:
    return Detector.from_pretrained(CHECKPOINT, ID2LABEL)


def pixel_values(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, 3, HEIGHT, WIDTH)


def test_construction():
    """loading a pretrained checkpoint against loading a surgint one"""

    detector = build_detector()

    # the classifier is resized to the taxonomy, not to the checkpoint's 80
    assert detector.id2label == ID2LABEL, f"got {detector.id2label}"
    assert detector.model.config.num_labels == len(ID2LABEL), "num_labels follows id2label"
    for name, parameter in detector.named_parameters():
        if "decoder.class_embed" in name and name.endswith("weight"):
            assert parameter.shape[0] == len(ID2LABEL), f"{name} kept the old width"

    # the denoising head is N+1; it needs a slot for the no-object class
    denoising = dict(detector.named_parameters())["model.model.denoising_class_embed.weight"]
    assert denoising.shape[0] == len(ID2LABEL) + 1, f"got {denoising.shape[0]}"

    # the box head is not resized; geometry is task-agnostic
    box_heads = [p for n, p in detector.named_parameters() if "bbox_embed" in n]
    assert box_heads, "box regression head is missing"

    # nn.Module, so the trainer needs nothing re-exposed
    assert isinstance(detector, torch.nn.Module)
    assert sum(p.numel() for p in detector.parameters()) > 0, "parameters() is empty"
    assert detector.device == next(detector.parameters()).device, "device is read from parameters"

    # a pretrained checkpoint carries no surgint meta
    assert detector.meta is None, "meta is written on save, not on load from pretrained"


def test_persistence():
    """a checkpoint is the weights plus meta.yaml"""

    detector = build_detector()
    geometry = {
        "input_size": [WIDTH, HEIGHT],
        "pad_color": 114,
        "rescale_factor": 1 / 255,
        "source": CHECKPOINT,
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best"
        detector.save_checkpoint(path, geometry)

        written = sorted(f.name for f in path.iterdir())
        assert "meta.yaml" in written, f"got {written}"
        assert "model.safetensors" in written, f"got {written}"

        reloaded = Detector.from_checkpoint(path)

        # the labels come from the model config, so they cannot disagree with the head
        assert reloaded.meta["labels"] == [ID2LABEL[i] for i in sorted(ID2LABEL)]
        assert reloaded.model.config.num_labels == len(ID2LABEL), "the head was resized on reload"
        assert {int(k): v for k, v in reloaded.id2label.items()} == ID2LABEL, "labels were lost"

        # the geometry survives, so inference letterboxes the way training did
        for key, value in geometry.items():
            assert reloaded.meta[key] == value, f"{key} changed to {reloaded.meta[key]}"

        # weights alone are not a checkpoint
        bare = Path(directory) / "bare"
        detector.model.save_pretrained(bare)
        try:
            Detector.from_checkpoint(bare)
            assert False, "should raise for a checkpoint without meta.yaml"
        except FileNotFoundError:
            pass


def test_predict():
    """eval and inference"""

    detector = build_detector()
    detector.train()

    logits, pred_boxes = detector.predict(pixel_values())

    assert logits.shape == (1, QUERIES, len(ID2LABEL)), f"got {tuple(logits.shape)}"
    assert pred_boxes.shape == (1, QUERIES, 4), f"got {tuple(pred_boxes.shape)}"

    # plain tensors, not the huggingface output object
    assert isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor)
    assert logits.device.type == "cpu" and pred_boxes.device.type == "cpu", "predict returns cpu"
    assert not logits.requires_grad, "predict runs without grad"

    # eval mode is set here, so train and inference compute the same function
    assert not detector.training, "predict must leave the model in eval mode"

    # boxes come back as the normalized cxcywh the model emits
    assert pred_boxes.min() >= 0.0 and pred_boxes.max() <= 1.0, "pred_boxes are normalized"

    # denoising queries exist only in train mode
    assert logits.shape[1] == QUERIES, "predict must return the 300 real queries only"

    # a batch keeps one entry per image
    logits, pred_boxes = detector.predict(pixel_values(2))
    assert logits.shape[0] == 2 and pred_boxes.shape[0] == 2, "batching collapsed"


def test_forward():
    """training"""

    detector = build_detector()
    detector.train()

    labels = [
        {
            "class_labels": torch.tensor([1], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        }
    ]
    outputs = detector(pixel_values(), labels)

    assert outputs.loss is not None, "labels must produce a loss"
    assert torch.isfinite(outputs.loss), f"loss is {outputs.loss.item()}"
    # training adds denoising queries alongside the 300 real ones
    assert outputs.logits.shape[0] == 1, "batch dimension changed"
    assert outputs.logits.shape[1] > QUERIES, "denoising queries are missing in train mode"
    assert outputs.logits.shape[2] == len(ID2LABEL), "wrong number of classes"

    # gradients reach the pretrained weights, not only the new head
    detector.zero_grad()
    outputs.loss.backward()
    backbone = [p for n, p in detector.named_parameters() if "backbone" in n and p.grad is not None]
    assert backbone, "no gradient reached the backbone"

    # without labels there is no loss
    assert detector(pixel_values()).loss is None, "a forward without labels must not report a loss"

