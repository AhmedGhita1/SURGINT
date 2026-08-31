"""
Evaluation tests
================

tests the loop that turns model output into a coco score.

a stub detector stands in for the weights, so the predictions are controlled and
the whole chain is checked end to end: decode, postprocess, the coco records, and
the score.

coverage:
- perfect:   predicting the ground truth scores 1.0
- shifted:   a displaced box scores below perfect
- keys:      the configured metrics and per class
- threshold: every query survives at 0.0
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from surgint.dataset.coco import SurgintDataset, collate
from surgint.model.transform import Transform
from surgint.engine.evaluator import evaluate

INPUT_SIZE = [320, 192]
FRAME_HEIGHT, FRAME_WIDTH = 180, 320
BOX = [40, 60, 60, 50]                      # xywh, the same box in every frame
CATEGORIES = [{"id": 0, "name": "scalpel"}, {"id": 1, "name": "scissors"}]


class StubDetector:
    """returns one query per image, at a fixed box and class"""

    def __init__(self, boxes: np.ndarray, class_id: int, score: float = 0.9):
        self.boxes = boxes
        self.class_id = class_id
        self.score = score
        self.calls = 0

    def predict(self, pixel_values):
        self.calls += 1
        batch = len(pixel_values)

        logits = torch.full((batch, len(self.boxes), len(CATEGORIES)), -10.0)
        logits[:, :, self.class_id] = torch.logit(torch.tensor(self.score))

        pred_boxes = torch.tensor(self.boxes, dtype=torch.float32).expand(batch, -1, -1)
        return logits, pred_boxes


def build_dataset(transform: Transform, images: int = 2) -> SurgintDataset:
    root = Path(tempfile.mkdtemp())
    (root / "val" / "images").mkdir(parents=True)
    (root / "annotations").mkdir()

    for index in range(images):
        Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT)).save(root / "val" / "images" / f"{index}.png")

    (root / "annotations" / "instances_val.json").write_text(json.dumps({
        "images": [
            {"id": i, "file_name": f"images/{i}.png",
             "width": FRAME_WIDTH, "height": FRAME_HEIGHT} for i in range(images)
        ],
        "annotations": [
            {"id": i + 1, "image_id": i, "category_id": 0, "bbox": BOX,
             "area": BOX[2] * BOX[3], "iscrowd": 0} for i in range(images)
        ],
        "categories": CATEGORIES,
    }))
    return SurgintDataset(root, "val", "detection-only", transform)


def truth_as_prediction(transform: Transform) -> np.ndarray:
    """the ground truth box in the normalized cxcywh the model would emit"""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    xyxy = np.array([[BOX[0], BOX[1], BOX[0] + BOX[2], BOX[1] + BOX[3]]], dtype=np.float32)
    return transform(frame, xyxy, np.array([0], dtype=np.int64))["boxes"]


def test_unit_evaluate():
    """the score of a controlled prediction"""

    transform = Transform(INPUT_SIZE)
    dataset = build_dataset(transform)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    predicted = truth_as_prediction(transform)

    # predicting the ground truth back scores a perfect mAP, which exercises the
    # whole chain: decode, postprocess to frame pixels, and the coco records
    metrics = evaluate(StubDetector(predicted, class_id=0), loader, transform)
    assert np.allclose(metrics["mAP50_95"], 1.0), f"got {metrics['mAP50_95']}"
    assert np.allclose(metrics["mAP50"], 1.0), f"got {metrics['mAP50']}"

    # every metric coco_evaluate reports, per class included
    assert set(metrics) == {"mAP50_95", "mAP50", "mAP75", "per_class"}, f"got {sorted(metrics)}"
    assert set(metrics["per_class"]) == {"scalpel", "scissors"}
    assert np.allclose(metrics["per_class"]["scalpel"], 1.0), "the predicted class scored below 1.0"

    # a box shifted by a quarter of the canvas still overlaps, so it scores lower
    shifted = predicted.copy()
    shifted[:, 0] += 0.25
    metrics = evaluate(StubDetector(shifted, class_id=0), loader, transform)
    assert 0.0 <= metrics["mAP50_95"] < 1.0, f"got {metrics['mAP50_95']}"

    # the right box under the wrong class scores nothing
    metrics = evaluate(StubDetector(predicted, class_id=1), loader, transform)
    assert metrics["mAP50_95"] == 0.0, f"got {metrics['mAP50_95']}"


def test_unit_threshold():
    """every query reaches the metric"""

    transform = Transform(INPUT_SIZE)
    dataset = build_dataset(transform)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    predicted = truth_as_prediction(transform)

    # a very low scoring prediction is still scored; mAP integrates the whole
    # recall range, so filtering by score first would truncate the curve
    metrics = evaluate(StubDetector(predicted, class_id=0, score=0.001), loader, transform)
    assert np.allclose(metrics["mAP50_95"], 1.0), f"got {metrics['mAP50_95']}"

    # one forward pass per batch, not per image
    detector = StubDetector(predicted, class_id=0)
    evaluate(detector, loader, transform)
    assert detector.calls == len(loader), f"got {detector.calls} calls for {len(loader)} batches"


if __name__ == "__main__":
    test_unit_evaluate()
    test_unit_threshold()
    print("\nall passed")
