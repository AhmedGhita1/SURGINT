# unit tests:
#   ground truth as predictions scores 1.0,
#   coco record format,
#   original category ids,
#   empty predictions,
#   a shifted box scores below perfect

from types import SimpleNamespace

import numpy as np
import pytest

from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions

CATEGORIES = [{"id": 1, "name": "scalpel"}, {"id": 2, "name": "scissors"}]
BOXES = [[100.0, 100.0, 50.0, 80.0], [400.0, 300.0, 60.0, 40.0]]


def build_annotations() -> dict:
    return {
        "images": [{"id": 7, "file_name": "a.png", "width": 1280, "height": 720}],
        "annotations": [
            {"id": i, "image_id": 7, "category_id": i, "bbox": box, "area": box[2] * box[3], "iscrowd": 0}
            for i, box in enumerate(BOXES, start=1)
        ],
        "categories": CATEGORIES,
    }


def result_from(boxes: list[list[float]], labels: list[int]):
    """the fields coco_evaluate() reads off a DetectionResult"""
    xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]
    return SimpleNamespace(
        boxes=np.array(xyxy, dtype=np.float32),
        scores=np.ones(len(boxes), dtype=np.float32),
        class_ids=np.array(labels, dtype=np.int64),
    )


def test_ground_truth_as_predictions_scores_perfect():
    annotations = build_annotations()
    predictions = coco_predictions(7, result_from(BOXES, [0, 1]), {0: 1, 1: 2})

    metrics = coco_evaluate(annotations, predictions)

    print(f"map {metrics['mAP50_95']:.3f}  map50 {metrics['mAP50']:.3f}  {metrics['per_class']}")
    assert metrics["mAP50_95"] == pytest.approx(1.0)
    assert metrics["mAP50"] == pytest.approx(1.0)
    assert set(metrics["per_class"]) == {"scalpel", "scissors"}


def test_records_use_xywh_and_original_category_ids():
    records = coco_predictions(7, result_from([BOXES[0]], [0]), {0: 5})

    print(records[0])
    assert records[0]["bbox"] == [100.0, 100.0, 50.0, 80.0]
    assert records[0]["category_id"] == 5
    assert records[0]["image_id"] == 7


def test_empty_predictions_score_zero():
    metrics = coco_evaluate(build_annotations(), [])

    print(f"map {metrics['mAP50_95']}")
    assert metrics["mAP50_95"] == 0.0
    assert metrics["per_class"] == {"scalpel": 0.0, "scissors": 0.0}


def test_shifted_boxes_score_below_perfect():
    shifted = [[x + 30, y + 30, w, h] for x, y, w, h in BOXES]
    predictions = coco_predictions(7, result_from(shifted, [0, 1]), {0: 1, 1: 2})

    metrics = coco_evaluate(build_annotations(), predictions)

    print(f"shifted by 30px: map {metrics['mAP50_95']:.3f}")
    assert 0.0 <= metrics["mAP50_95"] < 1.0


def test_unit_detection_eval():
    for test in [
        test_ground_truth_as_predictions_scores_perfect,
        test_records_use_xywh_and_original_category_ids,
        test_empty_predictions_score_zero,
        test_shifted_boxes_score_below_perfect,
    ]:
        print(f"\n{test.__name__}")
        test()


if __name__ == "__main__":
    test_unit_detection_eval()
    print("\nall passed")
