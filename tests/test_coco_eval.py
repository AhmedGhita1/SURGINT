"""
COCO evaluation tests
=====================

tests the scoring of predictions against the annotations json.

coverage:
- records:   xywh conversion, original category ids, image id
- mAP:       ground truth as predictions, a shifted box, no predictions
- per class: one entry per category
"""

import numpy as np

from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions

CATEGORIES = [{"id": 1, "name": "scalpel"}, {"id": 2, "name": "scissors"}]
MAPPINGS = {0: (1, "scalpel"), 1: (2, "scissors")}
BOXES = [[100.0, 100.0, 50.0, 80.0], [400.0, 300.0, 60.0, 40.0]]     # xywh


def build_annotations() -> dict:
    return {
        "images": [{"id": 7, "file_name": "a.png", "width": 1280, "height": 720}],
        "annotations": [
            {"id": i, "image_id": 7, "category_id": i, "bbox": box,
             "area": box[2] * box[3], "iscrowd": 0}
            for i, box in enumerate(BOXES, start=1)
        ],
        "categories": CATEGORIES,
    }


def predict(boxes, class_ids, mappings=None):
    """xywh to the xyxy in frame pixels that postprocess produces"""
    xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32)
    scores = np.ones(len(boxes), dtype=np.float32)
    return coco_predictions(7, xyxy, scores, np.array(class_ids, dtype=np.int64),
                            mappings if mappings is not None else MAPPINGS)


def test_unit_coco_predictions():
    """detections to coco records"""

    records = predict([BOXES[0]], [0], {0: (5, "scalpel")})
    assert len(records) == 1, f"got {len(records)}"

    # xyxy goes back to xywh, since that is what the json holds
    assert records[0]["bbox"] == [100.0, 100.0, 50.0, 80.0], f"got {records[0]['bbox']}"

    # written back in the file's category id, not the contiguous class id
    assert records[0]["category_id"] == 5, f"got {records[0]['category_id']}"
    assert records[0]["image_id"] == 7, "the join key back to the json"
    assert records[0]["score"] == 1.0

    # no detections produce no records
    assert predict([], []) == [], "an empty frame must produce no records"


def test_unit_coco_evaluate():
    """predictions scored against the annotations json"""

    annotations = build_annotations()

    # feeding the ground truth back scores a perfect mAP
    metrics = coco_evaluate(annotations, predict(BOXES, [0, 1]))
    assert np.allclose(metrics["mAP50_95"], 1.0), f"got {metrics['mAP50_95']}"
    assert np.allclose(metrics["mAP50"], 1.0), f"got {metrics['mAP50']}"

    # one entry per category, named
    assert set(metrics["per_class"]) == {"scalpel", "scissors"}, f"got {sorted(metrics['per_class'])}"

    # a box shifted by 30px still overlaps, so it scores between zero and perfect
    shifted = [[x + 30, y + 30, w, h] for x, y, w, h in BOXES]
    metrics = coco_evaluate(annotations, predict(shifted, [0, 1]))
    assert 0.0 <= metrics["mAP50_95"] < 1.0, f"got {metrics['mAP50_95']}"

    # nothing detected scores zero without raising
    metrics = coco_evaluate(annotations, [])
    assert metrics["mAP50_95"] == 0.0, f"got {metrics['mAP50_95']}"
    assert metrics["per_class"] == {"scalpel": 0.0, "scissors": 0.0}


if __name__ == "__main__":
    test_unit_coco_predictions()
    test_unit_coco_evaluate()
    print("\nall passed")
