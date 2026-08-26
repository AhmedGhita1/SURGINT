import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_predictions(image_id: int, result, label_to_category: dict[int, int]) -> list[dict]:
    """detections to coco records; xyxy back to xywh, label ids back to the original categories"""
    records = []
    for box, score, label in zip(result.boxes, result.scores, result.class_ids):
        x1, y1, x2, y2 = (float(value) for value in box)
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label_to_category[int(label)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return records


def evaluate(annotations: dict, predictions: list[dict]) -> dict:
    """coco mAP over the whole split; empty predictions score zero rather than raising"""
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = annotations
        truth.createIndex()
        names = {category["id"]: category["name"] for category in truth.loadCats(truth.getCatIds())}
        if not predictions:
            return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_class": dict.fromkeys(names.values(), 0.0)}

        evaluation = COCOeval(truth, truth.loadRes(predictions), "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()

    # precision is [iou, recall, category, area, max_detections]; area 0 and max 100 are the defaults
    precision = evaluation.eval["precision"][:, :, :, 0, 2]
    per_class = {}
    for index, category_id in enumerate(evaluation.params.catIds):
        scores = precision[:, :, index]
        scores = scores[scores > -1]
        per_class[names[category_id]] = float(scores.mean()) if scores.size else float("nan")

    return {
        "map": float(evaluation.stats[0]),
        "map50": float(evaluation.stats[1]),
        "map75": float(evaluation.stats[2]),
        "per_class": per_class,
    }
