import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_predictions(image_id, boxes, scores, class_ids, mappings) -> list[dict]:
    """xyxy in frame pixels to coco records, written back in the file's category ids"""
    records = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = (float(value) for value in box)
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(mappings[int(class_id)][0]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return records


def coco_evaluate(annotations: dict, predictions: list[dict]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = annotations
        truth.createIndex()
        names = {category["id"]: category["name"] for category in truth.loadCats(truth.getCatIds())}
        if not predictions:
            return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "per_class": dict.fromkeys(names.values(), 0.0)}

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
        "mAP50_95": float(evaluation.stats[0]),
        "mAP50": float(evaluation.stats[1]),
        "mAP75": float(evaluation.stats[2]),
        "per_class": per_class,
    }
