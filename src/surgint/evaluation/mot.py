import motmetrics as mm
import numpy as np

from surgint.model.boxes import iou_matrix

COUNTS = ("frames", "gt", "predictions", "fp", "fn", "id_switches", "idtp", "class_correct")

# motmetrics names for the counts above, in the same order after "frames"
METRICS = (
    "num_objects",
    "num_predictions",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "idtp",
)

# an object matched this frame is reported as one of these
MATCHED = ("MATCH", "SWITCH")


def mot_counts(frames, iou_threshold: float = 0.5) -> dict:
    """
    counts for one sequence.

    frames are (gt_boxes, gt_ids, gt_classes, boxes, track_ids, class_ids), in order.
    association is CLEAR MOT as implemented by motmetrics and ignores the class, so
    class_correct counts how many matched pairs also agree on the instrument.

    Raises:
        ValueError: a frame has fewer classes than boxes, or repeats an id.
    """
    accumulator = mm.MOTAccumulator(auto_id=True)
    classes = []

    for gt_boxes, gt_ids, gt_classes, boxes, track_ids, class_ids in frames:
        gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        gt_ids = np.asarray(gt_ids, dtype=np.int64).reshape(-1)
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
        class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)

        if len(gt_classes) != len(gt_boxes) or len(class_ids) != len(boxes):
            raise ValueError(
                f"every box needs a class: {len(gt_boxes)} gt boxes with {len(gt_classes)} "
                f"classes, {len(boxes)} predictions with {len(class_ids)} classes"
            )
        # an id naming two objects in one frame has no one-to-one matching to find
        if len(np.unique(gt_ids)) != len(gt_ids):
            raise ValueError(f"ground truth ids repeat within a frame: {gt_ids}")
        if len(np.unique(track_ids)) != len(track_ids):
            raise ValueError(f"track ids repeat within a frame: {track_ids}")

        accumulator.update(gt_ids.tolist(), track_ids.tolist(), _distances(gt_boxes, boxes, iou_threshold))
        classes.append((dict(zip(gt_ids.tolist(), gt_classes.tolist())),
                        dict(zip(track_ids.tolist(), class_ids.tolist()))))

    summary = mm.metrics.create().compute(accumulator, metrics=list(METRICS))
    counts = {name: int(summary[name].iloc[0]) for name in METRICS}

    fp, fn = counts["num_false_positives"], counts["num_misses"]
    if fp < 0 or fn < 0:
        raise ValueError(f"matching was not one-to-one: fp {fp}, fn {fn}")

    return {
        "frames": len(frames),
        "gt": counts["num_objects"],
        "predictions": counts["num_predictions"],
        "fp": fp,
        "fn": fn,
        "id_switches": counts["num_switches"],
        "idtp": counts["idtp"],
        "class_correct": _class_correct(accumulator, classes),
    }


def mot_evaluate(counts) -> dict:
    """
    MOTA, IDF1, class accuracy and id switches over one or more sequences.

    MOTA and IDF1 score geometry and identity. class_accuracy is the share of matched
    pairs that agree on the instrument, so a mislabeled track is visible separately.
    """
    if isinstance(counts, dict):
        counts = [counts]
    total = {key: sum(int(entry[key]) for entry in counts) for key in COUNTS}

    errors = total["fp"] + total["fn"] + total["id_switches"]
    mota = 1.0 - errors / total["gt"] if total["gt"] else 0.0

    id_fn = total["gt"] - total["idtp"]
    id_fp = total["predictions"] - total["idtp"]
    denominator = 2 * total["idtp"] + id_fp + id_fn
    idf1 = 2 * total["idtp"] / denominator if denominator else 0.0

    # every match consumes one ground truth box, so this is the matched pair count
    matched = total["gt"] - total["fn"]
    class_accuracy = total["class_correct"] / matched if matched else 0.0

    return {
        "MOTA": float(mota),
        "IDF1": float(idf1),
        "class_accuracy": float(class_accuracy),
        "id_switches": total["id_switches"],
        "sequences": len(counts),
        **{key: total[key] for key in ("frames", "gt", "predictions", "fp", "fn", "class_correct")},
    }


def _distances(gt_boxes: np.ndarray, boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    the pairwise matching cost motmetrics consumes, as 1 - iou.

    pairs below the threshold are nan, which motmetrics reads as unmatchable. the iou
    comes from this project's own matrix so the threshold means the same thing here as
    it does everywhere else.
    """
    if not len(gt_boxes) or not len(boxes):
        return np.empty((len(gt_boxes), len(boxes)))

    ious = iou_matrix(gt_boxes, boxes)
    cost = 1.0 - ious
    cost[ious < iou_threshold] = np.nan
    return cost


def _class_correct(accumulator: mm.MOTAccumulator, classes: list) -> int:
    """how many of the matched pairs name the same instrument on both sides"""
    events = accumulator.mot_events.reset_index()
    matched = events[events["Type"].isin(MATCHED)]

    correct = 0
    for frame, gt_id, track_id in zip(matched["FrameId"], matched["OId"], matched["HId"]):
        gt_classes, class_ids = classes[int(frame)]
        if gt_classes[int(gt_id)] == class_ids[int(track_id)]:
            correct += 1
    return correct
