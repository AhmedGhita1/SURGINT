from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from surgint.model.boxes import iou_matrix

COUNTS = ("frames", "gt", "predictions", "fp", "fn", "id_switches", "idtp")


def mot_counts(frames, iou_threshold: float = 0.5) -> dict:
    """counts for one sequence. frames are (gt_boxes, gt_ids, boxes, track_ids), in order"""
    last_match: dict[int, int] = {}
    pairs: Counter = Counter()
    gt_total = prediction_total = 0
    fp = fn = id_switches = 0

    for gt_boxes, gt_ids, boxes, track_ids in frames:
        gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        gt_ids = np.asarray(gt_ids, dtype=np.int64).reshape(-1)
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)

        gt_total += len(gt_boxes)
        prediction_total += len(boxes)

        matches = _match(gt_boxes, gt_ids, boxes, track_ids, iou_threshold, last_match)

        for gt_index, prediction_index in matches:
            gt_id, track_id = int(gt_ids[gt_index]), int(track_ids[prediction_index])
            if last_match.get(gt_id, track_id) != track_id:
                id_switches += 1
            last_match[gt_id] = track_id
            pairs[(gt_id, track_id)] += 1

        fp += len(boxes) - len(matches)
        fn += len(gt_boxes) - len(matches)

    return {
        "frames": len(frames),
        "gt": gt_total,
        "predictions": prediction_total,
        "fp": fp,
        "fn": fn,
        "id_switches": id_switches,
        "idtp": _id_true_positives(pairs),
    }


def mot_evaluate(counts) -> dict:
    """MOTA, IDF1 and id switches over one or more sequences"""
    if isinstance(counts, dict):
        counts = [counts]
    total = {key: sum(int(entry[key]) for entry in counts) for key in COUNTS}

    errors = total["fp"] + total["fn"] + total["id_switches"]
    mota = 1.0 - errors / total["gt"] if total["gt"] else 0.0

    id_fn = total["gt"] - total["idtp"]
    id_fp = total["predictions"] - total["idtp"]
    denominator = 2 * total["idtp"] + id_fp + id_fn
    idf1 = 2 * total["idtp"] / denominator if denominator else 0.0

    return {
        "MOTA": float(mota),
        "IDF1": float(idf1),
        "id_switches": total["id_switches"],
        "sequences": len(counts),
        **{key: total[key] for key in ("frames", "gt", "predictions", "fp", "fn")},
    }


def _match(gt_boxes, gt_ids, boxes, track_ids, iou_threshold, last_match):
    """one frame. returns (gt index, prediction index) pairs above the iou threshold"""
    if not len(gt_boxes) or not len(boxes):
        return []

    ious = iou_matrix(gt_boxes, boxes)
    matches = []
    taken_gt, taken_prediction = set(), set()

    # a pairing that held last frame is kept while it still clears the threshold. without
    # this, the assignment is free to swap two overlapping objects and score a switch.
    for gt_index, gt_id in enumerate(gt_ids):
        previous = last_match.get(int(gt_id))
        if previous is None:
            continue
        for prediction_index, track_id in enumerate(track_ids):
            if int(track_id) == previous and ious[gt_index, prediction_index] >= iou_threshold:
                matches.append((gt_index, prediction_index))
                taken_gt.add(gt_index)
                taken_prediction.add(prediction_index)
                break

    free_gt = [index for index in range(len(gt_ids)) if index not in taken_gt]
    free_prediction = [index for index in range(len(track_ids)) if index not in taken_prediction]
    if not free_gt or not free_prediction:
        return matches

    cost = 1.0 - ious[np.ix_(free_gt, free_prediction)]
    for row, column in zip(*linear_sum_assignment(cost)):
        if cost[row, column] <= 1.0 - iou_threshold:
            matches.append((free_gt[row], free_prediction[column]))
    return matches


def _id_true_positives(pairs: Counter) -> int:
    """frames covered by the best one-to-one assignment of gt tracks to predicted tracks"""
    if not pairs:
        return 0

    gt_ids = sorted({gt_id for gt_id, _ in pairs})
    track_ids = sorted({track_id for _, track_id in pairs})

    overlap = np.zeros((len(gt_ids), len(track_ids)))
    for (gt_id, track_id), frames in pairs.items():
        overlap[gt_ids.index(gt_id), track_ids.index(track_id)] = frames

    rows, columns = linear_sum_assignment(-overlap)
    return int(overlap[rows, columns].sum())
