import numpy as np

from surgint.model.boxes import iou_matrix


def count_matches(predicted: np.ndarray, ground_truth: np.ndarray, iou_threshold: float = 0.5) -> int:
    """greedy match, highest scoring prediction first; each ground-truth box is claimed once"""
    if len(predicted) == 0 or len(ground_truth) == 0:
        return 0

    ious = iou_matrix(predicted, ground_truth)
    claimed = np.zeros(len(ground_truth), dtype=bool)
    matches = 0
    for row in ious:
        candidates = np.where(claimed, -1.0, row)
        best = candidates.argmax()
        if candidates[best] >= iou_threshold:
            claimed[best] = True
            matches += 1
            if claimed.all():
                break
    return matches
