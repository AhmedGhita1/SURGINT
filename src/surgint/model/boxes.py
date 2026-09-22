import numpy as np


def iou_matrix(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    others = np.asarray(others, dtype=float)

    top_left = np.maximum(boxes[:, None, :2], others[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], others[None, :, 2:])
    overlap = np.clip(bottom_right - top_left, 0, None)
    intersection = overlap[..., 0] * overlap[..., 1]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    other_areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = areas[:, None] + other_areas[None, :] - intersection

    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """greedy non-maximum suppression over all classes. returns the kept indices, highest score first."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=float).reshape(-1))

    keep = []
    while len(order):
        best = order[0]
        keep.append(int(best))
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[iou_matrix(boxes[best][None], boxes[rest])[0] < iou_threshold]

    return np.array(keep, dtype=np.int64)
