import numpy as np
import torch
from typing import List, Tuple


def decode(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    score_threshold: float,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """per image, the surviving queries as normalized cxcywh with scores and class ids,
    highest score first. one class per query, so one query is never two detections"""
    scores, class_ids = logits.sigmoid().max(dim=-1)

    detections = []
    for image in range(logits.shape[0]):
        keep = scores[image] > score_threshold
        order = scores[image][keep].argsort(descending=True)
        detections.append(
            (
                pred_boxes[image][keep][order].numpy().astype(np.float32),
                scores[image][keep][order].numpy().astype(np.float32),
                class_ids[image][keep][order].numpy().astype(np.int64),
            )
        )
    return detections
