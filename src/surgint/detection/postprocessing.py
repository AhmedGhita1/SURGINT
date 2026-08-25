import numpy as np
import torch


def decode(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    width: int,
    height: int,
    score_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """per image, xyxy boxes in canvas pixels with scores and class ids, highest score first"""
    scores, class_ids = logits.sigmoid().max(dim=-1)

    cx, cy, w, h = pred_boxes.unbind(-1)
    boxes = torch.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        dim=-1,
    )

    detections = []
    for image in range(logits.shape[0]):
        keep = scores[image] > score_threshold
        order = scores[image][keep].argsort(descending=True)
        detections.append(
            (
                boxes[image][keep][order].numpy().astype(np.float32),
                scores[image][keep][order].numpy().astype(np.float32),
                class_ids[image][keep][order].numpy().astype(np.int64),
            )
        )
    return detections
