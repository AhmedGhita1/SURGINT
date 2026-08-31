import numpy as np
import torch
from PIL import Image

from typing import Tuple, List, Optional, Dict

PAD_COLOR = 114  # mid-gray
RESCALE_FACTOR = 1 / 255


def letterbox(
    frame: np.ndarray,
    width: int,
    height: int,
    PAD_COLOR: int = PAD_COLOR,
) -> Tuple[np.ndarray, float]:
    """aspect-preserving resize, padded to a fixed canvas"""
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = Image.fromarray(frame).resize(
        (round(source_width * scale), round(source_height * scale)), Image.BILINEAR
    )

    canvas = np.full((height, width, 3), PAD_COLOR, dtype=np.uint8)
    canvas[: resized.height, : resized.width] = np.asarray(resized)
    return canvas, scale


def to_pixel_values(canvas_list, rescale_factor: float = RESCALE_FACTOR) -> torch.Tensor:
    """canvas_list to a (B, 3, H, W) float batch"""
    batch = np.stack([np.asarray(canvas) for canvas in canvas_list])
    return torch.from_numpy(batch).permute(0, 3, 1, 2).float() * rescale_factor


def letterbox_boxes(boxes, scale: float) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float32) * scale


def to_normalized_cxcywh(boxes, width: int, height: int) -> np.ndarray:
    """canvas xyxy to the normalized cxcywh the detection loss expects"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    x1, y1, x2, y2 = boxes.T
    return np.stack(
        [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height],
        axis=-1,
    )


def to_canvas_xyxy(boxes, width: int, height: int) -> np.ndarray:
    """normalized cxcywh back to xyxy in canvas pixels"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, w, h = boxes.T
    return np.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        axis=-1,
    )


def unletterbox_boxes(boxes, scale: float) -> np.ndarray:
    """canvas pixels to camera pixels"""
    return np.asarray(boxes, dtype=np.float32) / scale


class Transform:
    def __init__(
            self, input_size: List[int],
            pad_color: int = PAD_COLOR,
            rescale_factor: float = RESCALE_FACTOR,
            ):

        self.width, self.height = input_size
        self.pad_color = pad_color
        self.rescale_factor = rescale_factor

    def __call__(
            self, frame: np.ndarray,
            boxes: Optional[np.ndarray] = None,
            class_ids: Optional[np.ndarray] = None,
            ) -> Dict:

        canvas, scale = letterbox(frame, self.width, self.height, self.pad_color)
        sample = {
            "pixel_values": to_pixel_values([canvas], self.rescale_factor)[0],
            "scale": scale,
            "frame_size": frame.shape[:2],
        }
        if boxes is None:
            return sample

        # scale the boxes onto the canvas and clip them to it
        boxes = letterbox_boxes(boxes, scale)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, self.width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, self.height)

        degenerate = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
        if degenerate.any():
            raise ValueError(f"{degenerate.sum()} boxes have zero area after letterbox")

        sample["boxes"] = to_normalized_cxcywh(boxes, self.width, self.height)
        sample["class_ids"] = class_ids
        return sample


    def postprocess(
            self,
            boxes: np.ndarray,
            scale: float,
            frame_size: Tuple[int, int]) -> np.ndarray:
        # the exact inverse of __call__: normalized cxcywh -> canvas xyxy -> frame pixels
        boxes = to_canvas_xyxy(boxes, self.width, self.height)
        boxes = unletterbox_boxes(boxes, scale)

        rows, columns = frame_size
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, columns)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, rows)
        return boxes
