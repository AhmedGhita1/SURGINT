"""Frame to model input. Shared by the training dataset and the inference pipeline,
so both paths produce identical pixels. See docs/decisions/0001-frame-type-and-resize.md:
frames are RGB uint8 (H, W, 3), and PIL bilinear is the only resize in the codebase."""

import numpy as np
import torch
from PIL import Image

PAD_VALUE = 114  # mid-gray


def letterbox(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    """aspect-preserving resize, padded to a fixed canvas"""
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = Image.fromarray(frame).resize(
        (round(source_width * scale), round(source_height * scale)), Image.BILINEAR
    )

    canvas = np.full((height, width, 3), PAD_VALUE, dtype=np.uint8)
    canvas[: resized.height, : resized.width] = np.asarray(resized)
    return canvas, scale


def to_pixel_values(canvases, rescale_factor: float = 1 / 255) -> torch.Tensor:
    """canvases to a (B, 3, H, W) float batch"""
    batch = np.stack([np.asarray(canvas) for canvas in canvases])
    return torch.from_numpy(batch).permute(0, 3, 1, 2).float() * rescale_factor


def letterbox_boxes(boxes, scale: float) -> np.ndarray:
    return np.asarray(boxes, dtype=float) * scale


def unletterbox_boxes(boxes, scale: float) -> np.ndarray:
    """canvas pixels to camera pixels"""
    return np.asarray(boxes, dtype=float) / scale
