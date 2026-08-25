import numpy as np
from PIL import Image

PAD_VALUE = 114  # mid-gray


def letterbox(image: Image.Image, width: int, height: int) -> tuple[Image.Image, float]:
    """aspect-preserving resize, padded to a fixed canvas"""
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)))

    canvas = Image.new("RGB", (width, height), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    canvas.paste(resized, (0, 0))
    return canvas, scale


def letterbox_boxes(boxes, scale: float) -> np.ndarray:
    return np.asarray(boxes, dtype=float) * scale


def unletterbox_boxes(boxes, scale: float) -> np.ndarray:
    """canvas pixels to camera pixels"""
    return np.asarray(boxes, dtype=float) / scale
