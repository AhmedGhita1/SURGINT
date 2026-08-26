from dataclasses import dataclass

import numpy as np
import torch

from surgint.detection.model import load_model
from surgint.detection.postprocessing import decode
from surgint.detection.preprocessing import letterbox, to_pixel_values, unletterbox_boxes


@dataclass(frozen=True)
class DetectionResult:
    boxes: np.ndarray  # xyxy in original frame pixels
    scores: np.ndarray
    class_ids: np.ndarray


class Detector:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self.model = load_model(checkpoint).to(device).eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def __call__(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) to raw logits and normalized cxcywh boxes"""
        outputs = self.model(pixel_values=pixel_values.to(self.device))
        return outputs.logits.cpu(), outputs.pred_boxes.cpu()


class InferencePipeline:
    def __init__(self, checkpoint: str, input_size: list[int], device: str = "cuda"):
        self.width, self.height = input_size
        self.detector = Detector(checkpoint, device)
        self.id2label = self.detector.id2label

    def predict(self, frame: np.ndarray, score_threshold: float) -> DetectionResult:
        canvas, scale = letterbox(frame, self.width, self.height)
        logits, pred_boxes = self.detector(to_pixel_values([canvas]))
        boxes, scores, class_ids = decode(
            logits, pred_boxes, self.width, self.height, score_threshold
        )[0]

        boxes = unletterbox_boxes(boxes, scale)
        height, width = frame.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)

        return DetectionResult(boxes.astype(np.float32), scores, class_ids)
