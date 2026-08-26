import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from surgint.detection.model import load_model
from surgint.detection.postprocessing import decode, to_frame_boxes
from surgint.detection.preprocessing import letterbox, to_pixel_values


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
    def __init__(self, checkpoint: str, input_size: list[int] | None = None, device: str = "cuda"):
        manifest = Path(checkpoint) / "manifest.json"
        if input_size is None:
            if not manifest.exists():
                raise ValueError(f"{checkpoint} has no manifest; pass input_size for an external checkpoint")
            input_size = json.loads(manifest.read_text())["input_size"]
        self.width, self.height = input_size
        self.detector = Detector(checkpoint, device)
        self.id2label = self.detector.id2label

    def predict(self, frame: np.ndarray, score_threshold: float) -> DetectionResult:
        canvas, scale = letterbox(frame, self.width, self.height)
        logits, pred_boxes = self.detector(to_pixel_values([canvas]))
        boxes, scores, class_ids = decode(
            logits, pred_boxes, self.width, self.height, score_threshold
        )[0]

        return DetectionResult(to_frame_boxes(boxes, scale, frame.shape[:2]), scores, class_ids)
