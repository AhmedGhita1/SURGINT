from pathlib import Path

import numpy as np
import torch
import yaml

from surgint.detection.model import load_model
from surgint.detection.postprocessing import decode, to_frame_boxes
from surgint.detection.preprocessing import (
    PAD_VALUE,
    RESCALE_FACTOR,
    letterbox,
    to_pixel_values,
)
from surgint.inference import DETOutput


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
    def __init__(
        self,
        checkpoint: str,
        input_size: list[int] | None = None,
        device: str = "cuda",
    ):
        meta_path = Path(checkpoint) / "meta.yaml"
        meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else None
        if meta is not None:
            artifact_size = meta["input_size"]
            if input_size is not None and input_size != artifact_size:
                raise ValueError(
                    f"input_size {input_size} disagrees with checkpoint metadata {artifact_size}"
                )
            if meta["color_space"] != "RGB":
                raise ValueError(f"unsupported color space {meta['color_space']}")
            if meta["resize"] != "letterbox" or meta["letterbox_anchor"] != "top_left":
                raise ValueError("checkpoint requires unsupported resize behavior")
            if meta["normalize"]:
                raise ValueError("normalized checkpoint inputs are not supported")

            input_size = artifact_size
            self.pad_value = int(meta["pad_value"])
            self.rescale_factor = float(meta["rescale_factor"])
            artifact_labels = {index: name for index, name in enumerate(meta["labels"])}
        else:
            if input_size is None:
                raise ValueError(
                    f"{checkpoint} has no meta.yaml; pass input_size for an external checkpoint"
                )
            self.pad_value = PAD_VALUE
            self.rescale_factor = RESCALE_FACTOR
            artifact_labels = None

        self.width, self.height = input_size
        self.detector = Detector(checkpoint, device)
        self.id2label = artifact_labels or self.detector.id2label

    def predict(self, frame: np.ndarray, score_threshold: float) -> DETOutput:
        canvas, scale = letterbox(frame, self.width, self.height, self.pad_value)
        logits, pred_boxes = self.detector(to_pixel_values([canvas], self.rescale_factor))
        boxes, scores, class_ids = decode(
            logits, pred_boxes, self.width, self.height, score_threshold
        )[0]

        return DETOutput(to_frame_boxes(boxes, scale, frame.shape[:2]), scores, class_ids)
