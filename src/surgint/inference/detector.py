import torch
from transformers import RTDetrImageProcessor

from surgint.detection.model import load_model


class Detector:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self.processor = RTDetrImageProcessor.from_pretrained(checkpoint)
        self.model = load_model(checkpoint).to(device).eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def __call__(self, canvases) -> tuple[torch.Tensor, torch.Tensor]:
        """letterboxed canvases to raw logits and normalized cxcywh boxes"""
        pixel_values = self.processor(images=canvases, do_resize=False, return_tensors="pt")["pixel_values"]
        outputs = self.model(pixel_values=pixel_values.to(self.device))
        return outputs.logits, outputs.pred_boxes
