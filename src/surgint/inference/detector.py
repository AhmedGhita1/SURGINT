import torch

from surgint.detection.model import load_model


class Detector:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self.model = load_model(checkpoint).to(device).eval()

    @torch.inference_mode()
    def __call__(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) to raw logits and normalized cxcywh boxes"""
        outputs = self.model(pixel_values=pixel_values.to(self.device))
        return outputs.logits.cpu(), outputs.pred_boxes.cpu()
