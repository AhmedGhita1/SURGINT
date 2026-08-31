from typing import Dict, Tuple, cast

import torch
from torch import nn
from transformers import RTDetrForObjectDetection


class Detector(nn.Module):
    def __init__(self, model: RTDetrForObjectDetection):
        super().__init__()
        self.model = model

    @classmethod
    def from_checkpoint(cls, checkpoint: str) -> "Detector":
        """loads a surgint checkpoint."""
        return cls(
            cast(RTDetrForObjectDetection, RTDetrForObjectDetection.from_pretrained(checkpoint))
        )

    @classmethod
    def from_pretrained(cls, checkpoint: str, id2label: Dict[int, str]) -> "Detector":
        """loads a pretrained checkpoint with the classifier reinitialized and resized to id2label"""
        return cls(
            cast(
                RTDetrForObjectDetection,
                RTDetrForObjectDetection.from_pretrained(
                    checkpoint,
                    id2label=id2label,
                    label2id={name: index for index, name in id2label.items()},
                    ignore_mismatched_sizes=True,
                ),
            )
        )

    @property
    def id2label(self) -> Dict[int, str]:
        return self.model.config.id2label

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, pixel_values, labels=None):
        """training."""
        pixel_values = pixel_values.to(self.device)
        if labels is not None:
            labels = [{key: value.to(self.device) for key, value in label.items()} for label in labels]
        return self.model(pixel_values=pixel_values, labels=labels)

    @torch.inference_mode()
    def predict(self, pixel_values) -> Tuple[torch.Tensor, torch.Tensor]:
        """eval and inference."""
        self.eval()
        outputs = self(pixel_values)
        return outputs.logits.cpu(), outputs.pred_boxes.cpu()
