from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn
from transformers import RTDetrForObjectDetection

from surgint.artifacts import ModelManifest


class Detector(nn.Module):
    def __init__(
        self,
        model: RTDetrForObjectDetection,
        manifest: Optional[ModelManifest] = None,
    ):
        super().__init__()
        self.model = model
        self.manifest = manifest
        self.meta = manifest.runtime_meta if manifest is not None else None

    @classmethod
    def from_checkpoint(cls, checkpoint: Union[str, Path]) -> "Detector":
        """Load and validate a Surgint model artifact."""
        manifest = ModelManifest.load(checkpoint)
        model = cast(
            RTDetrForObjectDetection,
            RTDetrForObjectDetection.from_pretrained(checkpoint),
        )
        labels = {int(index): label for index, label in model.config.id2label.items()}
        expected = {index: label for index, label in enumerate(manifest.labels)}
        if labels != expected:
            raise ValueError(f"model config labels {labels} do not match manifest labels {expected}")
        return cls(model, manifest)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        id2label: Dict[int, str],
        revision: Optional[str] = None,
    ) -> "Detector":
        """loads a pretrained checkpoint with the classifier reinitialized and resized to id2label"""
        return cls(
            cast(
                RTDetrForObjectDetection,
                RTDetrForObjectDetection.from_pretrained(
                    checkpoint,
                    revision=revision,
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
    def revision(self) -> Optional[str]:
        """the hub commit the weights were resolved from."""
        return getattr(self.model.config, "_commit_hash", None)

    def parameter_split(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """backbone and head parameters"""
        backbone_ids = {id(p) for p in self.model.model.backbone.parameters()}
        backbone, head = [], []
        for parameter in self.parameters():
            if parameter.requires_grad:
                (backbone if id(parameter) in backbone_ids else head).append(parameter)
        return backbone, head

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

    def save_checkpoint(
        self,
        checkpoint: Union[str, Path],
        meta: Dict,
        *,
        training_run: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> None:
        """Write weights and their versioned model manifest."""
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint)

        labels = [self.id2label[index] for index in sorted(self.id2label)]
        manifest = ModelManifest.create(
            checkpoint,
            meta,
            labels,
            training_run=training_run,
            dataset_version=dataset_version,
        )
        manifest.write(checkpoint)
        self.manifest = manifest
        self.meta = manifest.runtime_meta
