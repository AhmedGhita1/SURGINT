import json
from pathlib import Path

import torch
import yaml

from surgint.config import Config, save_config
from surgint.detection.preprocessing import PAD_VALUE, RESCALE_FACTOR


def build_checkpoint_meta(input_size: list[int], id2label: dict[int, str]) -> dict:
    """Build the run metadata."""
    label_ids = sorted(id2label)
    if label_ids != list(range(len(label_ids))):
        raise ValueError(f"label ids must be contiguous from zero, got {label_ids}")

    return {
        "input_size": input_size,
        "color_space": "RGB",
        "resize": "letterbox",
        "letterbox_anchor": "top_left",
        "pad_value": PAD_VALUE,
        "rescale_factor": RESCALE_FACTOR,
        "normalize": False,
        "labels": [id2label[index] for index in label_ids],
    }


class RunWriter:
    """Write one training or evaluation run without owning computation."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def initialize(
        self,
        config: Config,
        categories: list[str],
        model: str,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        save_config(config, self.root / "config.yaml")
        manifest = {
            "dataset_id": config.dataset_id,
            "dataset_root": config.data_root,
            "format": "coco",
            "categories": categories,
            "model": model,
        }
        (self.root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    def append_log(self, record: dict) -> None:
        with (self.root / "run.log").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def save_checkpoint(
        self,
        name: str,
        model: torch.nn.Module,
        meta: dict,
        training_state: dict | None = None,
    ) -> Path:
        if name not in {"best", "latest"}:
            raise ValueError(f"checkpoint must be best or latest, got {name}")
        if name == "latest" and training_state is None:
            raise ValueError("latest checkpoint requires training state")
        if name == "best" and training_state is not None:
            raise ValueError("best checkpoint must not contain training state")

        checkpoint = self.root / name
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        (checkpoint / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        if training_state is not None:
            torch.save(training_state, checkpoint / "training_state.pt")
        return checkpoint

    def write_summary(self, summary: dict) -> None:
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
