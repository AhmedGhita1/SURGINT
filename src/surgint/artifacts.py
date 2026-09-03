import json
from pathlib import Path

import torch
import yaml

from surgint.config import Config, save_config

DIGITS = 6  


def round_floats(value, digits: int = DIGITS):
    """trim float precision for the json records"""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    return value


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
            stream.write(json.dumps(round_floats(record)) + "\n")

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
        (self.root / "summary.json").write_text(json.dumps(round_floats(summary), indent=2) + "\n")
