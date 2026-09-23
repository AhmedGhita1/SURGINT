import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import torch
import yaml

from surgint.config import Config, save_config

DIGITS = 6  

# the installed versions recorded with every run. these change results
DEPENDENCIES = ("torch", "transformers", "numpy", "scipy", "pycocotools")

WEIGHTS = "model.safetensors"


def round_floats(value, digits: int = DIGITS):
    """trim float precision for the json records"""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    return value


def digest(path: str | Path, block: int = 1 << 20) -> str:
    """Return the sha256 of one file as `sha256:<hex>`."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            sha.update(chunk)
    return f"sha256:{sha.hexdigest()}"


def installed_versions(names: Sequence[str] = DEPENDENCIES) -> dict:
    """Return the installed version of each distribution, or absent when it is not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return versions


def weights_digest(model: str) -> str | None:
    """Return the sha256 of a local checkpoint's weights, or None when model is not a local checkpoint."""
    weights = Path(model) / WEIGHTS
    return digest(weights) if weights.is_file() else None


def hardware() -> dict:
    """Return the device the run executes on."""
    if not torch.cuda.is_available():
        return {"device": platform.processor() or "cpu", "devices": 0, "cuda": None, "cudnn": None}

    return {
        "device": torch.cuda.get_device_name(0),
        "devices": torch.cuda.device_count(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
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
        revision: str | None = None,
        annotations: Sequence[str | Path] = (),
    ) -> None:
        """
        create the run directory, then write config.yaml and manifest.yaml.

        annotations are the coco files the run reads, recorded by digest. revision is
        the resolved hub commit of model; a local checkpoint is digested instead.
        """
        self.root.mkdir(parents=True, exist_ok=False)
        save_config(config, self.root / "config.yaml")
        manifest = {
            "dataset_id": config.dataset_id,
            "dataset_root": config.data_root,
            "format": "coco",
            "categories": categories,
            "annotations": {Path(path).name: digest(path) for path in annotations},
            "model": model,
            "revision": revision,
            "weights": weights_digest(model),
            "surgint": installed_versions(("surgint",))["surgint"],
            "dependencies": installed_versions(),
            "hardware": hardware(),
        }
        (self.root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    def append_log(self, record: dict) -> None:
        with (self.root / "run.log").open("a") as stream:
            stream.write(json.dumps(round_floats(record)) + "\n")

    def write_summary(self, summary: dict) -> None:
        (self.root / "summary.json").write_text(json.dumps(round_floats(summary), indent=2) + "\n")
