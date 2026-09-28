import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from surgint.config import Config, save_config

DIGITS = 6  

# the installed versions recorded with every run. these change results
DEPENDENCIES = ("torch", "transformers", "numpy", "scipy", "pycocotools")

WEIGHTS = "model.safetensors"
MODEL_MANIFEST = "manifest.yaml"
LEGACY_MODEL_META = "meta.yaml"
MODEL_SCHEMA_VERSION = 1
MODEL_TYPE = "object-detection"


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


@dataclass(frozen=True)
class ModelManifest:
    """The versioned contract carried by every deployable model artifact."""

    labels: tuple[str, ...]
    input_size: tuple[int, int]
    pad_color: int
    rescale_factor: float
    source: str
    weights_sha256: str
    training_run: str | None = None
    dataset_version: str | None = None
    schema_version: int = MODEL_SCHEMA_VERSION
    model_type: str = MODEL_TYPE
    weights_file: str = WEIGHTS

    def validate(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model manifest schema {self.schema_version}; "
                f"expected {MODEL_SCHEMA_VERSION}"
            )
        if self.model_type != MODEL_TYPE:
            raise ValueError(f"unsupported model type {self.model_type!r}")
        if not self.labels or any(not isinstance(label, str) or not label for label in self.labels):
            raise ValueError("model manifest labels must be non-empty strings")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("model manifest labels must be unique")
        if len(self.input_size) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in self.input_size
        ):
            raise ValueError("model manifest input_size must contain two positive integers")
        if not isinstance(self.pad_color, int) or not 0 <= self.pad_color <= 255:
            raise ValueError("model manifest pad_color must be an integer from 0 to 255")
        if not isinstance(self.rescale_factor, (int, float)) or self.rescale_factor <= 0:
            raise ValueError("model manifest rescale_factor must be positive")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("model manifest source must not be empty")
        if self.weights_file != WEIGHTS:
            raise ValueError(f"model manifest weights file must be {WEIGHTS!r}")
        if not isinstance(self.weights_sha256, str):
            raise ValueError("model manifest weights sha256 is invalid")
        prefix, separator, value = self.weights_sha256.partition(":")
        if prefix != "sha256" or separator != ":" or len(value) != 64:
            raise ValueError("model manifest weights sha256 is invalid")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("model manifest weights sha256 is invalid") from error
        for name, value in (
            ("training_run", self.training_run),
            ("dataset_version", self.dataset_version),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"model manifest {name} must be a non-empty string or null")

    def validate_files(self, checkpoint: str | Path) -> None:
        """Validate the files that make this manifest an immutable artifact."""
        weights = Path(checkpoint) / self.weights_file
        if not weights.is_file():
            raise FileNotFoundError(f"model artifact has no {self.weights_file}")
        actual = digest(weights)
        if actual != self.weights_sha256:
            raise ValueError(
                f"model artifact weights digest mismatch: expected {self.weights_sha256}, got {actual}"
            )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "model": {"type": self.model_type, "source": self.source},
            "labels": list(self.labels),
            "preprocessing": {
                "input_size": list(self.input_size),
                "pad_color": self.pad_color,
                "rescale_factor": self.rescale_factor,
            },
            "weights": {"file": self.weights_file, "sha256": self.weights_sha256},
            "provenance": {
                "training_run": self.training_run,
                "dataset_version": self.dataset_version,
            },
        }

    @property
    def runtime_meta(self) -> dict[str, Any]:
        """Return the flat runtime view used by the existing inference components."""
        return {
            "input_size": list(self.input_size),
            "pad_color": self.pad_color,
            "rescale_factor": self.rescale_factor,
            "source": self.source,
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelManifest":
        if not isinstance(values, Mapping):
            raise ValueError("model manifest must be a mapping")
        try:
            model = values["model"]
            preprocessing = values["preprocessing"]
            weights = values["weights"]
            provenance = values.get("provenance", {})
            for name, section in (
                ("model", model),
                ("preprocessing", preprocessing),
                ("weights", weights),
                ("provenance", provenance),
            ):
                if not isinstance(section, Mapping):
                    raise ValueError(f"model manifest {name} must be a mapping")
            labels = values["labels"]
            input_size = preprocessing["input_size"]
            if not isinstance(labels, (list, tuple)):
                raise ValueError("model manifest labels must be a sequence")
            if not isinstance(input_size, (list, tuple)):
                raise ValueError("model manifest input_size must be a sequence")
            manifest = cls(
                schema_version=values["schema_version"],
                model_type=model["type"],
                source=model["source"],
                labels=tuple(labels),
                input_size=tuple(input_size),
                pad_color=preprocessing["pad_color"],
                rescale_factor=preprocessing["rescale_factor"],
                weights_file=weights["file"],
                weights_sha256=weights["sha256"],
                training_run=provenance.get("training_run"),
                dataset_version=provenance.get("dataset_version"),
            )
        except KeyError as error:
            raise ValueError(f"invalid model manifest structure: {error}") from error
        manifest.validate()
        return manifest

    @classmethod
    def create(
        cls,
        checkpoint: str | Path,
        meta: Mapping[str, Any],
        labels: Sequence[str],
        training_run: str | None = None,
        dataset_version: str | None = None,
    ) -> "ModelManifest":
        weights = Path(checkpoint) / WEIGHTS
        if not weights.is_file():
            raise FileNotFoundError(f"model artifact has no {WEIGHTS}")
        try:
            manifest = cls(
                labels=tuple(labels),
                input_size=tuple(meta["input_size"]),
                pad_color=meta["pad_color"],
                rescale_factor=meta["rescale_factor"],
                source=meta["source"],
                weights_sha256=digest(weights),
                training_run=training_run,
                dataset_version=dataset_version,
            )
        except KeyError as error:
            raise ValueError(f"checkpoint metadata is missing {error.args[0]!r}") from error
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, checkpoint: str | Path) -> "ModelManifest":
        root = Path(checkpoint)
        path = root / MODEL_MANIFEST
        if path.is_file():
            manifest = cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
            manifest.validate_files(root)
            return manifest

        legacy = root / LEGACY_MODEL_META
        if not legacy.is_file():
            raise FileNotFoundError(
                f"model artifact has no {MODEL_MANIFEST} or legacy {LEGACY_MODEL_META}"
            )
        values = yaml.safe_load(legacy.read_text(encoding="utf-8"))
        try:
            manifest = cls.create(root, values, values["labels"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid legacy model metadata: {error}") from error
        manifest.validate_files(root)
        return manifest

    def write(self, checkpoint: str | Path) -> None:
        self.validate_files(checkpoint)
        path = Path(checkpoint) / MODEL_MANIFEST
        path.write_text(yaml.safe_dump(self.as_dict(), sort_keys=False), encoding="utf-8")


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
