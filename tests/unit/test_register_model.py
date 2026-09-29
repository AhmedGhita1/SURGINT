"""Tests for bootstrapping an existing checkpoint into W&B."""

from pathlib import Path

import pytest

from pipelines import register_model
from surgint.artifacts import ModelManifest, WEIGHTS

META = {
    "input_size": [1024, 576],
    "pad_color": 114,
    "rescale_factor": 1 / 255,
    "source": "source-model",
}


def model_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / WEIGHTS).write_bytes(b"weights")
    manifest = ModelManifest.create(root, META, ["scalpel"])
    manifest.write(root)
    return root


def test_register_model_targets_and_verifies_the_exact_project(monkeypatch, tmp_path):
    checkpoint = model_artifact(tmp_path)
    reference = "SETLabs-HCT/surgint/surgint-detector:v0"
    calls = []

    class Artifact:
        qualified_name = reference

        def wait(self):
            calls.append(("wait",))

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            calls.append(("finish",))

        def log_artifact(self, path, **settings):
            calls.append(("log", path, settings))
            return Artifact()

    class Api:
        def artifact(self, name, type):
            calls.append(("verify", name, type))
            return object()

    class Wandb:
        @staticmethod
        def init(**settings):
            calls.append(("init", settings))
            return Run()

        @staticmethod
        def Api():
            return Api()

    monkeypatch.setattr(register_model.importlib, "import_module", lambda _: Wandb)

    result = register_model.register_model(checkpoint, "surgint", "SETLabs-HCT")

    assert result == reference
    assert calls == [
        (
            "init",
            {
                "project": "surgint",
                "entity": "SETLabs-HCT",
                "job_type": "model-bootstrap",
                "name": "register-model",
            },
        ),
        (
            "log",
            str(checkpoint),
            {
                "name": "surgint-detector",
                "type": "model",
                "aliases": ["candidate"],
            },
        ),
        ("wait",),
        ("finish",),
        ("verify", reference, "model"),
    ]


def test_invalid_checkpoint_is_rejected_before_upload(monkeypatch, tmp_path):
    checkpoint = model_artifact(tmp_path)
    (checkpoint / WEIGHTS).write_bytes(b"changed")

    class Wandb:
        @staticmethod
        def init(**_):
            raise AssertionError("invalid checkpoint must not start a W&B run")

    monkeypatch.setattr(register_model.importlib, "import_module", lambda _: Wandb)

    with pytest.raises(ValueError, match="digest mismatch"):
        register_model.register_model(checkpoint, "surgint", "SETLabs-HCT")
