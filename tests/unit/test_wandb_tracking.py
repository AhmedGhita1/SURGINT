"""Tests for the pipeline-owned W&B adapter."""

from pathlib import Path

import pytest

from pipelines import wandb_tracking
from surgint.artifacts import ModelManifest, WEIGHTS

META = {
    "input_size": [1024, 576],
    "pad_color": 114,
    "rescale_factor": 1 / 255,
    "source": "source-model",
}


class LoggedArtifact:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class FakeRun:
    def __init__(self):
        self.logs = []
        self.artifacts = []

    def log(self, record, step):
        self.logs.append((record, step))

    def log_artifact(self, path, **settings):
        artifact = LoggedArtifact()
        self.artifacts.append((path, settings, artifact))
        return artifact


def model_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / WEIGHTS).write_bytes(b"weights")
    manifest = ModelManifest.create(
        root,
        META,
        ["scalpel", "scissors"],
        training_run="run-123",
        dataset_version="dataset-v5",
    )
    manifest.write(root)
    return root


def test_init_training_run_forwards_pipeline_context(monkeypatch):
    calls = []
    expected = object()

    class FakeWandb:
        @staticmethod
        def init(**settings):
            calls.append(settings)
            return expected

    monkeypatch.setattr(wandb_tracking.importlib, "import_module", lambda name: FakeWandb)

    run = wandb_tracking.init_training_run(
        "surgint",
        "team",
        "run-123",
        {"epochs": 2},
    )

    assert run is expected
    assert calls == [
        {
            "project": "surgint",
            "entity": "team",
            "job_type": "train",
            "name": "run-123",
            "config": {"epochs": 2},
        }
    ]


def test_missing_sdk_has_an_actionable_error(monkeypatch):
    def missing(_):
        raise ModuleNotFoundError("wandb")

    monkeypatch.setattr(wandb_tracking.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match=r"\[training\]"):
        wandb_tracking.init_training_run("surgint", None, "run-123", {})


def test_epoch_logging_preserves_the_epoch_step():
    run = FakeRun()
    wandb_tracking.log_epoch(run, {"train_loss": 0.5}, epoch=3)
    assert run.logs == [({"train_loss": 0.5}, 3)]


def test_candidate_model_is_validated_and_logged(tmp_path):
    root = model_artifact(tmp_path)
    run = FakeRun()

    artifact = wandb_tracking.log_candidate_model(
        run,
        root,
        "surgint-detector",
        "run-123",
    )

    assert run.artifacts[0][0] == str(root)
    assert run.artifacts[0][1] == {
        "name": "surgint-detector",
        "type": "model",
        "aliases": ["run-123"],
    }
    assert artifact.waited


def test_invalid_candidate_is_not_uploaded(tmp_path):
    root = model_artifact(tmp_path)
    (root / WEIGHTS).write_bytes(b"changed")
    run = FakeRun()

    with pytest.raises(ValueError, match="digest mismatch"):
        wandb_tracking.log_candidate_model(run, root, "surgint-detector", "run-123")

    assert not run.artifacts
