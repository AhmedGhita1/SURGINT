"""Unit tests for the deployable model artifact contract."""

from pathlib import Path

import pytest
import yaml

from surgint.artifacts import MODEL_MANIFEST, WEIGHTS, ModelManifest, digest

LABELS = ("scalpel", "scissors")
META = {
    "input_size": [1024, 576],
    "pad_color": 114,
    "rescale_factor": 1 / 255,
    "source": "PekingU/rtdetr_r18vd_coco_o365",
}


def checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / WEIGHTS).write_bytes(b"model weights")
    return root


def write_manifest(tmp_path: Path) -> tuple[Path, ModelManifest]:
    root = checkpoint(tmp_path)
    manifest = ModelManifest.create(
        root,
        META,
        LABELS,
        training_run="run-123",
        dataset_version="dataset-v5",
    )
    manifest.write(root)
    return root, manifest


def test_manifest_round_trip(tmp_path):
    root, expected = write_manifest(tmp_path)

    loaded = ModelManifest.load(root)
    values = yaml.safe_load((root / MODEL_MANIFEST).read_text(encoding="utf-8"))

    assert loaded == expected
    assert values["schema_version"] == 1
    assert values["model"] == {"type": "object-detection", "source": META["source"]}
    assert values["preprocessing"]["input_size"] == META["input_size"]
    assert values["weights"] == {"file": WEIGHTS, "sha256": digest(root / WEIGHTS)}
    assert values["provenance"] == {
        "training_run": "run-123",
        "dataset_version": "dataset-v5",
    }


def test_manifest_rejects_changed_weights(tmp_path):
    root, _ = write_manifest(tmp_path)
    (root / WEIGHTS).write_bytes(b"tampered weights")

    with pytest.raises(ValueError, match="digest mismatch"):
        ModelManifest.load(root)


def test_manifest_rejects_unknown_schema(tmp_path):
    root, _ = write_manifest(tmp_path)
    path = root / MODEL_MANIFEST
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["schema_version"] = 2
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported model manifest schema 2"):
        ModelManifest.load(root)


def test_manifest_rejects_invalid_structure(tmp_path):
    root, _ = write_manifest(tmp_path)
    path = root / MODEL_MANIFEST
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["weights"]["sha256"] = 7
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="weights sha256 is invalid"):
        ModelManifest.load(root)


def test_legacy_meta_remains_loadable(tmp_path):
    root = checkpoint(tmp_path)
    legacy = {**META, "labels": list(LABELS)}
    (root / "meta.yaml").write_text(yaml.safe_dump(legacy), encoding="utf-8")

    loaded = ModelManifest.load(root)

    assert loaded.runtime_meta == legacy
    assert loaded.weights_sha256 == digest(root / WEIGHTS)
    assert loaded.training_run is None
    assert loaded.dataset_version is None
