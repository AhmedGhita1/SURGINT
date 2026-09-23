"""
Run artifact tests
==================

tests what a run records about itself. a metric is only comparable against another
run when both state the code, data, and model they came from.

coverage:
- manifest:   dataset, categories, and the annotation digests the run read
- provenance: surgint version, dependency versions, hardware
- model:      a hub model records its revision, a local checkpoint its weights digest
- run dir:    an existing run is never reinitialized
- records:    the log appends one json object per line, floats are trimmed
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from surgint.artifacts import DEPENDENCIES, WEIGHTS, RunWriter, digest
from surgint.config import Config

CATEGORIES = ["scalpel", "scissors"]
HUB_MODEL = "PekingU/rtdetr_r18vd_coco_o365"
REVISION = "386c43855d01bcd966e548f3296fbd2cd2a2a40b"


@pytest.fixture
def annotations(tmp_path) -> Path:
    """one coco file with known bytes"""
    path = tmp_path / "instances_train.json"
    path.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
    return path


@pytest.fixture
def checkpoint(tmp_path) -> Path:
    """a local checkpoint directory carrying weights"""
    path = tmp_path / "best"
    path.mkdir()
    (path / WEIGHTS).write_bytes(b"not real weights, but real bytes")
    return path


def read_manifest(root: Path) -> dict:
    return yaml.safe_load((root / "manifest.yaml").read_text())


def build_config(**settings) -> Config:
    return Config(dataset_id="test", data_root="data/test/dataset", **settings)


def test_manifest_records_the_data_the_run_read(tmp_path, annotations):
    """the dataset, its categories, and a digest per annotation file"""

    root = tmp_path / "run"
    RunWriter(root).initialize(
        build_config(), CATEGORIES, HUB_MODEL, REVISION, [annotations]
    )
    manifest = read_manifest(root)

    assert manifest["dataset_id"] == "test"
    assert manifest["dataset_root"] == "data/test/dataset"
    assert manifest["format"] == "coco"
    assert manifest["categories"] == CATEGORIES

    expected = hashlib.sha256(annotations.read_bytes()).hexdigest()
    assert manifest["annotations"] == {"instances_train.json": f"sha256:{expected}"}


def test_manifest_records_the_environment(tmp_path, annotations):
    """the code version, the versions that change results, and the device"""

    root = tmp_path / "run"
    RunWriter(root).initialize(build_config(), CATEGORIES, HUB_MODEL, REVISION, [annotations])
    manifest = read_manifest(root)

    assert manifest["surgint"], "the surgint version is missing"
    assert set(manifest["dependencies"]) == set(DEPENDENCIES), f"got {sorted(manifest['dependencies'])}"
    assert all(manifest["dependencies"].values()), "a dependency version is empty"
    assert set(manifest["hardware"]) == {"device", "devices", "cuda", "cudnn"}
    assert manifest["hardware"]["device"], "the device is missing"


def test_hub_model_records_its_revision(tmp_path, annotations):
    """a hub id pins to a commit and has no local weights to digest"""

    root = tmp_path / "run"
    RunWriter(root).initialize(build_config(), CATEGORIES, HUB_MODEL, REVISION, [annotations])
    manifest = read_manifest(root)

    assert manifest["model"] == HUB_MODEL
    assert manifest["revision"] == REVISION
    assert manifest["weights"] is None, "a hub id is not a local checkpoint"


def test_local_checkpoint_records_its_weights(tmp_path, annotations, checkpoint):
    """an evaluation run identifies the checkpoint by content, not by path"""

    root = tmp_path / "run"
    RunWriter(root).initialize(build_config(), CATEGORIES, str(checkpoint), annotations=[annotations])
    manifest = read_manifest(root)

    assert manifest["revision"] is None, "a local checkpoint has no hub commit"
    assert manifest["weights"] == digest(checkpoint / WEIGHTS)

    # the digest tracks content, so an edited checkpoint stops matching its record
    (checkpoint / WEIGHTS).write_bytes(b"different bytes entirely")
    assert manifest["weights"] != digest(checkpoint / WEIGHTS)


def test_config_is_saved_with_the_pinned_revision(tmp_path, annotations):
    """the run keeps the config it ran under, revision included"""

    root = tmp_path / "run"
    config = build_config(revision=REVISION)
    RunWriter(root).initialize(config, CATEGORIES, HUB_MODEL, REVISION, [annotations])

    saved = yaml.safe_load((root / "config.yaml").read_text())
    assert saved["revision"] == REVISION
    assert saved["run_id"] == config.run_id


def test_an_existing_run_is_never_reinitialized(tmp_path, annotations):
    """a second run must not write over the first one's records"""

    root = tmp_path / "run"
    writer = RunWriter(root)
    writer.initialize(build_config(), CATEGORIES, HUB_MODEL, REVISION, [annotations])

    with pytest.raises(FileExistsError):
        writer.initialize(build_config(), CATEGORIES, HUB_MODEL, REVISION, [annotations])


def test_records_append_and_trim(tmp_path, annotations):
    """one json object per log line, and floats cut to six significant digits"""

    root = tmp_path / "run"
    writer = RunWriter(root)
    writer.initialize(build_config(), CATEGORIES, HUB_MODEL, REVISION, [annotations])

    writer.append_log({"epoch": 1, "train_loss": 1.234567891})
    writer.append_log({"epoch": 2, "train_loss": 0.5})
    lines = [json.loads(line) for line in (root / "run.log").read_text().splitlines()]

    assert [record["epoch"] for record in lines] == [1, 2], "records must append in order"
    assert lines[0]["train_loss"] == 1.23457, f"got {lines[0]['train_loss']}"

    writer.write_summary({"best": {"mAP50_95": 0.123456789}})
    summary = json.loads((root / "summary.json").read_text())
    assert summary["best"]["mAP50_95"] == 0.123457
