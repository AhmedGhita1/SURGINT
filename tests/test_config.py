"""
Config tests
============

tests the run configuration, which gates every training and evaluation run.

a config that is wrong in a way the dataclass accepts is a wasted GPU run, so the
invalid combinations matter more than the valid ones.

coverage:
- defaults: a minimal config is usable and names itself
- rejects:  each validated field refuses the values that cannot work
- round trip: a saved config loads back unchanged
"""

import pytest

from surgint.config import Config, load_config, save_config

INVALID = {
    "dataset_id": ("", "dataset_id must not be empty"),
    "input_size": ([1024], "input_size must be"),
    "task": ("segmentation", "task must be one of"),
    "schedule": ("linear", "schedule must be"),
    "batch_size": (0, "batch_size must be positive"),
    "val_interval": (0, "val_interval must be positive"),
    "iou_threshold": (0.0, r"iou_threshold must be in \(0, 1\]"),
}


def test_defaults_are_usable():
    """a config with only a dataset id is complete and names its own run"""
    config = Config(dataset_id="test")

    assert config.run_id, "a run must name itself"
    assert config.select_metric in config.metrics, "the selection metric must be measured"
    assert config.revision is None, "the pretrained revision is unpinned unless set"


@pytest.mark.parametrize("field, value, message", [(k, v, m) for k, (v, m) in INVALID.items()])
def test_invalid_values_are_refused(field, value, message):
    """a value that cannot produce a usable run is refused at construction"""
    with pytest.raises(ValueError, match=message):
        Config(**{"dataset_id": "test", field: value})


def test_input_size_must_be_divisible_by_32():
    """the backbone strides by 32, so an odd canvas silently changes the geometry"""
    with pytest.raises(ValueError, match="divisible by 32"):
        Config(dataset_id="test", input_size=[1000, 576])


def test_select_metric_must_be_measured():
    """selecting on a metric the run never computes cannot pick a best checkpoint"""
    with pytest.raises(ValueError, match="is not in metrics"):
        Config(dataset_id="test", metrics=["mAP50"], select_metric="mAP50_95")


def test_round_trip_preserves_the_run(tmp_path):
    """a saved config loads back as the same run, so a rerun is the same run"""
    config = Config(dataset_id="test", revision="abc123", iou_threshold=0.75)
    path = tmp_path / "config.yaml"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config, "the config changed across a save and load"
    assert loaded.run_id == config.run_id, "the run id must not be regenerated on load"
