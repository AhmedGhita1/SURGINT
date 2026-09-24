"""
Trainer tests
=============

tests the training loop: optimization setup, the epoch step, validation, and the
state that resumes a run.

coverage:
- setup:      two parameter groups, schedule length
- epoch:      finite loss, the schedule advances, batchnorm stays frozen
- validate:   configured metrics plus per class, ground truth scores perfect
- loop:       val_interval, the last epoch, best selection
- state:      state_dict round trip
"""

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from surgint.config import Config
from surgint.dataset.coco import SurgintDataset, collate
from surgint.model.transform import Transform
from surgint.engine.trainer import Trainer
from surgint.model.detector import Detector

# these load a real checkpoint from the hub
pytestmark = pytest.mark.integration

CHECKPOINT = "PekingU/rtdetr_r18vd_coco_o365"
INPUT_SIZE = [320, 192]
FRAME_HEIGHT, FRAME_WIDTH = 180, 320
BOX = [40, 60, 60, 50]                          # xywh, the bright patch in every frame
CATEGORIES = [{"id": 0, "name": "scalpel"}, {"id": 1, "name": "scissors"}]


@lru_cache(maxsize=1)
def build_root() -> Path:
    """two frames per split, one box each, drawn where the annotation says"""
    root = Path(tempfile.mkdtemp())
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        for index in range(2):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 30, np.uint8)
            frame[BOX[1]:BOX[1] + BOX[3], BOX[0]:BOX[0] + BOX[2]] = 220
            Image.fromarray(frame).save(root / split / "images" / f"{index}.png")

    (root / "annotations").mkdir()
    for split in ("train", "val"):
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps({
            "images": [
                {"id": i, "file_name": f"images/{i}.png",
                 "width": FRAME_WIDTH, "height": FRAME_HEIGHT} for i in range(2)
            ],
            "annotations": [
                {"id": i + 1, "image_id": i, "category_id": i % 2, "bbox": BOX,
                 "area": BOX[2] * BOX[3], "iscrowd": 0} for i in range(2)
            ],
            "categories": CATEGORIES,
        }))
    return root


def build_trainer(**overrides) -> Trainer:
    settings = {"epochs": 1, "batch_size": 2, "warmup_steps": 1, **overrides}
    config = Config(dataset_id="test", **settings)
    root = build_root()
    transform = Transform(INPUT_SIZE)

    train_set = SurgintDataset(root, "train", "detection-only", transform)
    val_set = SurgintDataset(root, "val", "detection-only", transform)
    detector = Detector.from_pretrained(
        CHECKPOINT, {class_id: name for class_id, (_, name) in train_set.mappings.items()}
    )

    return Trainer(
        detector,
        config,
        DataLoader(train_set, batch_size=config.batch_size, collate_fn=collate),
        DataLoader(val_set, batch_size=config.batch_size, collate_fn=collate),
        transform,
    )


def test_unit_setup():
    """what the constructor builds before any step runs"""

    trainer = build_trainer(epochs=3)

    # the backbone trains slower than the reinitialized head
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2, f"expected two parameter groups, got {len(groups)}"
    assert groups[0]["lr"] < groups[1]["lr"], "the backbone must take the lower rate"
    assert all(group["params"] for group in groups), "a parameter group is empty"

    # the schedule spans every step of the run, so cosine reaches zero at the end
    assert trainer.scheduler is not None, "the schedule is built at construction"
    assert trainer.epoch == 0 and trainer.best_epoch == 0
    assert trainer.best_score == float("-inf"), "nothing has been scored yet"


def test_unit_train_epoch():
    """one pass over the training split"""

    trainer = build_trainer()
    before = trainer.scheduler.last_epoch

    loss = trainer.train_epoch()

    assert np.isfinite(loss), f"loss is {loss}"
    assert loss > 0.0, "detection loss cannot be zero on the first epoch"

    # the schedule advances once per batch, not once per epoch
    batches = len(trainer.train_loader)
    assert trainer.scheduler.last_epoch == before + batches, "the schedule did not step per batch"

    # batchnorm stays in eval mode so train and inference compute the same function
    frozen = [m for m in trainer.detector.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert frozen and all(not m.training for m in frozen), "batchnorm was not frozen"


def test_unit_validate():
    """coco mAP over the val split"""

    trainer = build_trainer()
    metrics = trainer.validate()

    # everything coco_evaluate reports, per class included
    for name in trainer.config.metrics:
        assert name in metrics, f"{name} missing from {sorted(metrics)}"
    assert "per_class" in metrics, "per class AP must reach the run summary"
    assert set(metrics["per_class"]) == {"scalpel", "scissors"}

    assert 0.0 <= metrics["mAP50_95"] <= 1.0, f"got {metrics['mAP50_95']}"

    # validate leaves the model in eval mode, since predict sets it
    assert not trainer.detector.training, "validate must not leave the model in train mode"


def test_unit_train_loop():
    """val_interval, best selection, and the epoch record"""

    trainer = build_trainer(epochs=3, val_interval=2)
    results = list(trainer.train())

    assert [r.epoch for r in results] == [1, 2, 3]

    # interval 2 skips epoch 1; the last epoch always validates
    assert results[0].val is None, "a skipped epoch must say it did not run, not report nothing"
    assert results[1].val, "epoch 2 should validate"
    assert results[2].val, "the last epoch must always validate"

    # a skipped epoch cannot be best
    assert not results[0].is_best, "an unscored epoch must not be selected"

    # best tracks the highest select_metric seen
    scored = [r for r in results if r.val]
    best = max(scored, key=lambda r: r.val[trainer.config.select_metric])
    assert trainer.best_epoch == best.epoch, f"got {trainer.best_epoch}"
    assert trainer.best_score == best.val[trainer.config.select_metric]
    assert trainer.best_val == best.val, "the trainer keeps the best epoch's validation"

    # the epoch record is one json line, per class AP included
    row = results[2].as_dict()
    assert set(row) == {"epoch", "lr", "train_loss", "per_class", *trainer.config.metrics}
    json.dumps(row)

    # a skipped epoch writes the training half only
    assert set(results[0].as_dict()) == {"epoch", "lr", "train_loss"}


def test_unit_state():
    """resuming a run"""

    trainer = build_trainer(epochs=2)
    list(trainer.train())
    state = trainer.state_dict()

    assert set(state) == {
        "epoch", "best_score", "best_epoch", "best_val", "optimizer", "scheduler"
    }, f"got {sorted(state)}"

    # the trainer writes and reads its own state, next to the weights
    with tempfile.TemporaryDirectory() as directory:
        trainer.save(directory)
        assert (Path(directory) / "training_state.pt").exists(), "training_state.pt was not written"

        # a fresh trainer picks up where the first left off
        resumed = build_trainer(epochs=2)
        assert resumed.epoch == 0
        resumed.load(directory)

    assert resumed.epoch == trainer.epoch
    assert resumed.best_score == trainer.best_score
    assert resumed.best_epoch == trainer.best_epoch
    assert resumed.scheduler.last_epoch == trainer.scheduler.last_epoch, "the schedule did not resume"

    # a resumed run continues rather than restarting
    assert list(resumed.train()) == [], "epochs already run must not repeat"


if __name__ == "__main__":
    test_unit_setup()
    test_unit_train_epoch()
    test_unit_validate()
    test_unit_train_loop()
    test_unit_state()
    print("\nall passed")
