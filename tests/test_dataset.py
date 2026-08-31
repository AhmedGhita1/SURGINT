"""
SurgintDataset and collate tests
================================

tests the boundary between the raw data on disk and the prepared data for model consumption.

coverage:
- mappings:   category ids to contiguous class ids, in either direction
- samples:    keys, dtypes, coordinate space, with and without a transform
- task:       detection-only against detection-tracking
- collate:    stacking, ragged labels, HF key names
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from surgint.dataset.coco import SurgintDataset, collate
from surgint.model.transform import Transform

INPUT_SIZE = [1024, 576]
FRAME_HEIGHT, FRAME_WIDTH = 720, 1280

CATEGORIES = [{"id": 1, "name": "scalpel"}, {"id": 2, "name": "scissors"}]
BOX = [100, 200, 200, 200]                     # xywh, becomes xyxy [100, 200, 300, 400]


def build_root(
    annotations: list[dict],
    split: str = "val",
    images: int = 1,
    categories: list[dict] | None = None,
) -> Path:
    """write a minimal COCO split into a fresh temp directory"""
    root = Path(tempfile.mkdtemp())
    (root / split / "images").mkdir(parents=True)
    (root / "annotations" / Path(split).parent).mkdir(parents=True, exist_ok=True)

    for index in range(images):
        Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT)).save(root / split / "images" / f"{index}.png")

    payload = {
        "images": [
            {
                "id": index,
                "file_name": f"images/{index}.png",
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
            }
            for index in range(images)
        ],
        "annotations": annotations,
        "categories": categories if categories is not None else CATEGORIES,
    }
    path = root / "annotations" / Path(split).parent / f"instances_{Path(split).name}.json"
    path.write_text(json.dumps(payload))
    return root


def annotation(index: int = 1, image_id: int = 0, category_id: int = 1, **extra) -> dict:
    return {
        "id": index,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": BOX,
        "area": BOX[2] * BOX[3],
        "iscrowd": 0,
        **extra,
    }


def test_unit_mappings():
    """category ids from the file become contiguous class ids for the head"""

    # one entry per class id, holding the category id and the category name
    root = build_root([annotation()])
    mappings = SurgintDataset(root, "val", "detection-only").mappings
    assert mappings == {0: (1, "scalpel"), 1: (2, "scissors")}, f"got {mappings}"

    # the original category id stays reachable, so evaluation can write
    # predictions back in the file's own id space
    assert mappings[1][0] == 2, "original category id was lost"

    # ids 3 and 9 with a gap still become 0 and 1
    gappy = [{"id": 3, "name": "scalpel"}, {"id": 9, "name": "scissors"}]
    root = build_root([annotation(category_id=9)], categories=gappy)
    dataset = SurgintDataset(root, "val", "detection-only")
    assert sorted(dataset.mappings) == [0, 1], "class ids must be contiguous from zero"
    assert dataset[0]["class_ids"].tolist() == [1], "category 9 should map to class 1"

    # order inside the json must not decide the class id, or a checkpoint trained
    # on one ordering would mislabel under the other
    reversed_order = [{"id": 2, "name": "scissors"}, {"id": 1, "name": "scalpel"}]
    root = build_root([annotation(category_id=1)], categories=reversed_order)
    dataset = SurgintDataset(root, "val", "detection-only")
    assert dataset.mappings[0] == (1, "scalpel"), "categories must be sorted before numbering"
    assert dataset[0]["class_ids"].tolist() == [0], "category 1 should map to class 0"

    print("mappings passed")


def test_unit_sample():

    root = build_root([annotation()])
    sample = SurgintDataset(root, "val", "detection-only")[0]

    # a raw sample is frame space, no preprocessing
    assert set(sample) == {"image_id", "frame", "boxes", "class_ids"}, f"got {sorted(sample)}"
    assert sample["frame"].dtype == np.uint8, "frame must stay uint8"
    assert sample["frame"].shape == (FRAME_HEIGHT, FRAME_WIDTH, 3), "frame must be HWC at full size"
    assert sample["boxes"].dtype == np.float32, "boxes must be float32"
    assert sample["class_ids"].dtype == np.int64, "class ids must be int64"

    # coco xywh [100, 200, 200, 200] becomes xyxy [100, 200, 300, 400]
    assert sample["boxes"].tolist() == [[100.0, 200.0, 300.0, 400.0]], f"got {sample['boxes'].tolist()}"

    # an image with no annotations still yields a sample
    root = build_root([annotation(image_id=0)], images=2)
    empty = SurgintDataset(root, "val", "detection-only")[1]
    assert empty["boxes"].shape == (0, 4), "empty boxes lost their shape"
    assert empty["class_ids"].shape == (0,), "empty class ids lost their shape"

    # a transform replaces the frame with model input
    dataset = SurgintDataset(root, "val", "detection-only", Transform(INPUT_SIZE))
    transformed = dataset[0]
    assert "frame" not in transformed, "the frame should not survive the transform"
    assert transformed["pixel_values"].shape == (3, INPUT_SIZE[1], INPUT_SIZE[0]), "wrong input shape"
    assert transformed["scale"] == 0.8, "scale must survive for the inverse"
    assert transformed["frame_size"] == (FRAME_HEIGHT, FRAME_WIDTH), "frame size must survive"

    # image_id is the join key back to the json, so it outlives the transform
    assert [dataset[index]["image_id"] for index in range(2)] == [0, 1], "image ids were lost"

    print("sample contract passed")


def test_unit_task():
    """track_ids appear only under detection-tracking"""

    # detection-only omits track ids
    root = build_root([annotation(track_id=7)])
    assert "track_ids" not in SurgintDataset(root, "val", "detection-only")[0], "track ids leaked"

    # detection-tracking carries track ids
    sample = SurgintDataset(root, "val", "detection-tracking")[0]
    assert sample["track_ids"].tolist() == [7], f"got {sample['track_ids'].tolist()}"
    assert sample["track_ids"].dtype == np.int64, "track ids must be int64"

    # tracking against a file with no track_id is a data error, caught at load
    root = build_root([annotation()])
    try:
        SurgintDataset(root, "val", "detection-tracking")
        assert False, "should raise ValueError when the annotations carry no track_id"
    except ValueError:
        pass

    # an unrecognized task fails immediately
    try:
        SurgintDataset(root, "val", "segmentation")
        assert False, "should raise ValueError for an unknown task"
    except ValueError:
        pass


def test_unit_collate():

    # two images, one box on the first and two on the second
    annotations = [
        annotation(1, image_id=0, category_id=1),
        annotation(2, image_id=1, category_id=2),
        annotation(3, image_id=1, category_id=1),
    ]
    root = build_root(annotations, images=2)
    dataset = SurgintDataset(root, "val", "detection-only", Transform(INPUT_SIZE))
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))

    # pixel values stack because every canvas is the same size
    assert batch["pixel_values"].shape == (2, 3, INPUT_SIZE[1], INPUT_SIZE[0]), "stacking failed"
    assert batch["pixel_values"].dtype == torch.float32, "model input must be float32"

    # labels cannot stack, so they stay one dict per image
    assert [label["boxes"].shape for label in batch["labels"]] == [(1, 4), (2, 4)], "ragged labels merged"

    # the label keys are the ones RTDetrForObjectDetection reads
    assert set(batch["labels"][0]) == {"class_labels", "boxes"}, f"got {sorted(batch['labels'][0])}"
    assert batch["labels"][0]["class_labels"].dtype == torch.int64, "class labels must be int64"
    assert batch["labels"][0]["boxes"].dtype == torch.float32, "boxes must be float32"

    # the terms evaluation needs to invert survive batching
    assert batch["image_ids"] == [0, 1], "image ids were lost"
    assert batch["scales"] == [0.8, 0.8], "scales were lost"
    assert batch["frame_sizes"] == [(FRAME_HEIGHT, FRAME_WIDTH)] * 2, "frame sizes were lost"

    # a batch of one keeps its batch dimension
    single = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate)))
    assert single["pixel_values"].shape[0] == 1, "batch dimension collapsed"
    assert len(single["labels"]) == 1, "labels lost the batch dimension"


if __name__ == "__main__":
    test_unit_mappings()
    test_unit_sample()
    test_unit_task()
    test_unit_collate()
    print("\nall passed")
