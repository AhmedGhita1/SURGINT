# unit tests:
#   box conversion,
#   category to label mapping,
#   images without annotations,
#   zero-area boxes,
#   clipping to the canvas,
#   collate batching,
#   uint8 until collate,
#   reordered and non-contiguous category ids

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from surgint.dataset.coco import CocoDetection, collate

INPUT_SIZE = [1024, 576]
FRAME = (720, 1280)


def build_dataset(annotations: list[dict], images: int = 1, categories: list[dict] | None = None) -> CocoDetection:
    root = Path(tempfile.mkdtemp())
    (root / "val").mkdir()
    (root / "annotations").mkdir()

    for index in range(images):
        Image.new("RGB", (FRAME[1], FRAME[0])).save(root / "val" / f"{index}.png")

    (root / "annotations" / "instances_val.json").write_text(
        json.dumps(
            {
                "images": [{"id": i, "file_name": f"{i}.png"} for i in range(images)],
                "annotations": annotations,
                "categories": categories or [{"id": 1, "name": "scalpel"}, {"id": 2, "name": "scissors"}],
            }
        )
    )
    return CocoDetection(root, "val", INPUT_SIZE)


def test_boxes_become_normalized_cxcywh_on_the_canvas():
    # 1280x720 scales by 0.8, so [100, 200, 300, 400] lands at [80, 160, 240, 320]
    dataset = build_dataset([{"id": 1, "image_id": 0, "category_id": 1, "bbox": [100, 200, 200, 200]}])

    boxes = dataset[0]["boxes"]

    print(f"coco xywh [100, 200, 200, 200]: cxcywh {boxes[0].round(4).tolist()}")
    assert boxes[0] == pytest.approx([160 / 1024, 240 / 576, 160 / 1024, 160 / 576])


def test_labels_come_from_the_category_map():
    dataset = build_dataset(
        [
            {"id": 1, "image_id": 0, "category_id": 2, "bbox": [10, 10, 50, 50]},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [80, 80, 50, 50]},
        ]
    )

    print(f"category ids [2, 1]: labels {dataset[0]['class_labels'].tolist()}")
    assert dataset[0]["class_labels"].tolist() == [1, 0]


def test_image_without_annotations_yields_no_boxes():
    dataset = build_dataset([])

    sample = dataset[0]

    print(f"no annotations: boxes {sample['boxes'].shape}, labels {sample['class_labels'].shape}")
    assert sample["boxes"].shape == (0, 4)
    assert sample["class_labels"].shape == (0,)


def test_zero_area_box_raises():
    dataset = build_dataset([{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 0, 50]}])

    with pytest.raises(ValueError, match="zero-area"):
        dataset[0]


def test_box_outside_the_frame_is_clipped():
    dataset = build_dataset([{"id": 1, "image_id": 0, "category_id": 1, "bbox": [1200, 600, 400, 400]}])

    boxes = dataset[0]["boxes"]

    print(f"box past the edge: cxcywh {boxes[0].round(4).tolist()}")
    assert boxes.max() <= 1.0


def test_collate_stacks_pixels_and_keeps_labels_per_image():
    dataset = build_dataset(
        [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [90, 90, 50, 50]},
        ],
        images=2,
    )

    batch = collate([dataset[0], dataset[1]])

    print(f"pixel_values {tuple(batch['pixel_values'].shape)}, labels {[len(l['class_labels']) for l in batch['labels']]}")
    assert batch["pixel_values"].shape == (2, 3, 576, 1024)
    assert batch["pixel_values"].dtype == torch.float32
    assert [len(label["class_labels"]) for label in batch["labels"]] == [1, 2]
    assert batch["labels"][0]["boxes"].dtype == torch.float32


def test_reordered_category_ids_map_by_position_not_value():
    # ids 4 and 9 in a different order than the label space they define
    dataset = build_dataset(
        [
            {"id": 1, "image_id": 0, "category_id": 9, "bbox": [10, 10, 50, 50]},
            {"id": 2, "image_id": 0, "category_id": 4, "bbox": [80, 80, 50, 50]},
        ],
        categories=[{"id": 9, "name": "scissors"}, {"id": 4, "name": "scalpel"}],
    )

    print(f"category_map {dataset.category_map}  id2label {dataset.id2label}")
    assert dataset.category_map == {4: 0, 9: 1}
    assert dataset.id2label == {0: "scalpel", 1: "scissors"}
    assert dataset[0]["class_labels"].tolist() == [1, 0]


def test_canvas_stays_uint8_until_collate():
    dataset = build_dataset([{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50]}])

    assert dataset[0]["canvas"].dtype == np.uint8


def test_unit_dataset():
    for test in [
        test_boxes_become_normalized_cxcywh_on_the_canvas,
        test_labels_come_from_the_category_map,
        test_image_without_annotations_yields_no_boxes,
        test_zero_area_box_raises,
        test_box_outside_the_frame_is_clipped,
        test_collate_stacks_pixels_and_keeps_labels_per_image,
        test_reordered_category_ids_map_by_position_not_value,
        test_canvas_stays_uint8_until_collate,
    ]:
        print(f"\n{test.__name__}")
        test()


if __name__ == "__main__":
    test_unit_dataset()
    print("\nall passed")
