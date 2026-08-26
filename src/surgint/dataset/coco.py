import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from surgint.dataset.labels import category_to_label, label_maps, load_classes
from surgint.detection.preprocessing import (
    letterbox,
    letterbox_boxes,
    to_normalized_cxcywh,
    to_pixel_values,
)


class CocoDetection(Dataset):
    def __init__(self, data_root: str | Path, split: str, input_size: list[int]):
        root = Path(data_root)
        self.images = root / split
        self.width, self.height = input_size

        annotations = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
        self.id2label, label2id = label_maps(load_classes(root / "classes.txt"))
        category_map = category_to_label(annotations["categories"], label2id)

        boxes: dict[int, list] = {image["id"]: [] for image in annotations["images"]}
        labels: dict[int, list] = {image["id"]: [] for image in annotations["images"]}
        for annotation in annotations["annotations"]:
            x, y, width, height = annotation["bbox"]
            boxes[annotation["image_id"]].append([x, y, x + width, y + height])
            labels[annotation["image_id"]].append(category_map[annotation["category_id"]])

        self.samples = [
            (
                image["id"],
                image["file_name"],
                np.array(boxes[image["id"]], dtype=float).reshape(-1, 4),
                np.array(labels[image["id"]], dtype=np.int64),
            )
            for image in annotations["images"]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_id, file_name, boxes, class_labels = self.samples[index]
        frame = np.asarray(Image.open(self.images / file_name).convert("RGB"))

        canvas, scale = letterbox(frame, self.width, self.height)
        boxes = letterbox_boxes(boxes, scale)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, self.width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, self.height)

        degenerate = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
        if degenerate.any():
            raise ValueError(f"{file_name} has {degenerate.sum()} zero-area boxes after letterbox")

        return {
            "image_id": image_id,
            "canvas": canvas,
            "boxes": to_normalized_cxcywh(boxes, self.width, self.height),
            "class_labels": class_labels,
        }


def collate(batch: list[dict]) -> dict:
    """uint8 canvases become one float batch here, so workers ship the smaller array"""
    return {
        "pixel_values": to_pixel_values([sample["canvas"] for sample in batch]),
        "labels": [
            {
                "class_labels": torch.from_numpy(sample["class_labels"]),
                "boxes": torch.from_numpy(sample["boxes"]).float(),
            }
            for sample in batch
        ],
    }
