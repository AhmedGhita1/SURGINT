import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


TASKS = ("detection-only", "detection-tracking")


class SurgintDataset(Dataset):
    def __init__(
            self, root: str | Path,
            split: str,
            task: str,
            transform=None,
            ):
        
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")

        self.root = Path(root)
        self.split = Path(split)


        self.task = task
        self.transform = transform
        tracking = task == "detection-tracking"

        annos_path = self.root / "annotations" / self.split.parent / f"instances_{self.split.name}.json"
        self.raw = json.loads(annos_path.read_text())

        classes = sorted(self.raw["categories"], key=lambda category: category["id"])

        # map the contiguous class ids to (category id, category name)
        self.mappings = {
            class_id: (category["id"], category["name"])
            for class_id, category in enumerate(classes)
        }
        category2class = {class_name: class_id for class_id, (class_name, _) in self.mappings.items()}

        boxes = {image["id"]: [] for image in self.raw["images"]}
        classes = {image["id"]: [] for image in self.raw["images"]}
        tracks = {image["id"]: [] for image in self.raw["images"]}
        
        for annotation in self.raw["annotations"]:
            x, y, width, height = annotation["bbox"]
            boxes[annotation["image_id"]].append([x, y, x + width, y + height])

            classes[annotation["image_id"]].append(category2class[annotation["category_id"]])

            if tracking:
                if "track_id" not in annotation:
                    raise ValueError(f"{annos_path.name} has no track_id; {task} requires it")
                tracks[annotation["image_id"]].append(annotation["track_id"])

        self.samples = [
            (
                image["id"],
                image["file_name"],
                np.array(boxes[image["id"]], dtype=np.float32).reshape(-1, 4),
                np.array(classes[image["id"]], dtype=np.int64),
                np.array(tracks[image["id"]], dtype=np.int64) if tracking else None,
            )
            for image in self.raw["images"]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_id, file_name, boxes, class_ids, track_ids = self.samples[index]
        frame = np.asarray(Image.open(self.root / self.split / file_name).convert("RGB"))

        if self.transform is None:
            sample = {"frame": frame, "boxes": boxes, "class_ids": class_ids}
        else:
            sample = self.transform(frame, boxes, class_ids)

        sample["image_id"] = image_id
        if track_ids is not None:
            sample["track_ids"] = track_ids

        return sample


def collate(batch: list[dict]) -> dict:
    labels = [
        {
            "class_labels": torch.as_tensor(sample["class_ids"], dtype=torch.int64),
            "boxes": torch.as_tensor(sample["boxes"], dtype=torch.float32),
        }
        for sample in batch
    ]

    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in batch]),
        "image_ids": [sample["image_id"] for sample in batch],
        "scales": [sample["scale"] for sample in batch],
        "frame_sizes": [sample["frame_size"] for sample in batch],
        "labels": labels,
    }
