import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from surgint.artifacts import RunWriter
from surgint.config import load_config
from surgint.evaluation.recall import count_matches
from surgint.inference.detector import InferencePipeline

CONFIG = Path("configs/zero_shot.yaml")
DEVICE = "cuda"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def load_ground_truth(
    annotations: Path,
) -> tuple[dict[int, str], dict[int, np.ndarray], list[str]]:
    coco = json.loads(annotations.read_text())
    file_names = {image["id"]: image["file_name"] for image in coco["images"]}
    categories = [
        category["name"] for category in sorted(coco["categories"], key=lambda item: item["id"])
    ]

    boxes = defaultdict(list)
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        boxes[annotation["image_id"]].append([x, y, x + width, y + height])

    return file_names, {image_id: np.array(values) for image_id, values in boxes.items()}, categories


def main():
    config = load_config(CONFIG)
    data_root = Path(config.data_root)

    file_names, ground_truth, categories = load_ground_truth(
        data_root / "annotations" / f"instances_{config.split}.json"
    )
    pipeline = InferencePipeline(config.checkpoint, config.input_size, DEVICE)
    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    writer.initialize(config, categories, config.checkpoint)

    instruments = 0
    matches = defaultdict(int)
    detections = defaultdict(int)

    for index, (image_id, file_name) in enumerate(sorted(file_names.items()), start=1):
        truth = ground_truth.get(image_id, np.empty((0, 4)))
        frame = np.asarray(Image.open(data_root / config.split / file_name).convert("RGB"))

        # one forward pass; every threshold is a filter on the same scored detections
        result = pipeline.predict(frame, 0.0)
        instruments += len(truth)
        for threshold in THRESHOLDS:
            kept = result.boxes[result.scores >= threshold]
            matches[threshold] += count_matches(kept, truth, config.iou_threshold)
            detections[threshold] += len(kept)

        if index % 50 == 0:
            print(f"{index}/{len(file_names)} frames")
            writer.append_log({"frames": index})

    frames = len(file_names)
    metrics = {
        "iou_threshold": config.iou_threshold,
        "class_agnostic": True,
        "by_score_threshold": {
            f"{threshold:.1f}": {
                "recall": round(matches[threshold] / instruments, 4),
                "detections_per_frame": round(detections[threshold] / frames, 2),
            }
            for threshold in THRESHOLDS
        },
    }
    report = {
        "run_id": config.run_id,
        "mode": "eval",
        "checkpoint": config.checkpoint,
        "split": config.split,
        "frames": frames,
        "instruments": instruments,
        "metrics": metrics,
    }
    writer.append_log({"frames": frames, "metrics": metrics})
    writer.write_summary(report)

    print(f"\n{frames} frames, {instruments} instruments, recall at IoU {config.iou_threshold}\n")
    print(f"{'score':>6}  {'recall':>7}  {'det/frame':>10}")
    for threshold, values in metrics["by_score_threshold"].items():
        print(f"{threshold:>6}  {values['recall']:>7.3f}  {values['detections_per_frame']:>10.2f}")

    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
