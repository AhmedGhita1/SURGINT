import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from surgint.config import load_config
from surgint.dataset.coco import CocoDetection
from surgint.evaluation.detection import coco_predictions, evaluate
from surgint.inference.detector import InferencePipeline

DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="a checkpoint directory, best/ or latest/, holding weights, manifest.json, and config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.checkpoint / "config.yaml")
    root = Path(config.data_root)

    data = CocoDetection(root, config.split, config.input_size)

    to_category = {label: category for category, label in data.category_map.items()}

    pipeline = InferencePipeline(args.checkpoint, device=DEVICE)
    predictions = []
    for index, (image_id, file_name, _, _) in enumerate(data.samples, start=1):
        frame = np.asarray(Image.open(data.images / file_name).convert("RGB"))
        predictions += coco_predictions(image_id, pipeline.predict(frame, 0.0), to_category)
        if index % 100 == 0:
            print(f"{index}/{len(data)} frames")

    metrics = evaluate(data.annotations, predictions)
    metrics["run_id"] = config.run_id
    metrics["checkpoint"] = args.checkpoint.name
    metrics["split"] = config.split
    metrics["frames"] = len(data)

    print(f"\n{config.run_id} {args.checkpoint.name} on {config.split}, {len(data)} frames")
    print(f"  mAP50_95  {metrics['mAP50_95']:.4f}")
    print(f"  mAP50     {metrics['mAP50']:.4f}")
    print(f"  mAP75     {metrics['mAP75']:.4f}")
    for name, value in sorted(metrics["per_class"].items(), key=lambda item: -item[1]):
        print(f"    {name:<12} {value:.3f}")

    destination = args.checkpoint / "metrics.json"
    destination.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
