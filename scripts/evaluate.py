import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from surgint.artifacts import RunWriter
from surgint.config import load_config
from surgint.dataset.coco import CocoDetection
from surgint.evaluation.detection import coco_predictions, evaluate
from surgint.inference.detector import InferencePipeline

DEVICE = "cuda"
CONFIG = Path("configs/evaluate.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="a SURGINT checkpoint directory containing weights and meta.yaml",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    config.checkpoint = str(args.checkpoint)
    pipeline = InferencePipeline(args.checkpoint, device=DEVICE)
    config.input_size = [pipeline.width, pipeline.height]
    root = Path(config.data_root)
    data = CocoDetection(root, config.split, config.input_size)
    categories = [data.id2label[index] for index in sorted(data.id2label)]
    labels = [pipeline.id2label[index] for index in sorted(pipeline.id2label)]
    if labels != categories:
        raise ValueError(f"checkpoint labels {labels} do not match dataset categories {categories}")

    to_category = {label: category for category, label in data.category_map.items()}
    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    writer.initialize(config, categories, str(args.checkpoint))

    predictions = []
    for index, (image_id, file_name, _, _) in enumerate(data.samples, start=1):
        frame = np.asarray(Image.open(data.images / file_name).convert("RGB"))
        predictions += coco_predictions(image_id, pipeline.predict(frame, 0.0), to_category)
        if index % 100 == 0:
            print(f"{index}/{len(data)} frames")
            writer.append_log({"frames": index})

    metrics = evaluate(data.annotations, predictions)
    writer.append_log({"frames": len(data), "metrics": metrics})
    writer.write_summary(
        {
            "run_id": config.run_id,
            "mode": "eval",
            "checkpoint": str(args.checkpoint),
            "split": config.split,
            "frames": len(data),
            "metrics": metrics,
        }
    )

    print(f"\n{config.run_id} {args.checkpoint.name} on {config.split}, {len(data)} frames")
    print(f"  mAP50_95  {metrics['mAP50_95']:.4f}")
    print(f"  mAP50     {metrics['mAP50']:.4f}")
    print(f"  mAP75     {metrics['mAP75']:.4f}")
    for name, value in sorted(metrics["per_class"].items(), key=lambda item: -item[1]):
        print(f"    {name:<12} {value:.3f}")
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
