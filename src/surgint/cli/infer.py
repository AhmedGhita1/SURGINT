import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from surgint.model import Detections
from surgint.runtime.pipeline import Pipeline

BOX_COLOR = (255, 64, 0)


def draw(frame: np.ndarray, detections: Detections, labels: list) -> Image.Image:
    overlay = Image.fromarray(frame)
    canvas = ImageDraw.Draw(overlay)

    for box, score, class_id in zip(detections.boxes, detections.scores, detections.class_ids):
        canvas.rectangle(box.tolist(), outline=BOX_COLOR, width=3)
        canvas.text((box[0] + 4, max(box[1] - 12, 0)), f"{labels[class_id]} {score:.2f}", fill=BOX_COLOR)

    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("checkpoint", type=Path, help="a surgint checkpoint directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/infer"))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pipeline = Pipeline.from_checkpoint(args.checkpoint, "detection-only", args.device)
    labels = pipeline.detector.meta["labels"]

    frame = np.asarray(Image.open(args.image).convert("RGB"))
    detections = pipeline.predict(frame, args.threshold)

    for box, score, class_id in zip(detections.boxes, detections.scores, detections.class_ids):
        print(f"{labels[class_id]:>16}  {score:.2f}  {np.round(box, 1).tolist()}")

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.image.stem}_detections.png"
    draw(frame, detections, labels).save(destination)
    print(f"\n{len(detections.boxes)} detections at threshold {args.threshold}, wrote {destination}")


if __name__ == "__main__":
    main()
