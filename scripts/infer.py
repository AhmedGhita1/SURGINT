# a smoke entrpoint.

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from surgint.inference import DETOutput
from surgint.inference.detector import InferencePipeline

BOX_COLOR = (255, 64, 0)
DEFAULT_CHECKPOINT = "PekingU/rtdetr_r18vd_coco_o365"


def draw_detections(frame: np.ndarray, result: DETOutput, id2label: dict[int, str]) -> Image.Image:
    overlay = Image.fromarray(frame)
    canvas = ImageDraw.Draw(overlay)
    for box, score, class_id in zip(result.boxes, result.scores, result.class_ids):
        canvas.rectangle(box.tolist(), outline=BOX_COLOR, width=3)
        label = f"{id2label[int(class_id)]} {score:.2f}"
        canvas.text((box[0] + 4, max(box[1] - 12, 0)), label, fill=BOX_COLOR)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        help="required for external checkpoints; SURGINT artifacts use meta.yaml",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/infer"))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_size = args.input_size
    if input_size is None and args.checkpoint == DEFAULT_CHECKPOINT:
        input_size = [1024, 576]

    frame = np.asarray(Image.open(args.image).convert("RGB"))
    pipeline = InferencePipeline(args.checkpoint, input_size, args.device)
    result = pipeline.predict(frame, args.threshold)

    for box, score, class_id in zip(result.boxes, result.scores, result.class_ids):
        print(f"{pipeline.id2label[int(class_id)]:>16}  {score:.2f}  {np.round(box, 1).tolist()}")

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.image.stem}_detections.png"
    draw_detections(frame, result, pipeline.id2label).save(destination)
    print(f"\n{len(result.boxes)} detections at threshold {args.threshold}, wrote {destination}")


if __name__ == "__main__":
    main()
