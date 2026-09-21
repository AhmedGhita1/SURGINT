import argparse
import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from surgint.dataset.coco import SurgintDataset
from surgint.model import Detections
from surgint.runtime.pipeline import Pipeline

DATA_ROOT = Path("data/synthetic/production_v5/dataset")
OUTPUT = Path("outputs/videos")

# golden-ratio hue stepping, so consecutive track ids never land on the same color
GOLDEN_RATIO = 0.618033988749895


def track_color(track_id: int) -> Tuple[int, int, int]:
    """a stable color for a track id"""
    hue = (track_id * GOLDEN_RATIO) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def load_font(size: int) -> ImageFont.ImageFont:
    """the default bitmap font at a usable size, falling back to the smallest one available"""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(
    canvas: ImageDraw.ImageDraw,
    detections: Detections,
    labels: List[str],
    font: ImageFont.ImageFont,
    width: int,
) -> None:
    """one box per track, colored by id and labelled with id, class and score"""
    track_ids = detections.track_ids
    if track_ids is None:
        track_ids = np.zeros(len(detections.boxes), dtype=np.int64)

    for box, score, class_id, track_id in zip(
        detections.boxes, detections.scores, detections.class_ids, track_ids
    ):
        color = track_color(int(track_id))
        canvas.rectangle(box.tolist(), outline=color, width=width)

        caption = f"{int(track_id)} {labels[int(class_id)]} {score:.2f}"
        left, top = float(box[0]), float(box[1])
        text_box = canvas.textbbox((left, top), caption, font=font)
        text_width = text_box[2] - text_box[0] + 6
        height = text_box[3] - text_box[1]

        # the label sits above the box, or inside it when the box touches the top edge,
        # and is pulled back from the right edge so the caption is never cut off
        anchor = top - height - 4 if top - height - 4 >= 0 else top + 2
        left = min(left, canvas.im.size[0] - text_width)
        canvas.rectangle((left, anchor, left + text_width, anchor + height + 4), fill=color)
        canvas.text((left + 3, anchor + 2), caption, fill=(0, 0, 0), font=font)


def instrument_pipeline(pipeline: Pipeline) -> Dict[str, int]:
    """
    count the raw detections reaching the tracker.

    the tracker decodes at low_thresh, so the boxes clearing high_thresh are the
    detector's operating point. comparing that count against the ground truth count
    separates a detector problem from a tracker problem.
    """
    counts = {"low": 0, "high": 0}
    tracker = pipeline.tracker
    update = tracker.update

    def counting_update(detections: Detections) -> Detections:
        scores = np.asarray(detections.scores, dtype=float)
        counts["low"] = int(len(scores))
        counts["high"] = int((scores >= tracker.high_thresh).sum())
        return update(detections)

    tracker.update = counting_update
    return counts


def render(
    pipeline: Pipeline,
    dataset: SurgintDataset,
    destination: Path,
    fps: int,
    scale: float,
    limit: Optional[int],
) -> Dict:
    """write one session to an mp4 and return the counts behind it"""
    import imageio.v2 as imageio

    labels = pipeline.detector.meta["labels"]
    counts = instrument_pipeline(pipeline)
    pipeline.reset()

    frames = len(dataset) if limit is None else min(limit, len(dataset))
    totals = {"gt": 0, "predictions": 0, "raw_high": 0, "raw_low": 0}

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        destination, fps=fps, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    progress = tqdm(range(frames), desc=dataset.split.name, unit="frame", leave=False)
    try:
        for index in progress:
            sample = dataset[index]
            frame = sample["frame"]
            detections = pipeline.predict(frame, 0.0)

            width = frame.shape[1]
            line = max(2, round(width / 640))
            font = load_font(max(13, round(width / 60)))

            output = Image.fromarray(frame.copy())
            draw_boxes(ImageDraw.Draw(output), detections, labels, font, line)

            gt_count = len(np.asarray(sample["boxes"]).reshape(-1, 4))
            prediction_count = len(detections.boxes)
            totals["gt"] += gt_count
            totals["predictions"] += prediction_count
            totals["raw_high"] += counts["high"]
            totals["raw_low"] += counts["low"]

            if scale != 1.0:
                # even dimensions, which yuv420p requires
                size = (
                    max(2, round(output.width * scale) // 2 * 2),
                    max(2, round(output.height * scale) // 2 * 2),
                )
                output = output.resize(size, Image.BILINEAR)

            writer.append_data(np.asarray(output))
            progress.set_postfix(gt=gt_count, tracked=prediction_count, raw=counts["high"])
    finally:
        progress.close()
        writer.close()

    return {"frames": frames, **totals}


def main():
    parser = argparse.ArgumentParser(description="render one eval session to an annotated mp4")
    parser.add_argument("session", help="a session name, for example session_000")
    parser.add_argument("checkpoint", type=Path, help="a surgint checkpoint directory")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split", default="eval")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cuda or cpu; autodetected by default")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="render only the first N frames")

    # tracker thresholds, so the operating point can be changed without editing the config
    parser.add_argument("--high-thresh", type=float, default=None)
    parser.add_argument("--low-thresh", type=float, default=None)
    parser.add_argument("--match-thresh", type=float, default=None)
    parser.add_argument("--track-buffer", type=int, default=None)
    parser.add_argument("--min-box-area", type=float, default=None)
    args = parser.parse_args()

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = Pipeline.from_checkpoint(args.checkpoint, "detection-tracking", args.device)

    overrides = {
        "high_thresh": args.high_thresh,
        "low_thresh": args.low_thresh,
        "match_thresh": args.match_thresh,
        "track_buffer": args.track_buffer,
        "min_box_area": args.min_box_area,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(pipeline.tracker, name, value)

    dataset = SurgintDataset(
        args.data_root, f"{args.split}/{args.session}", "detection-tracking"
    )

    destination = args.output or OUTPUT / f"{args.session}_{args.checkpoint.name}.mp4"
    print(f"{args.session}: {len(dataset)} frames on {args.device}, writing {destination}")

    counts = render(pipeline, dataset, destination, args.fps, args.scale, args.limit)

    frames = counts["frames"]
    print(f"\n{frames} frames")
    print(f"  ground truth per frame      {counts['gt'] / frames:.1f}")
    print(f"  tracked per frame           {counts['predictions'] / frames:.1f}")
    print(f"  raw detections per frame    {counts['raw_high'] / frames:.1f} "
          f"at >= {pipeline.tracker.high_thresh:.2f}, "
          f"{counts['raw_low'] / frames:.1f} at >= {pipeline.tracker.low_thresh:.2f}")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
