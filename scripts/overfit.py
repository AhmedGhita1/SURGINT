import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from surgint.artifacts import RunWriter, build_checkpoint_meta
from surgint.config import load_config
from surgint.dataset.coco import CocoDetection, collate
from surgint.detection.model import load_model_with_new_head
from surgint.detection.postprocessing import decode, to_frame_boxes
from surgint.evaluation import MetricsFn
from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions
from surgint.evaluation.metrics import count_matches
from surgint.inference import DETOutput
from surgint.inference.detector import InferencePipeline
from surgint.training.trainer import Trainer

CONFIG = Path("configs/overfit20.yaml")
DEVICE = "cuda"
THRESHOLDS = [0.1, 0.3, 0.5, 0.7]


def layout_subset(dataset: CocoDetection, layouts: int, seed: int = 0) -> list[int]:
    """every view of a few layouts; splitting by view would leak the same arrangement"""
    layout_ids = [image["layout_id"] for image in dataset.annotations["images"]]
    chosen = set(random.Random(seed).sample(sorted(set(layout_ids)), layouts))
    return [index for index, layout in enumerate(layout_ids) if layout in chosen]


def build_overfit_metrics_fn(dataset: CocoDetection, loader: DataLoader, indices: list[int]) -> MetricsFn:
    """COCO mAP over only the frames used by this overfit gate."""
    image_ids = {dataset.samples[index][0] for index in indices}
    annotations = {
        **dataset.annotations,
        "images": [image for image in dataset.annotations["images"] if image["id"] in image_ids],
        "annotations": [
            annotation
            for annotation in dataset.annotations["annotations"]
            if annotation["image_id"] in image_ids
        ],
    }
    to_category = {label: category for category, label in dataset.category_map.items()}

    @torch.inference_mode()
    def metrics_fn(model: torch.nn.Module) -> dict:
        model.eval()
        device = next(model.parameters()).device
        predictions = []

        for batch in loader:
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            detections = decode(
                outputs.logits.cpu(),
                outputs.pred_boxes.cpu(),
                dataset.width,
                dataset.height,
                0.0,
            )
            for (boxes, scores, class_ids), image_id, scale, frame_size in zip(
                detections,
                batch["image_ids"],
                batch["scales"],
                batch["frame_sizes"],
            ):
                result = DETOutput(to_frame_boxes(boxes, scale, frame_size), scores, class_ids)
                predictions += coco_predictions(image_id, result, to_category)

        return coco_evaluate(annotations, predictions)

    return metrics_fn


def report_recall(checkpoint: Path, dataset: CocoDetection, indices: list[int]) -> None:
    """the real check: the model must find what it memorized"""
    pipeline = InferencePipeline(checkpoint, device=DEVICE)
    instruments = 0
    matches = {threshold: 0 for threshold in THRESHOLDS}

    for index in indices:
        _, file_name, boxes, _ = dataset.samples[index]
        frame = np.asarray(Image.open(dataset.images / file_name).convert("RGB"))
        result = pipeline.predict(frame, 0.0)
        instruments += len(boxes)
        for threshold in THRESHOLDS:
            matches[threshold] += count_matches(result.boxes[result.scores >= threshold], boxes)

    print(f"\n{len(indices)} frames, {instruments} instruments, recall on its own training data")
    for threshold in THRESHOLDS:
        print(f"  score {threshold}: {matches[threshold] / instruments:.3f}")


def main():
    config = load_config(CONFIG)
    torch.manual_seed(config.seed)

    data = CocoDetection(config.data_root, "train", config.input_size)
    indices = layout_subset(data, config.layouts, config.seed)
    subset = Subset(data, indices)
    loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    score_loader = DataLoader(subset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    model = load_model_with_new_head(config.checkpoint, data.id2label)
    metrics_fn = build_overfit_metrics_fn(data, score_loader, indices)
    trainer = Trainer(model, config, metrics_fn, DEVICE)
    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    categories = [data.id2label[index] for index in sorted(data.id2label)]
    meta = build_checkpoint_meta(config.input_size, data.id2label)
    writer.initialize(config, categories, config.checkpoint)

    print(f"run {config.run_id}")
    print(f"{config.layouts} layouts, {len(subset)} frames, {config.epochs} epochs, lr {config.learning_rate}")

    for result in trainer.train(loader):
        if result.is_best:
            writer.save_checkpoint("best", model, meta)
        writer.save_checkpoint("latest", model, meta, trainer.state_dict())
        writer.append_log(result.as_dict())

        line = (
            f"epoch [{result.epoch}/{config.epochs}]  lr {result.lr:.2e}"
            f"  train_loss {result.train_loss:.4f}"
        )
        line += "".join(f"  {name} {result.metrics[name]:.4f}" for name in config.metrics)
        print(line + ("  best" if result.is_best else ""), flush=True)

    writer.write_summary(
        {
            "run_id": config.run_id,
            "mode": "train",
            "epochs": trainer.epoch,
            "best": {"epoch": trainer.best_epoch, **trainer.best_metrics},
            "checkpoints": {
                "best": str((run_dir / "best").resolve()),
                "latest": str((run_dir / "latest").resolve()),
            },
        }
    )
    report_recall(run_dir / "best", data, indices)
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
