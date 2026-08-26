import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from surgint.config import load_config, save_config
from surgint.dataset.coco import CocoDetection, collate
from surgint.detection.model import load_model, load_model_with_new_head
from surgint.evaluation.recall import count_matches
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


def report_recall(run_dir: Path, dataset: CocoDetection, indices: list[int]) -> None:
    """the real check: the model must find what it memorized"""
    pipeline = InferencePipeline(run_dir, dataset_input_size(dataset), DEVICE)
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


def dataset_input_size(dataset: CocoDetection) -> list[int]:
    return [dataset.width, dataset.height]


def main():
    config = load_config(CONFIG)
    torch.manual_seed(config.seed)

    data = CocoDetection(config.data_root, "train", config.input_size)
    indices = layout_subset(data, config.layouts, config.seed)
    subset = Subset(data, indices)
    loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)

    model = load_model_with_new_head(config.checkpoint, data.id2label)
    trainer = Trainer(model, config, DEVICE)
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, trainer.run_dir / "config.yaml")

    print(f"run {config.run_id}")
    print(f"{config.layouts} layouts, {len(subset)} frames, {config.epochs} epochs, lr {config.learning_rate}")

    trainer.train(loader, loader)
    report_recall(trainer.run_dir / "best", data, indices)
    print(f"\nwrote {trainer.run_dir}")


if __name__ == "__main__":
    main()
