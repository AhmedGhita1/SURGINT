from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surgint.config import load_config, save_config
from surgint.dataset.coco import CocoDetection, collate
from surgint.detection.model import load_model_with_new_head
from surgint.training.trainer import Trainer

CONFIG = Path("configs/train.yaml")
DEVICE = "cuda"
NUM_WORKERS = 4


def build_loader(dataset: CocoDetection, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )


def main():
    config = load_config(CONFIG)
    torch.manual_seed(config.seed)

    train = CocoDetection(config.data_root, "train", config.input_size)
    val = CocoDetection(config.data_root, config.split, config.input_size)

    model = load_model_with_new_head(config.checkpoint, train.id2label)
    trainer = Trainer(model, config, DEVICE)
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, trainer.run_dir / "config.yaml")

    print(f"run {config.run_id}")
    print(f"{len(train)} train, {len(val)} val, {config.epochs} epochs, batch {config.batch_size}")

    trainer.train(
        build_loader(train, config.batch_size, shuffle=True),
        build_loader(val, config.batch_size, shuffle=False),
    )
    print(f"wrote {trainer.run_dir}")


if __name__ == "__main__":
    main()
