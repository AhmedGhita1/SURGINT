from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surgint.artifacts import RunWriter, build_checkpoint_meta
from surgint.config import load_config
from surgint.dataset.coco import CocoDetection, collate
from surgint.detection.model import load_model_with_new_head
from surgint.evaluation.metrics import build_metrics_fn
from surgint.training.trainer import Trainer

CONFIG = Path("configs/train.yaml")
DEVICE = "cuda"


def build_loader(dataset: CocoDetection, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


def main():
    config = load_config(CONFIG)
    torch.manual_seed(config.seed)

    train = CocoDetection(config.data_root, "train", config.input_size)
    val = CocoDetection(config.data_root, config.split, config.input_size)
    if train.id2label != val.id2label:
        raise ValueError("training and validation categories do not match")

    train_loader = build_loader(train, config.batch_size, shuffle=True)
    val_loader = build_loader(val, config.batch_size, shuffle=False)

    model = load_model_with_new_head(config.checkpoint, train.id2label)
    trainer = Trainer(model, config, build_metrics_fn(val, val_loader), DEVICE)
    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    categories = [train.id2label[index] for index in sorted(train.id2label)]
    meta = build_checkpoint_meta(config.input_size, train.id2label)
    writer.initialize(config, categories, config.checkpoint)

    print(f"run {config.run_id}")
    print(f"{len(train)} train, {len(val)} val, {config.epochs} epochs, batch {config.batch_size}")

    for result in trainer.train(train_loader):
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
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
