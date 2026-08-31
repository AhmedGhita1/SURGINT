import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from surgint.artifacts import RunWriter
from surgint.config import Config, load_config
from surgint.dataset.coco import SurgintDataset, collate
from surgint.model.transform import Transform
from surgint.engine.trainer import Trainer
from surgint.model.detector import Detector

CONFIG = Path("configs/train.yaml")
DEVICE = "cuda"


def build_loader(dataset: SurgintDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


def train(config: Config, device: str) -> Dict:
    torch.manual_seed(config.seed)

    transform = Transform(config.input_size)
    train_split, val_split = config.splits
    train_set = SurgintDataset(config.data_root, train_split, "detection-only", transform)
    val_set = SurgintDataset(config.data_root, val_split, "detection-only", transform)
    if train_set.mappings != val_set.mappings:
        raise ValueError("train and val categories do not match")

    detector = Detector.from_pretrained(
        config.checkpoint,
        {class_id: name for class_id, (_, name) in train_set.mappings.items()},
    ).to(device)

    trainer = Trainer(
        detector,
        config,
        build_loader(train_set, config.batch_size, shuffle=True),
        build_loader(val_set, config.batch_size, shuffle=False),
        transform,
    )

    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    categories = [name for _, name in train_set.mappings.values()]
    writer.initialize(config, categories, config.checkpoint)

    meta = {
        "input_size": config.input_size,
        "pad_color": transform.pad_color,
        "rescale_factor": transform.rescale_factor,
        "source": config.checkpoint,
    }

    print(f"run {config.run_id}")
    print(f"{len(train_set)} train, {len(val_set)} val, {config.epochs} epochs, batch {config.batch_size}")

    for result in trainer.train():
        if result.is_best:
            detector.save_checkpoint(run_dir / "best", meta)

        detector.save_checkpoint(run_dir / "latest", meta)
        trainer.save(run_dir / "latest")
        writer.append_log(result.as_dict())

        line = f"epoch [{result.epoch}/{config.epochs}]"
        line += "".join(f"  {name} {value:.4f}" for name, value in result.train.items())
        line += "".join(f"  {name} {result.val[name]:.4f}" for name in config.metrics if result.val)
        print(line + ("  best" if result.is_best else ""), flush=True)

    summary = {
        "run_id": config.run_id,
        "epochs": trainer.epoch,
        "best": {"epoch": trainer.best_epoch, **trainer.best_val},
        "checkpoints": {
            "best": str((run_dir / "best").resolve()),
            "latest": str((run_dir / "latest").resolve()),
        },
    }
    writer.write_summary(summary)
    print(f"wrote {run_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    train(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
