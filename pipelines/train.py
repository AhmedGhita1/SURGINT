import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch

from surgint.artifacts import RunWriter
from surgint.config import Config, load_config
from surgint.dataset.coco import SurgintDataset, build_loader
from surgint.model.transform import Transform
from surgint.engine.trainer import Trainer
from surgint.model.detector import Detector

from pipelines.wandb_tracking import (
    DEFAULT_MODEL_ARTIFACT,
    init_training_run,
    log_candidate_model,
    log_epoch,
)

CONFIG = Path("configs/train.yaml")
DEVICE = "cuda"

def train(
    config: Config,
    device: str,
    wandb_run=None,
    artifact_name: str = DEFAULT_MODEL_ARTIFACT,
) -> Dict:
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
        config.revision,
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
    writer.initialize(
        config,
        categories,
        config.checkpoint,
        detector.revision,
        [train_set.annotations, val_set.annotations],
    )

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
            detector.save_checkpoint(
                run_dir / "best",
                meta,
                training_run=config.run_id,
                dataset_version=config.dataset_id,
            )

        detector.save_checkpoint(
            run_dir / "latest",
            meta,
            training_run=config.run_id,
            dataset_version=config.dataset_id,
        )
        trainer.save(run_dir / "latest")
        record = result.as_dict()
        writer.append_log(record)
        if wandb_run is not None:
            log_epoch(wandb_run, record, result.epoch)

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
    if wandb_run is not None:
        log_candidate_model(
            wandb_run,
            run_dir / "best",
            artifact_name,
            config.run_id,
        )
    print(f"wrote {run_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--artifact-name", default=DEFAULT_MODEL_ARTIFACT)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.wandb_project is None:
        train(config, args.device)
        return

    run = init_training_run(
        args.wandb_project,
        args.wandb_entity,
        config.run_id,
        asdict(config),
    )
    with run:
        train(config, args.device, run, args.artifact_name)


if __name__ == "__main__":
    main()
