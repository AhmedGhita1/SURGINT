import argparse
from pathlib import Path
from typing import Dict, List

from surgint.artifacts import RunWriter
from surgint.cli.train import build_loader
from surgint.config import Config, load_config
from surgint.dataset.coco import SurgintDataset
from surgint.model.transform import Transform
from surgint.engine.evaluator import evaluate, evaluate_sessions
from surgint.model.detector import Detector
from surgint.runtime.pipeline import Pipeline

CONFIG = Path("configs/eval.yaml")
DEVICE = "cuda"


def categories_of(meta: Dict, dataset: SurgintDataset) -> List[str]:
    """the dataset categories."""
    categories = [name for _, name in dataset.mappings.values()]
    if meta["labels"] != categories:
        raise ValueError(f"checkpoint labels {meta['labels']} do not match categories {categories}")
    return categories


def run_detection(config: Config, checkpoint: Path, device: str) -> Dict:
    """score the detector on one split."""
    detector = Detector.from_checkpoint(checkpoint).to(device)
    meta = detector.meta

    transform = Transform(meta["input_size"], meta["pad_color"], meta["rescale_factor"])
    dataset = SurgintDataset(config.data_root, config.split, config.task, transform)
    categories = categories_of(meta, dataset)

    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    writer.initialize(config, categories, str(checkpoint), annotations=[dataset.annotations])

    metrics = evaluate(detector, build_loader(dataset, config.batch_size, shuffle=False), transform)

    summary = {
        "run_id": config.run_id,
        "checkpoint": str(checkpoint),
        "task": config.task,
        "split": config.split,
        "frames": len(dataset),
        "metrics": metrics,
    }
    writer.append_log({"frames": len(dataset), **metrics})
    writer.write_summary(summary)

    print(f"\n{config.run_id}  {checkpoint.name} on {config.split}, {len(dataset)} frames")
    for name in config.metrics:
        print(f"  {name:<10} {metrics[name]:.4f}")
    for name, value in sorted(metrics["per_class"].items(), key=lambda item: -item[1]):
        print(f"    {name:<12} {value:.3f}")
    print(f"\nwrote {run_dir}")
    return summary


def run_sessions(config: Config, checkpoint: Path, device: str) -> Dict:
    """
    score the detector and tracker on the configured sessions.

    returns the run summary, and writes one log record per session under config.run_dir.
    """
    if not config.sessions:
        raise ValueError(f"task {config.task} needs sessions in the config")

    pipeline = Pipeline.from_checkpoint(
        checkpoint, config.task, device, config.tracker, config.max_detections, config.nms_iou
    )

    # no transform: the pipeline letterboxes the frame itself, as it does at serving
    sessions = {
        name: SurgintDataset(config.data_root, f"{config.split}/{name}", config.task)
        for name in config.sessions
    }
    categories = categories_of(pipeline.detector.meta, next(iter(sessions.values())))

    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    writer.initialize(
        config,
        categories,
        str(checkpoint),
        annotations=[dataset.annotations for dataset in sessions.values()],
    )

    metrics = evaluate_sessions(pipeline, sessions, config.iou_threshold)
    frames = sum(len(dataset) for dataset in sessions.values())

    summary = {
        "run_id": config.run_id,
        "checkpoint": str(checkpoint),
        "task": config.task,
        "split": config.split,
        "sessions": list(sessions),
        "frames": frames,
        "metrics": metrics,
    }
    for name, values in metrics["per_session"].items():
        writer.append_log({"session": name, **values})
    writer.write_summary(summary)

    print(f"\n{config.run_id}  {checkpoint.name} on {config.split}, {len(sessions)} sessions, {frames} frames")
    for name in config.metrics:
        value = metrics[name]
        print(f"  {name:<12} {value:.4f}" if isinstance(value, float) else f"  {name:<12} {value}")
    for name, values in sorted(metrics["per_session"].items(), key=lambda item: -item[1]["MOTA"]):
        print(
            f"    {name:<14} MOTA {values['MOTA']:.3f}"
            f"  IDF1 {values['IDF1']:.3f}"
            f"  switches {values['id_switches']}"
        )
    print(f"\nwrote {run_dir}")
    return summary


def run(config: Config, checkpoint: Path, device: str) -> Dict:
    """evaluate a checkpoint."""
    if config.task == "detection-tracking":
        return run_sessions(config, checkpoint, device)
    return run_detection(config, checkpoint, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path, help="a checkpoint directory")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    run(load_config(args.config), args.ckpt, args.device)


if __name__ == "__main__":
    main()
