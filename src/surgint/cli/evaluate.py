import argparse
from pathlib import Path
from typing import Dict

from surgint.artifacts import RunWriter
from surgint.cli.train import build_loader
from surgint.config import Config, load_config
from surgint.dataset.coco import SurgintDataset
from surgint.model.transform import Transform
from surgint.engine.evaluator import evaluate
from surgint.model.detector import Detector

CONFIG = Path("configs/eval.yaml")
DEVICE = "cuda"

def run(config: Config, checkpoint: Path, device: str) -> Dict:
    detector = Detector.from_checkpoint(checkpoint).to(device)
    meta = detector.meta

    transform = Transform(meta["input_size"], meta["pad_color"], meta["rescale_factor"])
    dataset = SurgintDataset(config.data_root, config.split, "detection-only", transform)

    categories = [name for _, name in dataset.mappings.values()]
    if meta["labels"] != categories:
        raise ValueError(f"checkpoint labels {meta['labels']} do not match categories {categories}")

    run_dir = Path(config.run_dir) / config.run_id
    writer = RunWriter(run_dir)
    writer.initialize(config, categories, str(checkpoint))

    metrics = evaluate(detector, build_loader(dataset, config.batch_size, shuffle=False), transform)

    summary = {
        "run_id": config.run_id,
        "checkpoint": str(checkpoint),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path, help="a checkpoint directory")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    run(load_config(args.config), args.ckpt, args.device)


if __name__ == "__main__":
    main()
