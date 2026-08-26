import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from surgint.config import Config, save_config
from surgint.detection.model import load_model
from surgint.detection.preprocessing import PAD_VALUE
from surgint.evaluation import MetricsFn


def parameter_groups(model: torch.nn.Module, config: Config) -> list[dict]:
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith("model.backbone") else head).append(parameter)
    return [
        {"params": backbone, "lr": config.backbone_learning_rate},
        {"params": head, "lr": config.learning_rate},
    ]


def build_schedule(optimizer: AdamW, config: Config, total_steps: int) -> LambdaLR:
    """linear warmup, then cosine decay to zero over the remaining steps"""
    warmup = config.warmup_steps

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if config.schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def freeze_batchnorm(model: torch.nn.Module) -> None:
    """freeze BN for stable training and inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def to_device(batch: dict, device: str) -> tuple[torch.Tensor, list[dict]]:
    labels = [{key: value.to(device) for key, value in label.items()} for label in batch["labels"]]
    return batch["pixel_values"].to(device), labels


class Trainer:
    def __init__(
            self, model: torch.nn.Module,
            config: Config,
            metrics_fn: MetricsFn,
            device: str = "cuda"):

        self.model = model.to(device)
        self.config = config
        self.metrics_fn = metrics_fn
        self.run_dir = Path(config.run_dir) / config.run_id
        self.latest_dir = self.run_dir / "latest"
        self.best_dir = self.run_dir / "best"
        self.device = device

        self.optimizer = AdamW(parameter_groups(self.model, config), weight_decay=config.weight_decay)
        self.scheduler: LambdaLR | None = None
        self.scheduler_state: dict | None = None
        self.log: list[dict] = []
        self.epoch = 0
        self.best_score = float("-inf")
        self.best_epoch = 0

    def compute_metrics(self) -> dict[str, float]:
        scored = self.metrics_fn(self.model)
        missing = [name for name in self.config.metrics if name not in scored]
        if missing:
            raise KeyError(
                f"metrics_fn returned {sorted(scored)}, missing configured metrics {missing}"
            )
        return {name: float(scored[name]) for name in self.config.metrics}

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:

        self.model.train()

        if self.config.freeze_batchnorm:
            freeze_batchnorm(self.model)

        total = 0.0
        steps = tqdm(loader, desc=f"epoch [{epoch}/{self.config.epochs}]", leave=False)
        for step, batch in enumerate(steps, start=1):
            pixel_values, labels = to_device(batch, self.device)
            loss = self.model(pixel_values=pixel_values, labels=labels).loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total += loss.item()
            steps.set_postfix(loss=f"{total / step:.3f}")

        return total / len(loader)

    def train(self, loader: DataLoader) -> list[dict]:

        self.scheduler = build_schedule(
            self.optimizer,
            self.config,
            self.config.epochs * len(loader)
        )
        if self.scheduler_state:
            self.scheduler.load_state_dict(self.scheduler_state)

        for epoch in range(self.epoch + 1, self.config.epochs + 1):
            self.epoch = epoch
            metrics = {"epoch": epoch, "lr": self.scheduler.get_last_lr()[-1]}
            metrics["train_loss"] = self.train_epoch(loader, epoch)
            metrics.update(self.compute_metrics())
            self.log.append(metrics)

            line = (f"epoch [{epoch}/{self.config.epochs}]  lr {metrics['lr']:.2e}"
                    f"  train_loss {metrics['train_loss']:.4f}")
            line += "".join(f"  {name} {metrics[name]:.4f}" for name in self.config.metrics)

            if metrics[self.config.select_metric] > self.best_score:
                self.best_score = metrics[self.config.select_metric]
                self.best_epoch = epoch
                self.save(self.best_dir)

                line += "  best"

            self.save_checkpoint()
            print(line, flush=True)

        return self.log

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.write_manifest(directory)

    def save_checkpoint(self) -> None:
        self.save(self.latest_dir)
        torch.save(
            {
                "epoch": self.epoch,
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "log": self.log,
            },
            self.latest_dir / "training_state.pt",
        )

    def load_checkpoint(self, checkpoint: str | Path | None = None) -> None:
        checkpoint = Path(checkpoint) if checkpoint is not None else self.latest_dir
        self.model.load_state_dict(load_model(checkpoint).state_dict())

        state = torch.load(checkpoint / "training_state.pt", map_location=self.device, weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler_state = state["scheduler"]
        self.log = state["log"]
        self.epoch = state["epoch"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]

    def write_manifest(self, directory: Path) -> None:

        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "version": version("surgint"),
                    "input_size": self.config.input_size,
                    "pad_value": PAD_VALUE,
                    "id2label": self.model.config.id2label,
                    "select_metric": self.config.select_metric,
                    "best_score": self.best_score if self.best_epoch else None,
                    "best_epoch": self.best_epoch or None,
                    "log": self.log,
                },
                indent=2,
            )
            + "\n"
        )
        save_config(self.config, directory / "config.yaml")
