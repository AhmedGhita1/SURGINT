import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from surgint.config import Config
from surgint.evaluation import MetricsFn


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    lr: float
    train_loss: float
    metrics: dict[str, float]
    is_best: bool

    def as_dict(self) -> dict[str, int | float]:
        return {
            "epoch": self.epoch,
            "lr": self.lr,
            "train_loss": self.train_loss,
            **self.metrics,
        }


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
        self,
        model: torch.nn.Module,
        config: Config,
        metrics_fn: MetricsFn,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.metrics_fn = metrics_fn
        self.device = device

        self.optimizer = AdamW(parameter_groups(self.model, config), weight_decay=config.weight_decay)
        self.scheduler: LambdaLR | None = None
        self.scheduler_state: dict | None = None
        self.epoch = 0
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.best_metrics: dict[str, float] = {}

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

    def train(self, loader: DataLoader) -> Iterator[EpochResult]:
        self.scheduler = build_schedule(
            self.optimizer,
            self.config,
            self.config.epochs * len(loader),
        )
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

        for epoch in range(self.epoch + 1, self.config.epochs + 1):
            self.epoch = epoch
            lr = self.scheduler.get_last_lr()[-1]
            train_loss = self.train_epoch(loader, epoch)
            metrics = self.compute_metrics()
            is_best = metrics[self.config.select_metric] > self.best_score
            if is_best:
                self.best_score = metrics[self.config.select_metric]
                self.best_epoch = epoch
                self.best_metrics = dict(metrics)

            yield EpochResult(epoch, lr, train_loss, metrics, is_best)

    def state_dict(self) -> dict:
        if self.scheduler is None:
            raise RuntimeError("training state is unavailable before train() starts")
        return {
            "epoch": self.epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler_state = state["scheduler"]
        self.epoch = state["epoch"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.best_metrics = state.get("best_metrics", {})
