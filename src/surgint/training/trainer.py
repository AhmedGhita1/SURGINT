import json
import math
import time
from pathlib import Path

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from surgint.config import Config


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
    """eval mode keeps the pretrained running statistics, so training and inference match"""
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def to_device(batch: dict, device: str) -> tuple[torch.Tensor, list[dict]]:
    labels = [{key: value.to(device) for key, value in label.items()} for label in batch["labels"]]
    return batch["pixel_values"].to(device), labels


class Trainer:
    def __init__(self, model: torch.nn.Module, config: Config, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.run_dir = Path(config.run_dir) / config.run_id
        self.device = device

        self.optimizer = AdamW(parameter_groups(self.model, config), weight_decay=config.weight_decay)
        self.scheduler: LambdaLR | None = None
        self.scheduler_state: dict | None = None
        self.history: list[dict] = []
        self.epoch = 0
        self.best_loss = float("inf")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        if self.config.freeze_batchnorm:
            freeze_batchnorm(self.model)
        total = 0.0
        for batch in loader:
            pixel_values, labels = to_device(batch, self.device)
            loss = self.model(pixel_values=pixel_values, labels=labels).loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total += loss.item()
        return total / len(loader)

    @torch.inference_mode()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for batch in loader:
            pixel_values, labels = to_device(batch, self.device)
            total += self.model(pixel_values=pixel_values, labels=labels).loss.item()
        return total / len(loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> list[dict]:
        """train for epochs, save train history after each epoch."""
        self.scheduler = build_schedule(self.optimizer, self.config, self.config.epochs * len(train_loader))
        if self.scheduler_state:
            self.scheduler.load_state_dict(self.scheduler_state)

        for epoch in range(self.epoch + 1, self.config.epochs + 1):
            self.epoch = epoch
            metrics = {"epoch": epoch, "lr": self.scheduler.get_last_lr()[-1]}
            started = time.perf_counter()
            metrics["train_loss"] = self.train_epoch(train_loader)
            if val_loader is not None:
                metrics["val_loss"] = self.validate(val_loader)

            elapsed = time.perf_counter() - started
            remaining = elapsed * (self.config.epochs - epoch)

            self.history.append(metrics)
            line = (f"epoch [{epoch}/{self.config.epochs}]  lr {metrics['lr']:.2e}"
                    f"  train_loss {metrics['train_loss']:.4f}")
            if "val_loss" in metrics:
                line += f"  val_loss {metrics['val_loss']:.4f}"
            line += f"  {elapsed:.0f}s  eta {remaining / 60:.0f}m"
            print(line, flush=True)
            self.save()

            loss = metrics.get("val_loss", metrics["train_loss"])
            if loss < self.best_loss:
                self.best_loss = loss
                self.model.save_pretrained(self.run_dir / "best")
        return self.history

    def save(self) -> None:
        """save the checkpoint"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.run_dir)
        torch.save(
            {
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "history": self.history,
            },
            self.run_dir / "training_state.pt",
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "input_size": self.config.input_size,
                    "id2label": self.model.config.id2label,
                    "best_loss": self.best_loss,
                    "last_loss": self.history[-1].get("val_loss", self.history[-1]["train_loss"])
                    self.history: self.history,
                },
                indent=2,
            )
            + "\n"
        )

    def resume(self) -> None:
        """restore optimizer, scheduler, epoch and history; load the weights separately"""
        state = torch.load(self.run_dir / "training_state.pt", map_location=self.device, weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler_state = state["scheduler"]
        self.history = state["history"]
        self.epoch = state["epoch"]
        self.best_loss = state["best_loss"]
