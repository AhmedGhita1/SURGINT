import json
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


def warmup_schedule(optimizer: AdamW, warmup_steps: int) -> LambdaLR:
    return LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0)


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
        self.scheduler = warmup_schedule(self.optimizer, config.warmup_steps)
        self.history: list[dict] = []
        self.epoch = 0

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            pixel_values, labels = to_device(batch, self.device)
            loss = self.model(pixel_values=pixel_values, labels=labels).loss

            loss.backward()
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
        for epoch in range(self.epoch + 1, self.config.epochs + 1):
            self.epoch = epoch
            metrics = {"epoch": epoch, "train_loss": self.train_epoch(train_loader)}
            if val_loader is not None:
                metrics["val_loss"] = self.validate(val_loader)

            self.history.append(metrics)
            print("  ".join(f"{key} {value:.4f}" if key != "epoch" else f"epoch {value}"
                            for key, value in metrics.items()))
            self.save()
        return self.history

    def save(self) -> None:
        """save the checkpoint"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.run_dir)
        torch.save(
            {
                "epoch": self.epoch,
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
                    "history": self.history,
                },
                indent=2,
            )
            + "\n"
        )

    def resume(self) -> None:
        """restore optimizer, scheduler, epoch and history; load the weights separately"""
        state = torch.load(self.run_dir / "training_state.pt", map_location=self.device, weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.history = state["history"]
        self.epoch = state["epoch"]
