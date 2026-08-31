import math
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

from surgint.config import Config
from surgint.model.detector import Detector


def build_optimizer(detector: Detector, config: Config) -> AdamW:
    backbone, head = detector.parameter_split()
    return AdamW(
        [
            {"params": backbone, "lr": config.backbone_learning_rate},
            {"params": head, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: AdamW, config: Config, total_steps: int) -> LambdaLR:
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
    """running statistics stay at their pretrained values"""
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
