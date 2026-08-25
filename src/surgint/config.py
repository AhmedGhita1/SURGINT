from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

@dataclass
class Config:
    checkpoint: str = "PekingU/rtdetr_r18vd_coco_o365"
    input_size: list[int] = field(default_factory=lambda: [1024, 576])

    data_root: str = "data/synthetic/production_v1/dataset"

    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    seed: int = 0

    def __post_init__(self):
        if len(self.input_size) != 2:
            raise ValueError(f"input_size must be [width, height], got {self.input_size}")
        if any(v <= 0 or v % 32 for v in self.input_size):
            raise ValueError(f"input_size must be positive and divisible by 32, got {self.input_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


def load_config(path: str | Path) -> Config:
    return Config(**yaml.safe_load(Path(path).read_text()))


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
