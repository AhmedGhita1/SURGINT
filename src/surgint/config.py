from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

@dataclass
class Config:
    checkpoint: str = "PekingU/rtdetr_r18vd_coco_o365"
    device: str = "cuda"
    input_size: int = 640
    confidence_threshold: float = 0.3

    data_root: str = "data/synthetic/production_v1/dataset"

    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    seed: int = 0

    def __post_init__(self):
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


def load_config(path: str | Path) -> Config:
    return Config(**yaml.safe_load(Path(path).read_text()))


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
