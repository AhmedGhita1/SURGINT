import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

ADJECTIVES = "brisk calm dry eager fair glad keen lone mild neat proud quick rare sharp tidy warm".split()
NOUNS = "adder bison crane dingo eagle falcon gecko heron ibis jackal koala lynx mink otter puma raven".split()


@dataclass
class Config:
    checkpoint: str = "PekingU/rtdetr_r18vd_coco_o365"
    input_size: list[int] = field(default_factory=lambda: [1024, 576])

    dataset_id: str = "production_v1"
    data_root: str = "data/synthetic/production_v1/dataset"
    run_dir: str = "outputs/runs"
    run_id: str = ""
    split: str = "val"
    layouts: int = 0
    iou_threshold: float = 0.5
    metrics: list[str] = field(default_factory=lambda: ["mAP50_95", "mAP50", "mAP75"])
    select_metric: str = "mAP50_95"

    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    schedule: str = "cosine"
    grad_clip: float = 0.1
    freeze_batchnorm: bool = True
    seed: int = 0

    def __post_init__(self):
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if len(self.input_size) != 2:
            raise ValueError(f"input_size must be [width, height], got {self.input_size}")
        if any(v <= 0 or v % 32 for v in self.input_size):
            raise ValueError(f"input_size must be positive and divisible by 32, got {self.input_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.select_metric not in self.metrics:
            raise ValueError(f"select_metric {self.select_metric} is not in metrics {self.metrics}")
        if self.schedule not in ("cosine", "constant"):
            raise ValueError(f"schedule must be cosine or constant, got {self.schedule}")
        if not self.run_id:
            self.set_run_id()

    def set_run_id(self, run_id: str | None = None) -> str:
        """name the run; secrets keeps the name independent of the seeded rng"""
        name = f"{secrets.choice(ADJECTIVES)}-{secrets.choice(NOUNS)}"
        self.run_id = run_id or f"{name}_{datetime.now():%Y%m%d_%H%M}"
        return self.run_id


def load_config(path: str | Path) -> Config:
    return Config(**yaml.safe_load(Path(path).read_text()))


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
