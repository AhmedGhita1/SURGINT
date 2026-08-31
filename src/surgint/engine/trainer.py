from dataclasses import dataclass, field
from typing import Dict, Iterator, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from surgint.config import Config
from surgint.dataset.transform import Transform
from surgint.engine.optim import build_optimizer, build_scheduler, freeze_batchnorm
from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions
from surgint.model.decode import decode
from surgint.model.detector import Detector


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    lr: float
    train_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False

    def as_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "epoch": self.epoch,
            "lr": self.lr,
            "train_loss": self.train_loss,
            **{name: value for name, value in self.metrics.items() if name != "per_class"},
        }


class Trainer:
    def __init__(
        self,
        detector: Detector,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        transform: Transform
    ):
        
        self.detector = detector
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.transform = transform

        self.optimizer = build_optimizer(detector, config)
        self.scheduler = build_scheduler(
            self.optimizer, 
            config, 
            config.epochs * len(train_loader)
        )

        self.epoch = 0
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.best_metrics: Dict[str, float] = {}

    def train_epoch(self) -> float:
        self.detector.train()
        if self.config.freeze_batchnorm:
            freeze_batchnorm(self.detector)

        total = 0.0
        steps = tqdm(self.train_loader, desc=f"epoch [{self.epoch}/{self.config.epochs}]", leave=False)
        for step, batch in enumerate(steps, start=1):
            loss = self.detector(batch["pixel_values"], batch["labels"]).loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total += loss.item()
            steps.set_postfix(loss=f"{total / step:.3f}")

        return total / len(self.train_loader)

    def validate(self) -> Dict[str, float]:
        dataset = self.val_loader.dataset
        predictions = []

        for batch in tqdm(self.val_loader, desc="val", leave=False):
            logits, pred_boxes = self.detector.predict(batch["pixel_values"])
            detections = decode(logits, pred_boxes, score_threshold=0.0)

            for (boxes, scores, class_ids), image_id, scale, frame_size in zip(
                detections, batch["image_ids"], batch["scales"], batch["frame_sizes"]
            ):
                boxes = self.transform.postprocess(boxes, scale, frame_size)
                predictions += coco_predictions(
                    image_id, boxes, scores, class_ids, dataset.mappings
                )

        return coco_evaluate(dataset.gt, predictions)

    def train(self) -> Iterator[EpochResult]:
        for epoch in range(self.epoch + 1, self.config.epochs + 1):
            self.epoch = epoch
            lr = self.scheduler.get_last_lr()[-1]
            train_loss = self.train_epoch()

            if epoch % self.config.val_interval and epoch != self.config.epochs:
                yield EpochResult(epoch, lr, train_loss)
                continue

            metrics = self.validate()
            is_best = metrics[self.config.select_metric] > self.best_score
            if is_best:
                self.best_score = metrics[self.config.select_metric]
                self.best_epoch = epoch
                self.best_metrics = dict(metrics)

            yield EpochResult(epoch, lr, train_loss, metrics, is_best)

    def state_dict(self) -> Dict:
        return {
            "epoch": self.epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = state["epoch"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.best_metrics = state.get("best_metrics", {})
