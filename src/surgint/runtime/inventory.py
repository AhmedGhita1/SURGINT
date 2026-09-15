from collections import Counter
from dataclasses import dataclass
from typing import Dict

from surgint.model import Detections


@dataclass
class InventoryItem:
    """one instrument instance, identified by its track id."""

    track_id: int
    class_id: int
    first_seen: int
    last_seen: int
    frames: int
    score: float


class Inventory:
    """per-session instrument inventory, keyed by track id."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """drop every item and rewind the frame counter. call between sessions"""
        self.items: Dict[int, InventoryItem] = {}
        self.simultaneous: Dict[int, int] = {}
        self.frame = 0

    def update(self, detections: Detections) -> None:
        """record one frame of tracked detections"""
        if detections.track_ids is None:
            raise ValueError("inventory requires track ids, so the task must be detection-tracking")

        for track_id, class_id, score in zip(
            detections.track_ids, detections.class_ids, detections.scores
        ):
            track_id, class_id, score = int(track_id), int(class_id), float(score)
            item = self.items.get(track_id)

            if item is None:
                self.items[track_id] = InventoryItem(
                    track_id, class_id, self.frame, self.frame, 1, score
                )
                continue

            # the track votes its own class, so the latest value is its current answer
            item.class_id = class_id
            item.last_seen = self.frame
            item.frames += 1
            item.score = max(item.score, score)

        present = Counter(int(class_id) for class_id in detections.class_ids)
        for class_id, count in present.items():
            self.simultaneous[class_id] = max(self.simultaneous.get(class_id, 0), count)

        self.frame += 1

    def counts(self) -> Dict[int, int]:
        """distinct tracks per class."""
        return dict(Counter(item.class_id for item in self.items.values()))
