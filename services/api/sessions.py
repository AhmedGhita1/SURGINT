from dataclasses import dataclass, field
from threading import Lock

from surgint.model.detector import Detector
from surgint.model.transform import Transform
from surgint.runtime.decision_support import InventoryDecisionSupport
from surgint.runtime.pipeline import Pipeline
from surgint.runtime.session import DecisionSupportSession


@dataclass
class ActiveSession:
    """State owned by one sequence of frames."""

    pipeline: Pipeline
    decision_support: DecisionSupportSession
    lock: Lock = field(default_factory=Lock, repr=False)


def create_active_session(detector: Detector, transform: Transform) -> ActiveSession:
    """Create isolated tracking and decision state around a shared model."""
    if detector.manifest is None:
        raise ValueError("the loaded detector has no model manifest")

    support = InventoryDecisionSupport(
        labels=detector.manifest.labels,
        perception_version=detector.manifest.weights_sha256,
    )
    return ActiveSession(
        pipeline=Pipeline(
            detector=detector,
            transform=transform,
            task="detection-tracking",
        ),
        decision_support=DecisionSupportSession(support),
    )
