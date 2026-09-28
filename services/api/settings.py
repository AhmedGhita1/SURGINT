import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingSettings:
    """Configuration needed by the local API to load the model ."""

    checkpoint: Path | None
    device: str = "cpu"
    max_sessions: int = 8

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_sessions, bool)
            or not isinstance(self.max_sessions, int)
            or self.max_sessions < 1
        ):
            raise ValueError("max_sessions must be a positive integer")

    @classmethod
    def from_environment(cls) -> "ServingSettings":
        checkpoint = os.getenv("SURGINT_CHECKPOINT")
        return cls(
            checkpoint=Path(checkpoint) if checkpoint else None,
            device=os.getenv("SURGINT_DEVICE", "cpu"),
            max_sessions=int(os.getenv("SURGINT_MAX_SESSIONS", "8")),
        )
