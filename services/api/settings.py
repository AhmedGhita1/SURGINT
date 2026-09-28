import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingSettings:
    """Configuration needed by the local API to load the model ."""

    checkpoint: Path | None
    device: str = "cpu"

    @classmethod
    def from_environment(cls) -> "ServingSettings":
        checkpoint = os.getenv("SURGINT_CHECKPOINT")
        return cls(
            checkpoint=Path(checkpoint) if checkpoint else None,
            device=os.getenv("SURGINT_DEVICE", "cpu"),
        )
