import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingSettings:
    """Configuration needed by the local API to load the model ."""

    checkpoint: Path | None
    device: str = "cpu"
    max_sessions: int = 8
    max_video_bytes: int = 256 * 1024 * 1024
    video_batch_size: int = 4
    video_sample_fps: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_sessions", "max_video_bytes", "video_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.video_sample_fps, bool)
            or not isinstance(self.video_sample_fps, (int, float))
            or not math.isfinite(self.video_sample_fps)
            or self.video_sample_fps <= 0
        ):
            raise ValueError("video_sample_fps must be a positive finite number")

    @classmethod
    def from_environment(cls) -> "ServingSettings":
        checkpoint = os.getenv("SURGINT_CHECKPOINT")
        return cls(
            checkpoint=Path(checkpoint) if checkpoint else None,
            device=os.getenv("SURGINT_DEVICE", "cpu"),
            max_sessions=int(os.getenv("SURGINT_MAX_SESSIONS", "8")),
            max_video_bytes=int(os.getenv("SURGINT_MAX_VIDEO_BYTES", str(256 * 1024 * 1024))),
            video_batch_size=int(os.getenv("SURGINT_VIDEO_BATCH_SIZE", "4")),
            video_sample_fps=float(os.getenv("SURGINT_VIDEO_SAMPLE_FPS", "1.0")),
        )
