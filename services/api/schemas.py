from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["alive", "ready", "not_ready"]


class SessionCreatedResponse(BaseModel):
    session_id: UUID


class TrackedDetectionResponse(BaseModel):
    track_id: int
    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]


class FrameProcessedResponse(BaseModel):
    frame_count: int
    detections: list[TrackedDetectionResponse]
