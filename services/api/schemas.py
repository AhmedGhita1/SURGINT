from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


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


class VideoProcessedResponse(BaseModel):
    frame_count: int


class ItemContextOverrideRequest(BaseModel):
    product_id: str | None = None
    lifecycle: Literal["reusable", "single-use"] | None = None


class FinalizeSessionRequest(BaseModel):
    workflow_stage: Literal[
        "pre-procedure-setup",
        "in-procedure",
        "post-procedure-clearing",
    ]
    use_state: Literal["unused", "used"] | None = None
    contamination_state: Literal[
        "not-regulated",
        "potentially-infectious",
        "chemical",
        "cytotoxic",
        "radioactive",
    ] | None = None
    overrides: dict[int, ItemContextOverrideRequest] = Field(default_factory=dict)


class FinalizedItemResponse(BaseModel):
    class_id: int
    label: str
    count: int
    confidence: float
    outcome: str
    action: str | None
    reason: str
    matched_rule: str | None
    missing_fields: list[str]


class FinalizedSessionResponse(BaseModel):
    frame_count: int
    items: list[FinalizedItemResponse]
