from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["alive", "ready", "not_ready"]


class SessionCreatedResponse(BaseModel):
    session_id: UUID
