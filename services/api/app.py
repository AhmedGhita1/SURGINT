import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import UUID, uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from PIL import Image, UnidentifiedImageError

from services.api.schemas import (
    FrameProcessedResponse,
    HealthResponse,
    SessionCreatedResponse,
    TrackedDetectionResponse,
)
from services.api.sessions import ActiveSession, create_active_session
from services.api.settings import ServingSettings
from surgint.model.detector import Detector
from surgint.model.transform import Transform

logger = logging.getLogger(__name__)


def create_app(
    settings: ServingSettings | None = None,
    detector_loader: Callable[[Path], Detector] | None = None,
) -> FastAPI:
    """Create the local API and load the model during application startup."""
    settings = settings or ServingSettings.from_environment()
    detector_loader = detector_loader or Detector.from_checkpoint

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.detector = None
        app.state.transform = None
        app.state.sessions: dict[UUID, ActiveSession] = {}
        app.state.sessions_lock = Lock()
        app.state.inference_lock = Lock()

        if settings.checkpoint is None:
            logger.warning("SURGINT_CHECKPOINT is not set; the API is not ready")
        else:
            try:
                detector = detector_loader(settings.checkpoint)
                detector.to(settings.device)
                detector.eval()
                if detector.manifest is None:
                    raise ValueError("the loaded detector has no model manifest")
                transform = Transform(
                    list(detector.manifest.input_size),
                    detector.manifest.pad_color,
                    detector.manifest.rescale_factor,
                )
                app.state.detector = detector
                app.state.transform = transform
                logger.info("loaded model from %s on %s", settings.checkpoint, settings.device)
            except Exception:
                logger.exception("failed to load model from %s", settings.checkpoint)

        yield
        with app.state.sessions_lock:
            app.state.sessions.clear()
        app.state.detector = None
        app.state.transform = None

    application = FastAPI(title="Surgint API", version="1", lifespan=lifespan)

    @application.get("/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        return HealthResponse(status="alive")

    @application.get("/health/ready", response_model=HealthResponse)
    def ready(response: Response) -> HealthResponse:
        if application.state.detector is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(status="not_ready")
        return HealthResponse(status="ready")

    @application.post(
        "/v1/sessions",
        response_model=SessionCreatedResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_session() -> SessionCreatedResponse:
        detector = application.state.detector
        transform = application.state.transform
        if detector is None or transform is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model is not ready",
            )

        session_id = uuid4()
        session = create_active_session(detector, transform)
        with application.state.sessions_lock:
            application.state.sessions[session_id] = session
        return SessionCreatedResponse(session_id=session_id)

    @application.delete(
        "/v1/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_session(session_id: UUID) -> Response:
        with application.state.sessions_lock:
            session = application.state.sessions.pop(session_id, None)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.post(
        "/v1/sessions/{session_id}/frames",
        response_model=FrameProcessedResponse,
    )
    def process_frame(
        session_id: UUID,
        image: Annotated[UploadFile, File()],
    ) -> FrameProcessedResponse:
        with application.state.sessions_lock:
            session = application.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )

        try:
            with Image.open(image.file) as uploaded:
                frame = np.asarray(uploaded.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid image",
            ) from error

        with session.lock:
            with application.state.inference_lock:
                detections = session.pipeline.predict(frame, score_threshold=0.0)
            session.decision_support.update(detections)
            frame_count = session.decision_support.frame_count

        labels = session.pipeline.detector.manifest.labels
        if detections.track_ids is None:
            raise RuntimeError("tracking pipeline returned detections without track ids")
        response_detections = [
            TrackedDetectionResponse(
                track_id=int(track_id),
                class_id=int(class_id),
                label=labels[int(class_id)],
                score=float(score),
                box=tuple(float(coordinate) for coordinate in box),
            )
            for box, score, class_id, track_id in zip(
                detections.boxes,
                detections.scores,
                detections.class_ids,
                detections.track_ids,
            )
        ]
        return FrameProcessedResponse(
            frame_count=frame_count,
            detections=response_detections,
        )

    return application


app = create_app()
