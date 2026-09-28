import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response, status

from services.api.schemas import HealthResponse
from services.api.settings import ServingSettings
from surgint.model.detector import Detector

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

        if settings.checkpoint is None:
            logger.warning("SURGINT_CHECKPOINT is not set; the API is not ready")
        else:
            try:
                detector = detector_loader(settings.checkpoint)
                detector.to(settings.device)
                detector.eval()
                app.state.detector = detector
                logger.info("loaded model from %s on %s", settings.checkpoint, settings.device)
            except Exception:
                logger.exception("failed to load model from %s", settings.checkpoint)

        yield
        app.state.detector = None

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

    return application


app = create_app()
