from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from services.api.app import create_app
from services.api.settings import ServingSettings


class FakeDetector:
    def __init__(self) -> None:
        self.device = None
        self.evaluating = False
        self.manifest = SimpleNamespace(
            input_size=(640, 640),
            pad_color=114,
            rescale_factor=1 / 255,
        )

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        self.evaluating = True


def test_health_is_ready_after_model_load() -> None:
    detector = FakeDetector()
    settings = ServingSettings(checkpoint=Path("model"), device="cpu")

    with TestClient(create_app(settings, lambda _: detector)) as client:
        assert client.get("/health/live").json() == {"status": "alive"}

        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
        assert detector.device == "cpu"
        assert detector.evaluating


def test_health_is_not_ready_without_checkpoint() -> None:
    settings = ServingSettings(checkpoint=None)

    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200

        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}


def test_console_is_served_from_the_package() -> None:
    """the operator page ships with the package, not beside the source tree"""
    settings = ServingSettings(checkpoint=None)

    with TestClient(create_app(settings)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "INVENTORY" in response.text
        assert 'get("__sign")' in response.text
        assert 'fetch(apiUrl("/v1/sessions")' in response.text
        assert "sessions/${session}/video" in response.text
        assert "FRAME_STEP" not in response.text
