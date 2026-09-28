from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import create_app
from services.api.settings import ServingSettings


class FakeDetector:
    def __init__(self) -> None:
        self.device = None
        self.evaluating = False

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
