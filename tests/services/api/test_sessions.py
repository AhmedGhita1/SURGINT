from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from services.api.app import create_app
from services.api.settings import ServingSettings


class FakeDetector:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(
            labels=("scalpel", "forceps"),
            input_size=(640, 640),
            pad_color=114,
            rescale_factor=1 / 255,
            weights_sha256="sha256:" + "a" * 64,
        )

    def to(self, device: str) -> None:
        pass

    def eval(self) -> None:
        pass


def test_session_can_be_created_and_deleted() -> None:
    application = create_app(
        ServingSettings(checkpoint=Path("model")),
        lambda _: FakeDetector(),
    )

    with TestClient(application) as client:
        response = client.post("/v1/sessions")
        assert response.status_code == 201

        session_id = UUID(response.json()["session_id"])
        session = application.state.sessions[session_id]
        assert session.pipeline.tracker is not None
        assert session.decision_support.frame_count == 0

        second_response = client.post("/v1/sessions")
        second_id = UUID(second_response.json()["session_id"])
        second = application.state.sessions[second_id]
        assert second.pipeline is not session.pipeline
        assert second.pipeline.tracker is not session.pipeline.tracker
        assert second.decision_support is not session.decision_support
        assert second.pipeline.detector is session.pipeline.detector
        assert second.pipeline.transform is session.pipeline.transform

        response = client.delete(f"/v1/sessions/{session_id}")
        assert response.status_code == 204
        assert response.content == b""

        response = client.delete(f"/v1/sessions/{session_id}")
        assert response.status_code == 404


def test_session_creation_requires_a_ready_model() -> None:
    application = create_app(ServingSettings(checkpoint=None))

    with TestClient(application) as client:
        response = client.post("/v1/sessions")

    assert response.status_code == 503
    assert response.json() == {"detail": "model is not ready"}


def test_unknown_session_returns_not_found() -> None:
    application = create_app(
        ServingSettings(checkpoint=Path("model")),
        lambda _: FakeDetector(),
    )

    with TestClient(application) as client:
        response = client.delete(f"/v1/sessions/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "session not found"}
