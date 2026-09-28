from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.api.app import create_app
from services.api.settings import ServingSettings
from surgint.model import Detections


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


def test_frame_updates_tracking_session() -> None:
    application = create_app(
        ServingSettings(checkpoint=Path("model")),
        lambda _: FakeDetector(),
    )

    with TestClient(application) as client:
        session_id = UUID(client.post("/v1/sessions").json()["session_id"])
        session = application.state.sessions[session_id]
        detections = Detections(
            boxes=np.asarray([[10, 20, 30, 40]], dtype=np.float32),
            scores=np.asarray([0.9], dtype=np.float32),
            class_ids=np.asarray([1], dtype=np.int64),
            track_ids=np.asarray([7], dtype=np.int64),
        )
        session.pipeline.predict = Mock(return_value=detections)

        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={"image": ("frame.png", image_bytes(), "image/png")},
        )

        assert response.status_code == 200
        assert response.json() == {
            "frame_count": 1,
            "detections": [
                {
                    "track_id": 7,
                    "class_id": 1,
                    "label": "forceps",
                    "score": pytest.approx(0.9),
                    "box": [10.0, 20.0, 30.0, 40.0],
                }
            ],
        }
        frame = session.pipeline.predict.call_args.args[0]
        assert frame.shape == (8, 12, 3)
        assert session.decision_support.frame_count == 1


def test_frame_rejects_invalid_image_without_updating_session() -> None:
    application = create_app(
        ServingSettings(checkpoint=Path("model")),
        lambda _: FakeDetector(),
    )

    with TestClient(application) as client:
        session_id = UUID(client.post("/v1/sessions").json()["session_id"])
        session = application.state.sessions[session_id]
        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={"image": ("frame.png", b"not an image", "image/png")},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "invalid image"}
        assert session.decision_support.frame_count == 0


def test_session_finalization_returns_inventory_decisions() -> None:
    application = create_app(
        ServingSettings(checkpoint=Path("model")),
        lambda _: FakeDetector(),
    )

    with TestClient(application) as client:
        session_id = UUID(client.post("/v1/sessions").json()["session_id"])
        session = application.state.sessions[session_id]
        session.decision_support.update(
            Detections(
                boxes=np.asarray([[10, 20, 30, 40]], dtype=np.float32),
                scores=np.asarray([0.9], dtype=np.float32),
                class_ids=np.asarray([0], dtype=np.int64),
                track_ids=np.asarray([7], dtype=np.int64),
            )
        )

        response = client.post(
            f"/v1/sessions/{session_id}/finalize",
            json={
                "workflow_stage": "post-procedure-clearing",
                "use_state": "used",
                "contamination_state": "not-regulated",
                "overrides": {"0": {"lifecycle": "reusable"}},
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "frame_count": 1,
            "items": [
                {
                    "class_id": 0,
                    "label": "scalpel",
                    "count": 1,
                    "confidence": pytest.approx(0.9),
                    "outcome": "recommendation",
                    "action": "secure-transport-to-reprocessing",
                    "reason": "Reusable sharp items require secure transport to reprocessing.",
                    "matched_rule": "reusable-sharp",
                    "missing_fields": [],
                }
            ],
        }

        response = client.post(
            f"/v1/sessions/{session_id}/finalize",
            json={"workflow_stage": "post-procedure-clearing"},
        )
        assert response.status_code == 409
        assert response.json() == {"detail": "session has already been finalized"}


def test_finalized_session_rejects_more_frames() -> None:
    application = create_app(
        ServingSettings(checkpoint=Path("model")),
        lambda _: FakeDetector(),
    )

    with TestClient(application) as client:
        session_id = UUID(client.post("/v1/sessions").json()["session_id"])
        response = client.post(
            f"/v1/sessions/{session_id}/finalize",
            json={"workflow_stage": "post-procedure-clearing"},
        )
        assert response.status_code == 200

        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={"image": ("frame.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 409
        assert response.json() == {"detail": "session has already been finalized"}


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (12, 8), color="black").save(stream, format="PNG")
    return stream.getvalue()
