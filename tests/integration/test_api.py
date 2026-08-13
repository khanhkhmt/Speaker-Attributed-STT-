"""HTTP and WebSocket surface — spec 8.1, 8.2, 8.4."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from sastt.api.http import create_app
from sastt.api.schemas import validate_segment_v2, validate_server_event
from sastt.config import Environment
from sastt.domain.errors import ConfigurationError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


class TestProbes:
    def test_readiness_names_the_engine_and_pin_state(self, client: TestClient) -> None:
        body = client.get("/readyz").json()
        assert body["engine"] == "fake"
        assert body["config_version"].startswith("cfg_")
        assert body["max_supported_concurrent_speakers"] == 2
        assert any(model["backend"] == "mossformer2_ss_16k" for model in body["models"])

    def test_health(self, client: TestClient) -> None:
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_production_refuses_the_dev_auth_stub(self) -> None:
        """Spec 14.2: the tenant must come from auth claims, not a header."""
        with pytest.raises(ConfigurationError):
            create_app(environment=Environment.PRODUCTION)


class TestOfflineJobs:
    def test_idempotency_key_is_required(self, client: TestClient) -> None:
        response = client.post("/v1/jobs", json={"scenario": "s02_two_speaker_overlap"})
        assert response.status_code == 400

    def test_retry_with_the_same_key_returns_the_same_job(self, client: TestClient) -> None:
        headers = {"Idempotency-Key": "retry-test"}
        body = {"scenario": "s02_two_speaker_overlap"}
        first = client.post("/v1/jobs", json=body, headers=headers).json()
        second = client.post("/v1/jobs", json=body, headers=headers).json()
        assert first["job_id"] == second["job_id"]
        assert first["created"] is True and second["created"] is False

    def test_result_matches_the_public_schema(self, client: TestClient) -> None:
        created = client.post(
            "/v1/jobs",
            json={"scenario": "s02_two_speaker_overlap"},
            headers={"Idempotency-Key": "schema-test"},
        ).json()
        result = client.get(f"/v1/jobs/{created['job_id']}/result").json()
        assert result["state"] == "SUCCEEDED"
        assert result["schema_version"] == "2.0"
        assert result["segments"]
        for segment in result["segments"]:
            validate_segment_v2(segment)
        overlapping = [s for s in result["segments"] if s["is_overlap"]]
        assert len(overlapping) == 2  # two concurrent speakers survive (spec 0.1.7)

    def test_a_job_needs_a_scenario_or_audio(self, client: TestClient) -> None:
        response = client.post("/v1/jobs", json={}, headers={"Idempotency-Key": "empty-test"})
        assert response.status_code == 400

    def test_real_audio_upload_reports_model_not_ready(self, client: TestClient) -> None:
        """Spec 18 rule 6: no pretending the fake engine transcribes real audio."""
        response = client.post(
            "/v1/jobs",
            json={"audio_base64": base64.b64encode(b"\x00\x00" * 16_000).decode()},
            headers={"Idempotency-Key": "upload-test"},
        )
        assert response.status_code == 503
        assert response.json()["error_code"] == "MODEL_NOT_READY"

    def test_unknown_tenant_cannot_read_a_job(self, client: TestClient) -> None:
        created = client.post(
            "/v1/jobs",
            json={"scenario": "s01_five_speakers_no_overlap"},
            headers={"Idempotency-Key": "tenant-test", "X-Tenant-Id": "tenant_a"},
        ).json()
        denied = client.get(f"/v1/jobs/{created['job_id']}", headers={"X-Tenant-Id": "tenant_b"})
        assert denied.status_code == 400
        assert denied.json()["error_code"] == "TENANT_ACCESS_DENIED"


class TestRealtimeSession:
    def _stream(self, client: TestClient, scenario: str) -> tuple[str, list[dict]]:
        session = client.post("/v1/sessions", json={"scenario": scenario}).json()
        pcm = client.get(f"/v1/demo/scenarios/{scenario}/audio.wav").content[44:]
        frame = session["frame_ms"] * session["sample_rate"] // 1000 * 2
        events: list[dict] = []
        with client.websocket_connect(session["websocket_url"]) as socket:
            for offset in range(0, len(pcm), frame):
                socket.send_bytes(pcm[offset : offset + frame])
            socket.send_text(json.dumps({"type": "finalize"}))
            while True:
                event = socket.receive_json()
                events.append(event)
                if event["type"] == "session.finalized":
                    break
        return session["session_id"], events

    def test_event_stream_follows_the_contract(self, client: TestClient) -> None:
        _, events = self._stream(client, "s04_overlap_at_start")
        for event in events:
            validate_server_event(event)
        kinds = [event["type"] for event in events]
        assert kinds[0] == "session.started"
        assert kinds[-1] == "session.finalized"
        assert "transcript.provisional" in kinds
        assert "transcript.final" in kinds
        sequences = [event["sequence_number"] for event in events]
        assert sequences == sorted(sequences) == list(range(1, len(events) + 1))

    def test_overlap_at_start_is_revised_to_real_speakers(self, client: TestClient) -> None:
        _, events = self._stream(client, "s04_overlap_at_start")
        provisional = [
            event
            for event in events
            if event["type"] == "transcript.provisional" and event["payload"]["start_ms"] == 0
        ]
        assert all(
            event["payload"]["speaker_label"].startswith("Temporary Speaker")
            for event in provisional
        )
        finals = [
            event
            for event in events
            if event["type"] == "transcript.final" and event["payload"]["start_ms"] == 0
        ]
        assert {event["payload"]["speaker_label"] for event in finals} == {"Speaker 1", "Speaker 2"}

    def test_reconnect_replays_without_duplicating_finals(self, client: TestClient) -> None:
        session_id, events = self._stream(client, "s02_two_speaker_overlap")
        expected = [event for event in events if event["sequence_number"] > 3]
        with client.websocket_connect(f"/v1/sessions/{session_id}/audio") as socket:
            socket.send_text(json.dumps({"type": "resume", "last_sequence_number": 3}))
            replayed = [socket.receive_json() for _ in expected]
        assert [e["sequence_number"] for e in replayed] == [e["sequence_number"] for e in expected]
        event_ids = [e["event_id"] for e in replayed if e["is_final"]]
        assert len(event_ids) == len(set(event_ids))

    def test_session_result_is_the_canonical_transcript(self, client: TestClient) -> None:
        session_id, _ = self._stream(client, "s02_two_speaker_overlap")
        body = client.get(f"/v1/sessions/{session_id}/result").json()
        assert body["segments"]
        for segment in body["segments"]:
            validate_segment_v2(segment)
        assert body["text"]

    def test_unknown_session_is_reported_not_ignored(self, client: TestClient) -> None:
        with client.websocket_connect("/v1/sessions/ses_missing/audio") as socket:
            assert socket.receive_json()["payload"]["error_code"] == "UNKNOWN_SESSION"
