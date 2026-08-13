"""Public output contract v2 — spec 7, 16.3."""

from __future__ import annotations

import jsonschema
import pytest

from sastt.api.schemas import (
    downgrade_segment_to_v1,
    validate_segment_v1,
    validate_segment_v2,
    validate_server_event,
)
from sastt.domain.errors import SchemaInvariantError
from sastt.domain.events import EventType, SessionEventLog, SystemClock
from sastt.domain.speakers import IdentityStatus
from sastt.domain.transcript import (
    Confidences,
    ModelVersions,
    TranscriptSegment,
    Word,
    render_segment,
    sort_segments,
    stable_word_id,
)

pytestmark = pytest.mark.unit


def make_segment(**overrides: object) -> TranscriptSegment:
    defaults: dict[str, object] = {
        "session_id": "ses_01J",
        "event_id": "evt_01J",
        "start_ms": 4000,
        "end_ms": 8000,
        "text": "Nội dung của người nói thứ hai",
        "speaker_id": "sess_spk_02",
        "session_speaker_id": "sess_spk_02",
        "identity_status": IdentityStatus.ANONYMOUS,
        "speaker_label": "Speaker 2",
    }
    defaults.update(overrides)
    return TranscriptSegment(**defaults)  # type: ignore[arg-type]


class TestInvariants:
    def test_start_must_be_before_end(self) -> None:
        with pytest.raises(SchemaInvariantError):
            make_segment(start_ms=8000, end_ms=8000)

    def test_revision_starts_at_one(self) -> None:
        with pytest.raises(SchemaInvariantError):
            make_segment(revision=0)

    def test_source_track_required_once_separation_ran(self) -> None:
        with pytest.raises(SchemaInvariantError):
            make_segment(separation_backend="mossformer2_ss_16k")
        make_segment(separation_backend="mossformer2_ss_16k", source_track=1)

    def test_enrolled_requires_registry_id_and_name(self) -> None:
        with pytest.raises(SchemaInvariantError):
            make_segment(identity_status=IdentityStatus.ENROLLED, speaker_id="EMP-042")
        make_segment(
            identity_status=IdentityStatus.ENROLLED,
            speaker_id="EMP-042",
            registry_speaker_id="EMP-042",
            speaker_name="Nguyễn Văn B",
        )

    def test_non_enrolled_speaker_id_falls_back_to_session_id(self) -> None:
        with pytest.raises(SchemaInvariantError):
            make_segment(speaker_id="EMP-042")

    def test_confidence_bounds(self) -> None:
        with pytest.raises(SchemaInvariantError):
            Confidences(asr=1.5)
        with pytest.raises(SchemaInvariantError):
            Confidences(overall=0.9)  # uncalibrated cannot carry an overall score

    def test_uncalibrated_confidences_are_null(self) -> None:
        segment = make_segment()
        payload = segment.to_public_dict()
        assert payload["confidence_status"] == "uncalibrated"
        for key in (
            "asr_confidence",
            "diarization_confidence",
            "linking_confidence",
            "voice_id_confidence",
            "overlap_confidence",
            "overall_confidence",
        ):
            assert payload[key] is None


class TestConcurrencyAndOrdering:
    def test_overlapping_segments_are_both_kept(self) -> None:
        first = make_segment(
            start_ms=4000, end_ms=8000, session_speaker_id="sess_spk_01", speaker_id="sess_spk_01"
        )
        second = make_segment(
            start_ms=4000, end_ms=8000, session_speaker_id="sess_spk_02", speaker_id="sess_spk_02"
        )
        ordered = sort_segments([second, first])
        assert [s.session_speaker_id for s in ordered] == ["sess_spk_01", "sess_spk_02"]
        assert ordered[0].interval.intersects(ordered[1].interval)

    def test_sort_key_is_start_speaker_track(self) -> None:
        segment = make_segment(source_track=1, separation_backend="mossformer2_ss_16k")
        assert segment.sort_key == (4000, "sess_spk_02", 1)


class TestJsonSchema:
    def test_example_payload_validates(self) -> None:
        segment = make_segment(
            identity_status=IdentityStatus.ENROLLED,
            speaker_id="EMP-042",
            registry_speaker_id="EMP-042",
            speaker_name="Nguyễn Văn B",
            speaker_label="Nguyễn Văn B",
            is_overlap=True,
            estimated_concurrent_speakers=2,
            source_track=1,
            separation_backend="mossformer2_ss_16k",
            revision=3,
            supersedes_event_id="evt_01J_previous",
            raw_scores={"asr_word_probability": 0.91, "speaker_cosine_similarity": 0.89},
            model_versions=ModelVersions(
                diarization="community-1@rev",
                embedding="campplus@sha",
                separation="mossformer2_ss_16k@sha",
                asr="large-v3-turbo@rev",
            ),
        )
        validate_segment_v2(segment.to_public_dict())

    def test_schema_rejects_calibrated_looking_confidence_when_uncalibrated(self) -> None:
        payload = make_segment().to_public_dict()
        payload["overall_confidence"] = 0.87
        with pytest.raises(jsonschema.ValidationError):
            validate_segment_v2(payload)

    def test_schema_rejects_unknown_identity_status(self) -> None:
        payload = make_segment().to_public_dict()
        payload["identity_status"] = "guessed"
        with pytest.raises(jsonschema.ValidationError):
            validate_segment_v2(payload)

    def test_server_event_validates(self) -> None:
        log = SessionEventLog("ses_01J")
        event = log.append(event_type=EventType.SESSION_STARTED, clock=SystemClock())
        validate_server_event(event.to_dict())


class TestV1BackwardCompatibility:
    def test_v2_segment_projects_onto_v1(self) -> None:
        payload = make_segment().to_public_dict()
        v1 = downgrade_segment_to_v1(payload)
        validate_segment_v1(v1)
        assert v1["speaker_id"] == payload["speaker_id"]
        assert (v1["start_ms"], v1["end_ms"]) == (payload["start_ms"], payload["end_ms"])

    def test_v1_projection_keeps_two_concurrent_speakers(self) -> None:
        first = make_segment(session_speaker_id="sess_spk_01", speaker_id="sess_spk_01")
        second = make_segment(session_speaker_id="sess_spk_02", speaker_id="sess_spk_02")
        v1 = [downgrade_segment_to_v1(s.to_public_dict()) for s in (first, second)]
        assert len({item["speaker_id"] for item in v1}) == 2


class TestRendering:
    def test_enrolled_rendering(self) -> None:
        segment = make_segment(
            start_ms=4000,
            end_ms=8000,
            identity_status=IdentityStatus.ENROLLED,
            speaker_id="EMP-042",
            registry_speaker_id="EMP-042",
            speaker_name="Nguyễn Văn B",
            text="Nội dung…",
        )
        assert (
            render_segment(segment) == "00:04.000–00:08.000 — Nguyễn Văn B [EMP-042]: “Nội dung…”"
        )

    def test_anonymous_rendering(self) -> None:
        segment = make_segment(
            start_ms=1000, end_ms=5000, speaker_label="Speaker 1", text="Nội dung…"
        )
        assert render_segment(segment) == "00:01.000–00:05.000 — Speaker 1: “Nội dung…”"


class TestStableWordId:
    def test_id_is_stable_under_small_timestamp_jitter(self) -> None:
        a = stable_word_id(session_id="ses", start_sample=16_000, text="xin", source_track=None)
        b = stable_word_id(session_id="ses", start_sample=16_050, text="Xin ", source_track=None)
        assert a == b

    def test_id_differs_per_source_track(self) -> None:
        a = stable_word_id(session_id="ses", start_sample=16_000, text="xin", source_track=0)
        b = stable_word_id(session_id="ses", start_sample=16_000, text="xin", source_track=1)
        assert a != b

    def test_word_gets_id_and_shifts_to_absolute_time(self) -> None:
        word = Word(text="xin", start_ms=100, end_ms=400).shifted(4000).with_stable_id("ses")
        assert (word.start_ms, word.end_ms) == (4100, 4400)
        assert word.word_id is not None
