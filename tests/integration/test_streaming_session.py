"""Near-realtime session over fake adapters — spec 16.2 (S04, S12, S13).

Structural only: no weights, no GPU, no wall-clock latency claims. The realtime
SLOs of spec 3 are measured in the load suite on real hardware.
"""

from __future__ import annotations

import pytest

from sastt.api.schemas import validate_segment_v2, validate_server_event
from sastt.application.offline_pipeline import OfflinePipeline
from sastt.application.streaming_pipeline import StreamingSession
from sastt.config import SasttConfig
from sastt.domain.events import EventType, ServerEvent
from sastt.observability import CallContext

pytestmark = pytest.mark.integration

from conftest import FrozenClock, build_adapters, load_scenario, scenario_pcm  # noqa: E402


def run_stream(
    scenario_name: str,
    config: SasttConfig,
    *,
    finalize: bool = True,
) -> tuple[StreamingSession, list[ServerEvent]]:
    scenario = load_scenario(scenario_name)
    session = StreamingSession(
        session_id="ses_stream_test",
        config=config,
        adapters=build_adapters(scenario),
        clock=FrozenClock(),
    )
    events = [session.start()]
    pcm = scenario_pcm(scenario)
    frame_bytes = int(config.streaming.frame_ms * config.audio.canonical_sample_rate / 1000) * 2
    for offset in range(0, len(pcm), frame_bytes):
        events.extend(session.push_pcm(pcm[offset : offset + frame_bytes]))
    if finalize:
        events.extend(session.finalize())
    return session, events


def by_type(events: list[ServerEvent], event_type: EventType) -> list[ServerEvent]:
    return [event for event in events if event.event_type is event_type]


class TestEventStream:
    def test_event_envelope_matches_the_contract(self, calibrated_config: SasttConfig) -> None:
        _, events = run_stream("s02_two_speaker_overlap.json", calibrated_config)
        assert events[0].event_type is EventType.SESSION_STARTED
        assert events[-1].event_type is EventType.SESSION_FINALIZED
        assert [event.sequence_number for event in events] == list(range(1, len(events) + 1))
        for event in events:
            validate_server_event(event.to_dict())
            assert event.config_version

    def test_provisional_then_final(self, calibrated_config: SasttConfig) -> None:
        _, events = run_stream("s02_two_speaker_overlap.json", calibrated_config)
        provisional = by_type(events, EventType.TRANSCRIPT_PROVISIONAL)
        finals = by_type(events, EventType.TRANSCRIPT_FINAL)
        assert provisional and finals
        assert all(event.is_final is False for event in provisional)
        assert all(event.is_final for event in finals)
        for event in finals:
            validate_segment_v2(event.payload)

    def test_every_final_names_the_event_it_replaces(self, calibrated_config: SasttConfig) -> None:
        _, events = run_stream("s02_two_speaker_overlap.json", calibrated_config)
        finals = by_type(events, EventType.TRANSCRIPT_FINAL)
        superseded = [event.supersedes_event_id for event in finals if event.supersedes_event_id]
        assert superseded
        assert len(superseded) == len(set(superseded))  # a provisional is replaced only once
        provisional_ids = {
            event.event_id for event in by_type(events, EventType.TRANSCRIPT_PROVISIONAL)
        }
        assert set(superseded) <= provisional_ids


class TestS04OverlapAtStartRealtime:
    def test_temporary_identities_then_revision_and_merge(
        self, calibrated_config: SasttConfig
    ) -> None:
        session, events = run_stream("s04_overlap_at_start.json", calibrated_config)

        provisional = by_type(events, EventType.TRANSCRIPT_PROVISIONAL)
        opening = [event for event in provisional if event.payload["start_ms"] == 0]
        assert len(opening) == 2
        assert all(
            str(event.payload["speaker_label"]).startswith("Temporary Speaker") for event in opening
        )

        revisions = by_type(events, EventType.TRANSCRIPT_REVISION)
        assert len(revisions) == 2
        assert all(
            str(event.payload["previous_label"]).startswith("Temporary Speaker")
            for event in revisions
        )
        assert all(
            str(event.payload["speaker_label"]).startswith("Speaker ") for event in revisions
        )

        finals = [
            event
            for event in by_type(events, EventType.TRANSCRIPT_FINAL)
            if event.payload["start_ms"] == 0
        ]
        assert len(finals) == 2
        assert {str(event.payload["speaker_label"]) for event in finals} == {
            "Speaker 1",
            "Speaker 2",
        }
        assert len({event.payload["session_speaker_id"] for event in finals}) == 2
        assert session.speakers.active_speaker_count == 2

    def test_the_session_result_keeps_both_opening_speakers(
        self, calibrated_config: SasttConfig
    ) -> None:
        session, _ = run_stream("s04_overlap_at_start.json", calibrated_config)
        opening = [segment for segment in session.result() if segment.start_ms == 0]
        assert len(opening) == 2
        assert opening[0].session_speaker_id != opening[1].session_speaker_id


class TestS12ReconnectReplay:
    def test_replay_returns_exactly_the_missed_events(self, calibrated_config: SasttConfig) -> None:
        session, events = run_stream("s02_two_speaker_overlap.json", calibrated_config)
        last_seen = 3
        replayed = session.replay(last_seen)
        assert [event.sequence_number for event in replayed] == [
            event.sequence_number for event in events if event.sequence_number > last_seen
        ]

    def test_replay_does_not_duplicate_finals(self, calibrated_config: SasttConfig) -> None:
        session, events = run_stream("s02_two_speaker_overlap.json", calibrated_config)
        finals = by_type(events, EventType.TRANSCRIPT_FINAL)
        replayed_finals = [event for event in session.replay(0) if event.is_final]
        assert len(replayed_finals) == len(finals)
        assert len({event.event_id for event in replayed_finals}) == len(replayed_finals)

    def test_finalizing_twice_emits_nothing_new(self, calibrated_config: SasttConfig) -> None:
        session, events = run_stream("s02_two_speaker_overlap.json", calibrated_config)
        assert session.finalize() == []
        assert session.log.last_sequence_number == events[-1].sequence_number

    def test_replay_from_the_current_head_is_empty(self, calibrated_config: SasttConfig) -> None:
        session, _ = run_stream("s02_two_speaker_overlap.json", calibrated_config)
        assert session.replay(session.log.last_sequence_number) == []


class TestS13IdempotentRetry:
    def test_rerunning_the_same_job_produces_the_same_structure(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        """Spec 3: a retry must not create a second, different transcript."""
        scenario = load_scenario("s02_two_speaker_overlap.json")
        payload = scenario_pcm(scenario)

        def run() -> list[tuple[int, int, str, bool]]:
            pipeline = OfflinePipeline(calibrated_config, build_adapters(scenario))
            result = pipeline.run(payload, ctx)
            return [(s.start_ms, s.end_ms, s.speaker_label, s.is_overlap) for s in result.segments]

        assert run() == run()
