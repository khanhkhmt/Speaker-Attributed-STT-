"""Offline pipeline scenarios over fake adapters — spec 16.2 (S01-S04, S11).

No model weights are loaded (spec 16.1.3). These assert structure and behaviour,
never accuracy: spec 19.1 states the deterministic harness cannot evaluate model
quality.
"""

from __future__ import annotations

import pytest

from sastt.api.schemas import validate_segment_v2
from sastt.application.offline_pipeline import OfflinePipeline
from sastt.config import SasttConfig
from sastt.domain.events import JobState
from sastt.domain.speakers import IdentityStatus
from sastt.domain.transcript import TranscriptSegment
from sastt.observability import CallContext

pytestmark = pytest.mark.integration

from conftest import build_adapters, load_scenario, scenario_pcm  # noqa: E402


def run_offline(
    scenario_name: str,
    config: SasttConfig,
    ctx: CallContext,
    **adapter_kwargs: object,
):
    scenario = load_scenario(scenario_name)
    adapters = build_adapters(scenario, **adapter_kwargs)  # type: ignore[arg-type]
    pipeline = OfflinePipeline(config, adapters)
    return scenario, pipeline.run(scenario_pcm(scenario), ctx)


def concurrent_pairs(
    segments: list[TranscriptSegment],
) -> list[tuple[TranscriptSegment, TranscriptSegment]]:
    return [
        (first, second)
        for index, first in enumerate(segments)
        for second in segments[index + 1 :]
        if first.interval.intersects(second.interval)
        and first.session_speaker_id != second.session_speaker_id
    ]


class TestS01FiveSpeakersNoOverlap:
    def test_five_stable_speakers(self, calibrated_config: SasttConfig, ctx: CallContext) -> None:
        _, result = run_offline("s01_five_speakers_no_overlap.json", calibrated_config, ctx)
        assert result.state is JobState.SUCCEEDED
        assert result.estimated_session_speakers == 5
        assert len({segment.session_speaker_id for segment in result.segments}) == 5
        assert {segment.speaker_label for segment in result.segments} == {
            f"Speaker {index}" for index in range(1, 6)
        }
        assert concurrent_pairs(result.segments) == []
        assert all(segment.is_overlap is False for segment in result.segments)
        assert all(segment.is_final for segment in result.segments)

    def test_output_matches_the_public_schema(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, result = run_offline("s01_five_speakers_no_overlap.json", calibrated_config, ctx)
        for segment in result.segments:
            validate_segment_v2(segment.to_public_dict())
            assert segment.confidences.status == "uncalibrated"
            assert segment.model_versions.calibration is None

    def test_labels_are_stable_across_runs(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, first = run_offline("s01_five_speakers_no_overlap.json", calibrated_config, ctx)
        _, second = run_offline("s01_five_speakers_no_overlap.json", calibrated_config, ctx)
        assert [s.speaker_label for s in first.segments] == [
            s.speaker_label for s in second.segments
        ]
        assert [(s.start_ms, s.end_ms) for s in first.segments] == [
            (s.start_ms, s.end_ms) for s in second.segments
        ]


class TestS02TwoSpeakerOverlap:
    def test_two_transcripts_share_the_overlap_window(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, result = run_offline("s02_two_speaker_overlap.json", calibrated_config, ctx)
        overlapping = [segment for segment in result.segments if segment.is_overlap]
        assert len(overlapping) == 2
        assert overlapping[0].interval.intersects(overlapping[1].interval)
        assert overlapping[0].session_speaker_id != overlapping[1].session_speaker_id
        assert {segment.source_track for segment in overlapping} == {0, 1}
        assert all(segment.separation_backend for segment in overlapping)

    def test_the_second_speaker_is_never_dropped(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, result = run_offline("s02_two_speaker_overlap.json", calibrated_config, ctx)
        assert len(concurrent_pairs(result.segments)) >= 1

    def test_uncertain_source_count_is_reported(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, result = run_offline("s02_two_speaker_overlap.json", calibrated_config, ctx)
        assert "count_uncertain" in result.warnings
        overlapping = [segment for segment in result.segments if segment.is_overlap]
        assert all(segment.estimated_concurrent_speakers == 2 for segment in overlapping)
        assert all(segment.count_confidence is None for segment in overlapping)

    def test_uncalibrated_thresholds_fail_closed_to_provisional(
        self, base_config: SasttConfig, ctx: CallContext
    ) -> None:
        """Spec 18 rule 7: no thresholds -> no identity claim, but no lost audio."""
        _, result = run_offline("s02_two_speaker_overlap.json", base_config, ctx)
        overlapping = [segment for segment in result.segments if segment.is_overlap]
        assert len(overlapping) == 2
        assert all(
            segment.identity_status in (IdentityStatus.PROVISIONAL, IdentityStatus.UNKNOWN)
            for segment in overlapping
        )


class TestS03SourceOrderSwap:
    def test_labels_do_not_swap_when_the_separator_reorders_sources(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        scenario, result = run_offline("s03_source_order_swap.json", calibrated_config, ctx)
        overlapping = [segment for segment in result.segments if segment.is_overlap]
        assert len(overlapping) == 4

        by_text = {segment.text.split()[-1]: segment for segment in overlapping}
        first_round = [s for s in overlapping if s.start_ms < 16_000]
        second_round = [s for s in overlapping if s.start_ms >= 16_000]

        def speaker_of(segments: list[TranscriptSegment], marker: str) -> str:
            return next(s.session_speaker_id for s in segments if s.text.endswith(marker))

        # The same person keeps the same session speaker across both rounds,
        # even though the separator emitted them on different source tracks.
        assert speaker_of(first_round, "một") == speaker_of(second_round, "một")
        assert speaker_of(first_round, "hai") == speaker_of(second_round, "hai")
        assert by_text["một"].session_speaker_id != by_text["hai"].session_speaker_id

    def test_source_track_actually_changed_between_rounds(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, result = run_offline("s03_source_order_swap.json", calibrated_config, ctx)
        overlapping = [segment for segment in result.segments if segment.is_overlap]
        tracks = {
            (segment.text.split()[-1], segment.start_ms < 16_000): segment.source_track
            for segment in overlapping
        }
        assert tracks[("một", True)] != tracks[("một", False)]


class TestS04OverlapAtStart:
    def test_offline_resolves_the_opening_overlap_from_later_clean_speech(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, result = run_offline("s04_overlap_at_start.json", calibrated_config, ctx)
        opening = [segment for segment in result.segments if segment.start_ms == 0]
        assert len(opening) == 2
        assert opening[0].session_speaker_id != opening[1].session_speaker_id
        assert all(segment.identity_status is IdentityStatus.ANONYMOUS for segment in opening)
        assert all(segment.speaker_label.startswith("Speaker ") for segment in opening)

    def test_no_speaker_is_invented_beyond_the_session_cap(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        _, result = run_offline("s04_overlap_at_start.json", calibrated_config, ctx)
        speakers = {segment.session_speaker_id for segment in result.segments}
        assert len(speakers) == 2
        assert len(speakers) <= calibrated_config.product.max_session_speakers


class TestS11DegradedSeparation:
    def test_separator_failure_retries_then_degrades_without_losing_audio(
        self, calibrated_config: SasttConfig, ctx: CallContext
    ) -> None:
        """Spec 15: retry once with a smaller crop, then degraded mixture ASR."""
        _, result = run_offline(
            "s02_two_speaker_overlap.json",
            calibrated_config,
            ctx,
            fail_first_separation=True,
        )
        assert result.succeeded
        assert result.segments
        assert any(segment.is_overlap for segment in result.segments)
        for segment in result.segments:
            validate_segment_v2(segment.to_public_dict())
