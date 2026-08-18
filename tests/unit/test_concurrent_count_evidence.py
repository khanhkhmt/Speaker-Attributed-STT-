"""The concurrent-speaker counter must be told what the diarizer saw — spec 5.3.

The counter was reached with an empty evidence object, so rule 4 fired for every
overlap region: assume two speakers, flag it uncertain. That made the three- and
four-speaker rows of the routing table unreachable, and a three-way overlap was
silently pushed through a two-source separator instead of being reported as
unsupported. These tests pin the evidence flowing in and the routing that follows
from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sastt.application.offline_pipeline import OfflinePipeline, PipelineAdapters
from sastt.application.overlap_router import (
    CountingEvidence,
    Route,
    estimate_source_count,
    route_overlap,
)
from sastt.config import SasttConfig, load_config
from sastt.domain.audio import TimeInterval
from sastt.domain.errors import ErrorCode
from sastt.domain.speakers import DiarizationResult, OverlapRegion, SpeakerTurn

pytestmark = pytest.mark.unit

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def config_with(*, separation: dict[str, Any] | None = None, **product: Any) -> SasttConfig:
    overrides: dict[str, Any] = {}
    if product:
        overrides["product"] = product
    if separation:
        overrides["separation"] = separation
    return load_config(
        CONFIG_PATH, environment="development", manifest_dir=None, overrides=overrides
    )


def diarization_with(*spans: tuple[str, int, int]) -> DiarizationResult:
    turns = [
        SpeakerTurn(cluster_id=cluster, interval=TimeInterval(start, end))
        for cluster, start, end in spans
    ]
    return DiarizationResult(
        turns=turns,
        regular_tracks=turns,
        exclusive_tracks=None,
        overlap_regions=[],
        estimated_session_speakers=len({c for c, _, _ in spans}),
        model_version="test@1",
    )


def pipeline_for(config: SasttConfig) -> OfflinePipeline:
    return OfflinePipeline(config, PipelineAdapters.__new__(PipelineAdapters))


class TestCountingEvidenceFromDiarization:
    def test_three_concurrent_clusters_are_counted_as_three(self) -> None:
        diarization = diarization_with(
            ("cluster_a", 0, 5_000), ("cluster_b", 1_000, 6_000), ("cluster_c", 2_000, 4_000)
        )

        evidence = pipeline_for(config_with()).counting_evidence(
            diarization, TimeInterval(2_500, 3_500)
        )

        assert evidence.diarization_active_speakers == 3

    def test_speakers_outside_the_window_are_not_counted(self) -> None:
        diarization = diarization_with(
            ("cluster_a", 0, 5_000), ("cluster_b", 1_000, 6_000), ("cluster_c", 30_000, 40_000)
        )

        evidence = pipeline_for(config_with()).counting_evidence(
            diarization, TimeInterval(2_500, 3_500)
        )

        assert evidence.diarization_active_speakers == 2

    def test_no_diarization_offers_no_evidence(self) -> None:
        evidence = pipeline_for(config_with()).counting_evidence(None, TimeInterval(0, 1_000))

        assert evidence.diarization_active_speakers is None

    def test_a_window_with_no_turns_offers_no_evidence(self) -> None:
        diarization = diarization_with(("cluster_a", 0, 1_000))

        evidence = pipeline_for(config_with()).counting_evidence(
            diarization, TimeInterval(50_000, 51_000)
        )

        assert evidence.diarization_active_speakers is None


class TestEstimateUsesTheEvidence:
    def test_three_active_speakers_produce_a_count_of_three(self) -> None:
        estimate = estimate_source_count(
            CountingEvidence(diarization_active_speakers=3), config_with()
        )

        assert estimate.count == 3
        assert estimate.method == "diarization_activity"

    def test_it_never_invents_a_confidence(self) -> None:
        """Diarization says how many, not how sure — spec 0.3 forbids the rest."""
        estimate = estimate_source_count(
            CountingEvidence(diarization_active_speakers=3), config_with()
        )

        assert estimate.confidence is None
        assert estimate.count_uncertain is True

    def test_without_evidence_the_old_fallback_still_applies(self) -> None:
        estimate = estimate_source_count(CountingEvidence(), config_with())

        assert estimate.count == 2
        assert estimate.method == "fixed_two"

    def test_a_confident_roster_still_outranks_diarization(self) -> None:
        estimate = estimate_source_count(
            CountingEvidence(
                roster_active_speakers=4, roster_confidence=0.9, diarization_active_speakers=2
            ),
            config_with(),
        )

        assert estimate.count == 4
        assert estimate.method == "ts_vad"


class TestRoutingNowReachesTheThreeSpeakerRows:
    def _region(self) -> OverlapRegion:
        return OverlapRegion(
            interval=TimeInterval(1_000, 2_000), osd_activation=0.9, model_version="test@1"
        )

    def test_three_speakers_without_the_beta_flag_are_reported_unsupported(self) -> None:
        """Previously unreachable: the counter always said two, so this never fired."""
        config = config_with()
        estimate = estimate_source_count(CountingEvidence(diarization_active_speakers=3), config)

        decision = route_overlap(self._region(), estimate, config, osd_positive=True)

        assert decision.route is Route.MIXTURE_ASR_UNSUPPORTED
        assert decision.error_code is ErrorCode.UNSUPPORTED_CONCURRENCY
        assert decision.degraded is True
        assert not decision.runs_separation, "a 2-source separator must not be handed 3 speakers"

    def test_three_speakers_with_the_beta_flag_route_to_the_three_source_path(self) -> None:
        # The config gate refuses the beta flag without a staged checkpoint, so
        # the path is supplied here; the routing decision is what is under test.
        config = config_with(
            max_supported_concurrent_speakers=3,
            three_source_beta=True,
            separation={"three_source_model_path": "/models/sepformer-libri3mix"},
        )
        estimate = estimate_source_count(CountingEvidence(diarization_active_speakers=3), config)

        decision = route_overlap(self._region(), estimate, config, osd_positive=True)

        assert decision.route is Route.SEPARATE_THREE_SOURCE_BETA
        assert decision.requested_source_count == 3

    def test_two_speakers_still_take_the_two_source_path(self) -> None:
        config = config_with()
        estimate = estimate_source_count(CountingEvidence(diarization_active_speakers=2), config)

        decision = route_overlap(self._region(), estimate, config, osd_positive=True)

        assert decision.route is Route.SEPARATE_TWO_SOURCE
        assert decision.requested_source_count == 2

    def test_a_single_active_speaker_does_not_become_zero_sources(self) -> None:
        config = config_with()
        estimate = estimate_source_count(CountingEvidence(diarization_active_speakers=1), config)

        decision = route_overlap(self._region(), estimate, config, osd_positive=True)

        assert estimate.count == 1
        assert decision.route is Route.SEPARATE_TWO_SOURCE
