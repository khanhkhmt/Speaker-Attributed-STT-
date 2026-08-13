"""Concurrent counting and the overlap router — spec 5.3."""

from __future__ import annotations

import numpy as np
import pytest

from sastt.application.overlap_router import (
    WARNING_COUNT_UNCERTAIN,
    WARNING_NARROWBAND_BETA,
    WARNING_SEPARATION_SUSPECT,
    WARNING_UNSUPPORTED_CONCURRENCY,
    CountingEvidence,
    Route,
    detect_separation_suspect,
    estimate_source_count,
    route_overlap,
)
from sastt.config import SasttConfig
from sastt.domain.audio import TimeInterval
from sastt.domain.errors import ErrorCode
from sastt.domain.speakers import OverlapRegion, SeparatedBatch, SourceCountEstimate, SourceQuality

pytestmark = pytest.mark.unit

REGION = OverlapRegion(interval=TimeInterval(1000, 3000), osd_activation=0.96)


def flagged(config: SasttConfig, **product: object) -> SasttConfig:
    return config.model_copy(update={"product": config.product.model_copy(update=product)})


class TestCounting:
    def test_falls_back_to_two_and_marks_it_uncertain(self, base_config: SasttConfig) -> None:
        estimate = estimate_source_count(CountingEvidence(), base_config)
        assert estimate.count == 2
        assert estimate.method == "fixed_two"
        assert estimate.count_uncertain is True
        assert estimate.confidence is None

    def test_confident_roster_evidence_wins(self, base_config: SasttConfig) -> None:
        estimate = estimate_source_count(
            CountingEvidence(roster_active_speakers=3, roster_confidence=0.9), base_config
        )
        assert (estimate.count, estimate.method) == (3, "ts_vad")

    def test_low_confidence_roster_is_ignored(self, base_config: SasttConfig) -> None:
        estimate = estimate_source_count(
            CountingEvidence(roster_active_speakers=3, roster_confidence=0.4), base_config
        )
        assert estimate.method == "fixed_two"

    def test_multichannel_activity_is_second_in_line(self, base_config: SasttConfig) -> None:
        estimate = estimate_source_count(
            CountingEvidence(multichannel_activity_speakers=4, multichannel_confidence=0.8),
            base_config,
        )
        assert (estimate.count, estimate.method) == (4, "multichannel_activity")

    def test_research_estimate_needs_the_research_flag(self, base_config: SasttConfig) -> None:
        evidence = CountingEvidence(research_estimate=5, research_confidence=0.9)
        assert estimate_source_count(evidence, base_config).method == "fixed_two"


class TestRouterTable:
    def test_osd_below_threshold_goes_direct(self, base_config: SasttConfig) -> None:
        decision = route_overlap(
            REGION, SourceCountEstimate(2, None, "fixed_two"), base_config, osd_positive=False
        )
        assert decision.route is Route.DIRECT_ASR
        assert decision.runs_separation is False

    def test_two_or_unknown_sources_use_the_two_source_backend(
        self, base_config: SasttConfig
    ) -> None:
        for count in (
            SourceCountEstimate(2, None, "fixed_two"),
            SourceCountEstimate(None, None, "unknown"),
        ):
            decision = route_overlap(REGION, count, base_config)
            assert decision.route is Route.SEPARATE_TWO_SOURCE
            assert decision.backend == "mossformer2_ss_16k"
            assert decision.requested_source_count == 2

    def test_uncertain_count_is_flagged(self, base_config: SasttConfig) -> None:
        decision = route_overlap(
            REGION, SourceCountEstimate(2, None, "fixed_two", count_uncertain=True), base_config
        )
        assert WARNING_COUNT_UNCERTAIN in decision.warnings

    def test_three_sources_with_beta_off_degrade_to_mixture_asr(
        self, base_config: SasttConfig
    ) -> None:
        decision = route_overlap(REGION, SourceCountEstimate(3, 0.9, "ts_vad"), base_config)
        assert decision.route is Route.MIXTURE_ASR_UNSUPPORTED
        assert decision.error_code is ErrorCode.UNSUPPORTED_CONCURRENCY
        assert WARNING_UNSUPPORTED_CONCURRENCY in decision.warnings
        assert decision.degraded is True

    def test_three_sources_with_beta_on_use_sepformer(self, base_config: SasttConfig) -> None:
        config = flagged(base_config, three_source_beta=True)
        config = config.model_copy(
            update={
                "separation": config.separation.model_copy(
                    update={"three_source_model_path": "/models/sepformer"}
                )
            }
        )
        decision = route_overlap(REGION, SourceCountEstimate(3, 0.9, "ts_vad"), config)
        assert decision.route is Route.SEPARATE_THREE_SOURCE_BETA
        assert WARNING_NARROWBAND_BETA in decision.warnings

    def test_four_or_five_mono_sources_never_load_the_research_checkpoint(
        self, base_config: SasttConfig
    ) -> None:
        for count in (4, 5):
            decision = route_overlap(
                REGION, SourceCountEstimate(count, 0.9, "multichannel_activity"), base_config
            )
            assert decision.route is Route.DEGRADED_RESEARCH_DISABLED
            assert decision.backend is None
            assert decision.degraded is True
            assert decision.error_code is ErrorCode.UNSUPPORTED_CONCURRENCY

    def test_multichannel_with_guidance_routes_to_gss(self, base_config: SasttConfig) -> None:
        config = flagged(base_config, multichannel_gss=True, max_supported_concurrent_speakers=2)
        decision = route_overlap(
            REGION,
            SourceCountEstimate(4, 0.9, "multichannel_activity"),
            config,
            evidence=CountingEvidence(is_multichannel=True, has_activity_guidance=True),
        )
        assert decision.route is Route.MULTICHANNEL_GSS
        assert decision.backend == "gpu_gss"

    def test_multichannel_without_guidance_stays_degraded(self, base_config: SasttConfig) -> None:
        config = flagged(base_config, multichannel_gss=True)
        decision = route_overlap(
            REGION,
            SourceCountEstimate(4, 0.9, "multichannel_activity"),
            config,
            evidence=CountingEvidence(is_multichannel=True, has_activity_guidance=False),
        )
        assert decision.route is Route.DEGRADED_RESEARCH_DISABLED


class TestSeparationQualityCheck:
    def _batch(self, estimated: int | None, passed: bool = True) -> SeparatedBatch:
        return SeparatedBatch(
            sources=np.zeros((2, 16), dtype=np.float32),
            sample_rate=16_000,
            requested_source_count=2,
            estimated_source_count=estimated,
            source_quality=[
                SourceQuality(speech_duration_ms=2000, passed_gate=passed),
                SourceQuality(speech_duration_ms=2000, passed_gate=True),
            ],
            separator_version="fake@1",
        )

    def test_more_sources_than_requested_marks_the_job_suspect(self) -> None:
        assert detect_separation_suspect(self._batch(3)) == (WARNING_SEPARATION_SUSPECT,)

    def test_failed_quality_gate_marks_the_job_suspect(self) -> None:
        assert detect_separation_suspect(self._batch(2, passed=False)) == (
            WARNING_SEPARATION_SUSPECT,
        )

    def test_clean_two_source_output_is_not_suspect(self) -> None:
        assert detect_separation_suspect(self._batch(2)) == ()
