"""Concurrent speaker counting and the overlap router — spec 5.3.

The router is a literal transcription of the spec 5.3 table. Two rules matter
most: only overlap regions are routed through a separator (FR-005), and a
research/beta branch is never taken to make a case "work" (spec 18 rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sastt.config import SasttConfig
from sastt.domain.errors import ErrorCode
from sastt.domain.speakers import OverlapRegion, SeparatedBatch, SourceCountEstimate

#: Warning codes surfaced on segments and in ``pipeline.warning`` events.
WARNING_UNSUPPORTED_CONCURRENCY = "unsupported_concurrency"
WARNING_RESEARCH_DISABLED = "research_checkpoint_disabled"
WARNING_NARROWBAND_BETA = "narrowband_beta"
WARNING_COUNT_UNCERTAIN = "count_uncertain"
WARNING_SEPARATION_SUSPECT = "separation_suspect"


class Route(str, Enum):
    """Router outcomes — spec 5.3 table."""

    DIRECT_ASR = "direct_asr"
    SEPARATE_TWO_SOURCE = "separate_two_source"
    SEPARATE_THREE_SOURCE_BETA = "separate_three_source_beta"
    MIXTURE_ASR_UNSUPPORTED = "mixture_asr_unsupported"
    DEGRADED_RESEARCH_DISABLED = "degraded_research_disabled"
    SEPARATE_RESEARCH_MULTIDECODER = "separate_research_multidecoder"
    MULTICHANNEL_GSS = "multichannel_gss"
    TARGET_EXTRACTION = "target_extraction"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    requested_source_count: int | None
    backend: str | None
    count: SourceCountEstimate
    warnings: tuple[str, ...] = ()
    error_code: ErrorCode | None = None
    degraded: bool = False

    @property
    def runs_separation(self) -> bool:
        return self.route in {
            Route.SEPARATE_TWO_SOURCE,
            Route.SEPARATE_THREE_SOURCE_BETA,
            Route.SEPARATE_RESEARCH_MULTIDECODER,
            Route.MULTICHANNEL_GSS,
            Route.TARGET_EXTRACTION,
        }


@dataclass(frozen=True)
class CountingEvidence:
    """Evidence available to the counter — spec 5.3 evidence order."""

    roster_active_speakers: int | None = None
    roster_confidence: float | None = None
    #: Distinct speakers the diarizer reports as active over the region. Real
    #: evidence about *how many* people are talking, but with no calibrated
    #: confidence attached, so it never claims one (spec 0.3).
    diarization_active_speakers: int | None = None
    multichannel_activity_speakers: int | None = None
    multichannel_confidence: float | None = None
    research_estimate: int | None = None
    research_confidence: float | None = None
    is_multichannel: bool = False
    has_activity_guidance: bool = False
    enrolled_targets: int = 0
    extras: dict[str, float] = field(default_factory=dict)


def estimate_source_count(evidence: CountingEvidence, config: SasttConfig) -> SourceCountEstimate:
    """Apply the evidence order of spec 5.3.

    1. enrolled roster + TS-VAD/target activity, when confident enough;
    2. multichannel activity guidance;
    3. Multi-Decoder DPRNN, research only;
    3b. diarization activity — weaker than the above because it has no
        calibrated confidence, but far better than assuming. Ranked here rather
        than higher precisely because it cannot clear a confidence gate;
    4. otherwise V1 assumes ``K = 2`` and marks the estimate uncertain.

    Rule 3b matters more than its position suggests: without it every overlap
    region reports two concurrent speakers whatever the audio contains, so the
    three-speaker rows of the routing table below are unreachable and a
    three-way overlap is silently forced through a two-source separator.
    """
    minimum = config.source_count.minimum_confidence

    if (
        evidence.roster_active_speakers is not None
        and evidence.roster_confidence is not None
        and evidence.roster_confidence >= minimum
    ):
        return SourceCountEstimate(
            count=evidence.roster_active_speakers,
            confidence=evidence.roster_confidence,
            method="ts_vad",
        )

    if (
        evidence.multichannel_activity_speakers is not None
        and evidence.multichannel_confidence is not None
        and evidence.multichannel_confidence >= minimum
    ):
        return SourceCountEstimate(
            count=evidence.multichannel_activity_speakers,
            confidence=evidence.multichannel_confidence,
            method="multichannel_activity",
        )

    if config.product.mono_four_five_source_research and evidence.research_estimate is not None:
        return SourceCountEstimate(
            count=evidence.research_estimate,
            confidence=evidence.research_confidence,
            method="multidecoder_research",
        )

    if evidence.diarization_active_speakers is not None:
        return SourceCountEstimate(
            count=max(1, evidence.diarization_active_speakers),
            # Deliberately null: diarization tells us how many, not how sure.
            confidence=None,
            method="diarization_activity",
            count_uncertain=True,
        )

    # Spec 5.3 rule 4: no evidence -> K=2, flagged uncertain, quality-checked later.
    return SourceCountEstimate(count=2, confidence=None, method="fixed_two", count_uncertain=True)


def route_overlap(
    region: OverlapRegion,
    count: SourceCountEstimate,
    config: SasttConfig,
    *,
    evidence: CountingEvidence | None = None,
    osd_positive: bool = True,
) -> RouteDecision:
    """Pick the route for one overlap region — spec 5.3 table, in table order."""
    evidence = evidence or CountingEvidence()
    warnings: list[str] = []
    if count.count_uncertain:
        warnings.append(WARNING_COUNT_UNCERTAIN)

    # Row 1: OSD below threshold -> direct ASR.
    if not osd_positive:
        return RouteDecision(
            route=Route.DIRECT_ASR,
            requested_source_count=None,
            backend=None,
            count=count,
            warnings=tuple(warnings),
        )

    k = count.count

    # Row 2: OSD positive with K=2 or unknown -> two-source separation.
    if k is None or k <= 2:
        return RouteDecision(
            route=Route.SEPARATE_TWO_SOURCE,
            requested_source_count=2,
            backend=config.separation.two_source_backend,
            count=count,
            warnings=tuple(warnings),
        )

    # Rows 3 and 4: K=3 depends on the beta flag.
    if k == 3:
        if config.product.three_source_beta:
            return RouteDecision(
                route=Route.SEPARATE_THREE_SOURCE_BETA,
                requested_source_count=3,
                backend=config.separation.three_source_backend,
                count=count,
                warnings=(*warnings, WARNING_NARROWBAND_BETA),
            )
        return RouteDecision(
            route=Route.MIXTURE_ASR_UNSUPPORTED,
            requested_source_count=None,
            backend=None,
            count=count,
            warnings=(*warnings, WARNING_UNSUPPORTED_CONCURRENCY),
            error_code=ErrorCode.UNSUPPORTED_CONCURRENCY,
            degraded=True,
        )

    # Rows 5 and 6: four or five concurrent speakers.
    if k in (4, 5):
        if (
            config.product.multichannel_gss
            and evidence.is_multichannel
            and evidence.has_activity_guidance
        ):
            return RouteDecision(
                route=Route.MULTICHANNEL_GSS,
                requested_source_count=k,
                backend="gpu_gss",
                count=count,
                warnings=tuple(warnings),
            )
        if config.product.mono_four_five_source_research:
            # R&D only; the config gate forbids this flag in production (spec 12, 20).
            return RouteDecision(
                route=Route.SEPARATE_RESEARCH_MULTIDECODER,
                requested_source_count=k,
                backend="multidecoder_dprnn",
                count=count,
                warnings=(*warnings, WARNING_RESEARCH_DISABLED),
                degraded=True,
            )
        return RouteDecision(
            route=Route.DEGRADED_RESEARCH_DISABLED,
            requested_source_count=None,
            backend=None,
            count=count,
            warnings=(*warnings, WARNING_UNSUPPORTED_CONCURRENCY, WARNING_RESEARCH_DISABLED),
            error_code=ErrorCode.UNSUPPORTED_CONCURRENCY,
            degraded=True,
        )

    # Row 7: enrolled targets with target extraction enabled.
    if config.product.target_speaker_extraction and evidence.enrolled_targets > 0:
        return RouteDecision(
            route=Route.TARGET_EXTRACTION,
            requested_source_count=min(
                evidence.enrolled_targets, config.product.max_session_speakers
            ),
            backend="wesep",
            count=count,
            warnings=tuple(warnings),
        )

    return RouteDecision(
        route=Route.DEGRADED_RESEARCH_DISABLED,
        requested_source_count=None,
        backend=None,
        count=count,
        warnings=(*warnings, WARNING_UNSUPPORTED_CONCURRENCY),
        error_code=ErrorCode.UNSUPPORTED_CONCURRENCY,
        degraded=True,
    )


def detect_separation_suspect(batch: SeparatedBatch) -> tuple[str, ...]:
    """Post-separation quality check — spec 5.3.

    Evidence that more sources were present than requested marks the job
    ``separation_suspect``. The pipeline MUST NOT silently upgrade to three
    sources while the beta flag is off.
    """
    flags: list[str] = []
    if (
        batch.estimated_source_count is not None
        and batch.estimated_source_count > batch.requested_source_count
    ) or any(not quality.passed_gate for quality in batch.source_quality):
        flags.append(WARNING_SEPARATION_SUSPECT)
    return tuple(flags)


__all__ = [
    "WARNING_COUNT_UNCERTAIN",
    "WARNING_NARROWBAND_BETA",
    "WARNING_RESEARCH_DISABLED",
    "WARNING_SEPARATION_SUSPECT",
    "WARNING_UNSUPPORTED_CONCURRENCY",
    "CountingEvidence",
    "Route",
    "RouteDecision",
    "detect_separation_suspect",
    "estimate_source_count",
    "route_overlap",
]
