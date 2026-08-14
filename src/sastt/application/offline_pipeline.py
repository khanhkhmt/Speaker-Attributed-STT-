"""Offline/file pipeline — spec 4.1, 8.1, 15.

Order of work follows the spec 4.1 diagram: decode, diarize + OSD, build global
speaker prototypes from the cleanest non-overlap speech, then route only the
overlap regions through a separator, link the separated sources back to those
global prototypes, and finally fuse.

Because the whole file is available, overlap at the very beginning is resolved
against prototypes built from clean speech that appears later (spec 4.1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from sastt.application.fusion import (
    ConfidenceCalibrator,
    FusionEngine,
    NullConfidenceCalibrator,
    WordGroup,
)
from sastt.application.overlap_router import (
    CountingEvidence,
    Route,
    RouteDecision,
    detect_separation_suspect,
    estimate_source_count,
    route_overlap,
)
from sastt.application.session_state import SessionSpeakerState, overlapping_clusters
from sastt.application.source_linking import link_sources
from sastt.config import SasttConfig
from sastt.domain.audio import (
    AudioBuffer,
    TimeInterval,
    merge_intervals,
    seconds_to_ms,
)
from sastt.domain.errors import (
    AudioTooLongError,
    InsufficientSpeechForEmbeddingError,
    SasttError,
    SeparationFailedError,
)
from sastt.domain.events import JobRecord, JobState
from sastt.domain.speakers import (
    DiarizationResult,
    EmbeddingOrigin,
    IdentityState,
    OverlapRegion,
    SeparatedBatch,
    SpeakerEmbedding,
)
from sastt.domain.transcript import ModelVersions, TranscriptSegment, Word, sort_segments
from sastt.observability import (
    METRIC_AUDIO_SECONDS,
    METRIC_DEGRADED_SESSION,
    METRIC_OVERLAP_SECONDS,
    CallContext,
)
from sastt.ports.asr import SpeechRecognizer
from sastt.ports.audio import AudioDecoder
from sastt.ports.diarization import Diarizer, OverlapDetector, VoiceActivityDetector
from sastt.ports.embedding import SpeakerEmbedder
from sastt.ports.registry import VoiceRegistry
from sastt.ports.separation import SpeechSeparator

MAX_CLEAN_PROTOTYPE_SEGMENTS = 3
# A physical plausibility guard, not a calibrated ASR-confidence threshold. A
# source can contain a valid one-word acknowledgement in a few hundred ms, but
# four or more tokens require this much VAD-confirmed speech and timestamp span.
# It keeps a separator artefact from becoming a convincing-looking transcript.
MIN_SOURCE_SPEECH_MS_WITH_TRANSCRIPT = 100
MIN_SOURCE_SPEECH_MS_PER_TOKEN = 60
MIN_TOKENS_FOR_DENSITY_GATE = 4
WARNING_UNRELIABLE_SEPARATED_TRANSCRIPT = "unreliable_separated_transcript"
WARNING_UNRELIABLE_MIXTURE_TRANSCRIPT = "unreliable_mixture_transcript"
WARNING_UNRELIABLE_NON_OVERLAP_TRANSCRIPT = "unreliable_non_overlap_transcript"


def _report_stage(callback: Callable[[JobState], None] | None, state: JobState) -> None:
    """Publish an offline job state at the moment its work begins (spec 8.1)."""
    if callback is not None:
        callback(state)


@dataclass
class PipelineAdapters:
    """The model ports one pipeline run needs (spec 9)."""

    decoder: AudioDecoder
    vad: VoiceActivityDetector
    diarizer: Diarizer
    overlap_detector: OverlapDetector
    recognizer: SpeechRecognizer
    embedder: SpeakerEmbedder
    separator: SpeechSeparator | None = None
    three_source_separator: SpeechSeparator | None = None
    registry: VoiceRegistry | None = None


@dataclass
class OfflineResult:
    """Result of one offline job."""

    job_id: str
    state: JobState
    segments: list[TranscriptSegment]
    config_version: str
    model_versions: ModelVersions
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False
    estimated_session_speakers: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.state in (JobState.SUCCEEDED, JobState.DEGRADED_SUCCEEDED)


class OfflinePipeline:
    """Runs one file end to end."""

    def __init__(
        self,
        config: SasttConfig,
        adapters: PipelineAdapters,
        *,
        tenant_id: str = "tenant_local",
        calibrator: ConfidenceCalibrator | None = None,
        allow_provisional_continuity: bool = True,
    ) -> None:
        self.config = config
        self.adapters = adapters
        self.tenant_id = tenant_id
        self.calibrator = calibrator or NullConfidenceCalibrator()
        self.allow_provisional_continuity = allow_provisional_continuity

    # -- entry point --------------------------------------------------------- #

    def run(
        self,
        payload: bytes,
        ctx: CallContext,
        *,
        job: JobRecord | None = None,
        session_id: str | None = None,
        state: SessionSpeakerState | None = None,
        on_stage: Callable[[JobState], None] | None = None,
    ) -> OfflineResult:
        warnings: list[str] = []
        model_versions = ModelVersions(
            diarization=self.adapters.diarizer.model_version,
            embedding=self.adapters.embedder.model_version,
            asr=self.adapters.recognizer.model_version,
            separation=None,
            calibration=self.calibrator.calibration_version,
        )
        job_id = job.job_id if job else "job_local"
        session = session_id or job_id

        # --- PREPROCESSING (spec 5.1) --------------------------------------- #
        _report_stage(on_stage, JobState.PREPROCESSING)
        asset = self.adapters.decoder.decode(payload, ctx.child("decode"))
        max_ms = int(self.config.audio.max_file_hours * 3_600_000)
        if asset.duration_ms > max_ms:
            raise AudioTooLongError(
                f"{asset.duration_ms} ms exceeds the {max_ms} ms limit",
                details={"duration_ms": asset.duration_ms},
            )
        ctx.metrics.increment(METRIC_AUDIO_SECONDS, asset.duration_ms / 1000.0, mode="batch")
        mono = asset.mono_16k

        # --- DIARIZING (spec 5.2) ------------------------------------------- #
        _report_stage(on_stage, JobState.DIARIZING)
        diarization = self.adapters.diarizer.diarize(
            mono,
            ctx.child("diarization"),
            min_speakers=self.config.diarization.min_speakers,
            max_speakers=self.config.diarization.max_speakers,
        )
        osd_regions = self.adapters.overlap_detector.detect(mono, ctx.child("osd"))
        overlap_regions = _merge_overlap_regions(
            diarization.overlap_regions,
            osd_regions,
            merge_gap_ms=seconds_to_ms(self.config.overlap_detection.merge_gap_seconds),
            min_duration_ms=seconds_to_ms(self.config.overlap_detection.min_duration_seconds),
        )
        ctx.metrics.increment(
            METRIC_OVERLAP_SECONDS,
            sum(r.interval.duration_ms for r in overlap_regions) / 1000.0,
            route="offline",
        )

        # --- global speakers and prototypes (spec 5.7) ---------------------- #
        state = state or SessionSpeakerState(
            session_id=session,
            config=self.config,
            embedding_model_version=self.adapters.embedder.model_version,
        )
        self.build_global_speakers(state, diarization, overlap_regions, mono, ctx)

        # --- non-overlap ASR (spec 5.5, FR-005) ------------------------------ #
        _report_stage(on_stage, JobState.TRANSCRIBING)
        speech = self.adapters.vad.detect(mono, ctx.child("vad"))
        groups: list[WordGroup] = []
        groups.extend(
            self.transcribe_non_overlap(mono, speech, overlap_regions, ctx, warnings=warnings)
        )

        # --- overlap routing and separation (spec 5.3, 5.4, 5.8) ------------- #
        degraded = bool(warnings)
        if overlap_regions:
            _report_stage(on_stage, JobState.SEPARATING)
        for region in overlap_regions:
            count = estimate_source_count(CountingEvidence(), self.config)
            decision = route_overlap(region, count, self.config, osd_positive=True)
            region_groups, region_warnings, region_degraded = self.handle_overlap_region(
                mono, region, decision, state, ctx
            )
            groups.extend(region_groups)
            warnings.extend(region_warnings)
            degraded = degraded or region_degraded

        # --- reconciliation and Voice ID (spec 5.9, 5.10) -------------------- #
        _report_stage(on_stage, JobState.LINKING)
        self.reconcile_temporary_speakers(state, ctx)
        self.apply_voice_id(state, ctx)
        state.finalize_unresolved()

        # --- FUSING (spec 5.11) ---------------------------------------------- #
        _report_stage(on_stage, JobState.FUSING)
        fusion = FusionEngine(
            session_id=session,
            state=state,
            config=self.config,
            model_versions=model_versions,
            calibrator=self.calibrator,
        )
        segments = sort_segments(
            fusion.fuse(groups, diarization.regular_tracks, is_final=True, revision=1)
        )

        if degraded:
            ctx.metrics.increment(METRIC_DEGRADED_SESSION, reason="separation_degraded")
        return OfflineResult(
            job_id=job_id,
            state=JobState.DEGRADED_SUCCEEDED if degraded else JobState.SUCCEEDED,
            segments=segments,
            config_version=self.config.config_version,
            model_versions=model_versions,
            warnings=sorted(set(warnings)),
            degraded=degraded,
            estimated_session_speakers=diarization.estimated_session_speakers,
            diagnostics={
                "overlap_regions": len(overlap_regions),
                "speech_ms": sum(interval.duration_ms for interval in speech),
                "session_speakers": state.active_speaker_count,
                "input_sha256": asset.input_sha256,
            },
        )

    # -- stages (public: the streaming pipeline reuses them) ------------------ #

    def build_global_speakers(
        self,
        state: SessionSpeakerState,
        diarization: DiarizationResult,
        overlap_regions: list[OverlapRegion],
        mono: AudioBuffer,
        ctx: CallContext,
    ) -> None:
        """Spec 5.7 offline steps 1-3: cluster keys, clean segments, centroids."""
        per_cluster: dict[str, list[TimeInterval]] = {}
        for turn in diarization.regular_tracks:
            per_cluster.setdefault(turn.cluster_id, []).append(turn.interval)

        for cluster_id in diarization.cluster_ids():
            state.ensure_cluster_speaker(cluster_id)

        # Clusters active at the same time can never be merged (spec 5.6).
        for first, second in overlapping_clusters(per_cluster):
            first_speaker = state.by_cluster(first)
            second_speaker = state.by_cluster(second)
            if first_speaker and second_speaker:
                state.add_cannot_link(
                    first_speaker.session_speaker_id, second_speaker.session_speaker_id
                )

        for cluster_id, intervals in per_cluster.items():
            speaker = state.by_cluster(cluster_id)
            if speaker is None:
                continue
            clean = _subtract(merge_intervals(intervals), [r.interval for r in overlap_regions])
            clean.sort(key=lambda interval: interval.duration_ms, reverse=True)
            for interval in clean[:MAX_CLEAN_PROTOTYPE_SEGMENTS]:
                if interval.duration_ms < state.minimum_speech_ms:
                    continue
                embedding = self.embed_buffer(mono, interval, ctx, origin="clean")
                if embedding is not None:
                    state.update_prototype(speaker.session_speaker_id, embedding)

    def transcribe_non_overlap(
        self,
        mono: AudioBuffer,
        speech: list[TimeInterval],
        overlap_regions: list[OverlapRegion],
        ctx: CallContext,
        *,
        warnings: list[str] | None = None,
    ) -> list[WordGroup]:
        """Direct ASR on the non-overlap parts only (FR-005, FR-006)."""
        groups: list[WordGroup] = []
        for interval in _subtract(speech, [region.interval for region in overlap_regions]):
            if interval.duration_ms < 50:
                continue
            crop = mono.crop_ms(interval)
            result = self.adapters.recognizer.transcribe(
                crop, ctx.child("asr"), language=self.config.asr.language
            )
            words = words_owned_by(result.words, interval)
            if not words:
                continue
            if not is_plausible_transcript(tuple(words), [interval]):
                if warnings is not None:
                    warnings.append(WARNING_UNRELIABLE_NON_OVERLAP_TRANSCRIPT)
                continue
            groups.append(
                WordGroup(
                    words=tuple(words),
                    interval=interval,
                    is_overlap=False,
                    raw_scores=dict(result.raw_scores),
                )
            )
        return groups

    def handle_overlap_region(
        self,
        mono: AudioBuffer,
        region: OverlapRegion,
        decision: RouteDecision,
        state: SessionSpeakerState,
        ctx: CallContext,
    ) -> tuple[list[WordGroup], list[str], bool]:
        warnings = list(decision.warnings)
        if not decision.runs_separation or self.adapters.separator is None:
            return (
                self.mixture_fallback(mono, region, decision, warnings),
                warnings,
                decision.degraded or self.adapters.separator is None,
            )

        separator = self._separator_for(decision)
        if separator is None:
            warnings.append("separator_unavailable")
            return self.mixture_fallback(mono, region, decision, warnings), warnings, True

        context_ms = seconds_to_ms(self.config.audio.overlap_context_seconds)
        padded = region.interval.pad(
            context_ms, lower_bound_ms=mono.start_ms, upper_bound_ms=mono.end_ms
        )
        try:
            groups, extra_warnings, separation_degraded = self.separate_and_link(
                mono, region, padded, decision, separator, state, ctx
            )
        except SeparationFailedError:
            # Spec 15: retry once with a smaller crop, then fall back to mixture ASR.
            try:
                groups, extra_warnings, separation_degraded = self.separate_and_link(
                    mono, region, region.interval, decision, separator, state, ctx
                )
            except SeparationFailedError:
                warnings.append("separation_failed_degraded")
                ctx.metrics.increment(METRIC_DEGRADED_SESSION, reason="separation_failed")
                return self.mixture_fallback(mono, region, decision, warnings), warnings, True
        warnings.extend(extra_warnings)
        return groups, warnings, decision.degraded or separation_degraded

    def _separator_for(self, decision: RouteDecision) -> SpeechSeparator | None:
        if decision.route is Route.SEPARATE_THREE_SOURCE_BETA:
            return self.adapters.three_source_separator
        return self.adapters.separator

    def separate_and_link(
        self,
        mono: AudioBuffer,
        region: OverlapRegion,
        padded: TimeInterval,
        decision: RouteDecision,
        separator: SpeechSeparator,
        state: SessionSpeakerState,
        ctx: CallContext,
    ) -> tuple[list[WordGroup], list[str], bool]:
        """Separate one region, link its sources, and emit one group per source."""
        requested = decision.requested_source_count or 2
        warnings: list[str] = []
        groups: list[WordGroup] = []
        previous_mapping: dict[int, str] | None = None

        for window, owned in _split_crop(
            padded,
            region.interval,
            max_crop_ms=seconds_to_ms(self.config.separation.max_crop_seconds),
            stitch_ms=seconds_to_ms(self.config.separation.stitch_overlap_seconds),
        ):
            crop = mono.crop_ms(window)
            batch = separator.separate(
                crop, ctx.child("separation"), requested_source_count=requested
            )
            flags = detect_separation_suspect(batch)
            if flags:
                warnings.extend(flags)

            embeddings: list[SpeakerEmbedding | None] = []
            source_asr_buffers: list[AudioBuffer] = []
            source_speech: list[list[TimeInterval]] = []
            for index in range(batch.source_count):
                source = _source_buffer(batch, index, crop)
                asr_buffer = source.crop_ms(owned) if owned.duration_ms > 0 else source
                speech = self.adapters.vad.detect(asr_buffer, ctx.child("vad"))
                source_asr_buffers.append(asr_buffer)
                source_speech.append(speech)
                embeddings.append(
                    self.embed_buffer(
                        source,
                        owned,
                        ctx,
                        source_track=index,
                        speech_intervals=speech,
                    )
                )

            result = link_sources(
                embeddings,
                (
                    state.linking_prototypes()
                    if self.allow_provisional_continuity
                    else state.prototypes()
                ),
                self.config.source_linking,
                previous_mapping=previous_mapping,
                ctx=ctx,
            )
            previous_mapping = result.mapping()
            assigned_in_crop: list[str] = []

            for index, link in enumerate(result.decisions):
                asr = self.adapters.recognizer.transcribe(
                    source_asr_buffers[index],
                    ctx.child("asr"),
                    language=self.config.asr.language,
                    source_track=index,
                )
                words = tuple(words_owned_by(asr.words, owned))
                if not words:
                    continue
                if not is_plausible_transcript(words, source_speech[index]):
                    # The source has no VAD-confirmed speech, or its ASR text is
                    # physically impossible for that amount of speech. Do not
                    # silently drop the original overlap: retry it as a single
                    # degraded mixture, which has no invented source identity.
                    warnings.append(WARNING_UNRELIABLE_SEPARATED_TRANSCRIPT)
                    return (
                        self.mixture_fallback(mono, region, decision, warnings),
                        warnings,
                        True,
                    )

                speaker_id = link.session_speaker_id
                if speaker_id is None:
                    # Spec 5.9: no confident link -> temporary identity, reconciled
                    # later. That only makes sense when the source carries evidence
                    # reconciliation can actually use. A source too short or too
                    # poor to embed can never be merged back, so minting an ID for
                    # it would strand a permanent `Unknown` speaker in the roster
                    # and push the session past max_session_speakers. Spec 15 says
                    # such a source is simply Unknown, and spec 0.1.1 caps the
                    # session at five speakers.
                    embedding = embeddings[index]
                    usable = (
                        embedding is not None
                        and embedding.speech_duration_ms >= state.minimum_speech_ms
                    )
                    if usable and state.can_admit_speaker:
                        temporary = state.create_temporary_speaker()
                        speaker_id = temporary.session_speaker_id
                        state.buffer_provisional_embedding(
                            speaker_id,
                            embedding,  # type: ignore[arg-type]
                        )
                    else:
                        warnings.append("unidentified_source")
                elif embeddings[index] is not None:
                    speaker = state.get(speaker_id)
                    if speaker.state is IdentityState.PROVISIONAL:
                        # A temporary prototype is continuity evidence only.  It
                        # is promoted at finalization, never merged by guesswork.
                        state.buffer_provisional_embedding(
                            speaker_id,
                            embeddings[index],  # type: ignore[arg-type]
                        )
                    else:
                        state.update_prototype(
                            speaker_id,
                            embeddings[index],  # type: ignore[arg-type]
                            link_margin=link.margin,
                            source_quality_passed=batch.source_quality[index].passed_gate,
                        )

                # Sources of one crop are concurrent by construction, so they are
                # different people and can never be merged later (spec 5.6, 5.8.7).
                # Without this the temporary identity of a rejected source is free
                # to be reconciled into the speaker another source already holds.
                # An unidentified source has no ID to constrain.
                if speaker_id is not None:
                    for other_id in assigned_in_crop:
                        state.add_cannot_link(other_id, speaker_id)
                    assigned_in_crop.append(speaker_id)

                raw_scores = dict(asr.raw_scores)
                if link.score is not None:
                    raw_scores["speaker_cosine_similarity"] = link.score
                if region.osd_activation is not None:
                    raw_scores["osd_activation"] = region.osd_activation

                groups.append(
                    WordGroup(
                        words=words,
                        interval=owned,
                        is_overlap=True,
                        source_track=index,
                        separation_backend=separator.backend,
                        session_speaker_id=speaker_id,
                        estimated_concurrent_speakers=decision.count.count,
                        count_confidence=decision.count.confidence,
                        raw_scores=raw_scores,
                        quality_flags=tuple(decision.warnings) + flags,
                        degraded_mode=decision.degraded,
                    )
                )
        return groups, warnings, False

    def mixture_fallback(
        self,
        mono: AudioBuffer,
        region: OverlapRegion,
        decision: RouteDecision,
        warnings: list[str],
    ) -> list[WordGroup]:
        """Mixture ASR for regions that must not be separated (spec 5.3, 15).

        The audio is transcribed rather than dropped, and the result is marked
        degraded — it is never presented as two separated speakers.
        """
        ctx = CallContext(stage="asr_mixture")
        crop = mono.crop_ms(region.interval)
        result = self.adapters.recognizer.transcribe(crop, ctx, language=self.config.asr.language)
        words = tuple(words_owned_by(result.words, region.interval))
        if not words:
            return []
        speech = self.adapters.vad.detect(crop, ctx.child("vad"))
        if not is_plausible_transcript(words, speech):
            warnings.append(WARNING_UNRELIABLE_MIXTURE_TRANSCRIPT)
            return []
        raw_scores = dict(result.raw_scores)
        if region.osd_activation is not None:
            raw_scores["osd_activation"] = region.osd_activation
        return [
            WordGroup(
                words=words,
                interval=region.interval,
                is_overlap=True,
                session_speaker_id=None,
                estimated_concurrent_speakers=decision.count.count,
                count_confidence=decision.count.confidence,
                raw_scores=raw_scores,
                quality_flags=tuple(sorted(set(warnings))),
                degraded_mode=True,
            )
        ]

    def reconcile_temporary_speakers(self, state: SessionSpeakerState, ctx: CallContext) -> None:
        """Second pass over provisional identities — spec 5.9 step 5."""
        globals_ = [
            speaker
            for speaker in state.active
            if speaker.prototype is not None and speaker.state is not IdentityState.PROVISIONAL
        ]
        if not globals_:
            return
        prototypes = [speaker.prototype for speaker in globals_ if speaker.prototype]
        for provisional in state.provisional_speakers():
            if provisional.prototype is None:
                continue
            pseudo = SpeakerEmbedding(
                vector=provisional.prototype.centroid,
                model_version=provisional.prototype.model_version,
                quality=1.0,
                speech_duration_ms=state.minimum_speech_ms,
                origin="separated",
            )
            result = link_sources([pseudo], prototypes, self.config.source_linking, ctx=ctx)
            decision = result.decisions[0]
            if decision.status != "linked" or decision.session_speaker_id is None:
                continue
            if not state.can_merge(provisional.session_speaker_id, decision.session_speaker_id):
                continue
            state.merge(
                provisional.session_speaker_id,
                decision.session_speaker_id,
                "overlap_start_reconciliation",
            )

    def apply_voice_id(self, state: SessionSpeakerState, ctx: CallContext) -> None:
        """Open-set Voice ID over the session prototypes (spec 5.10).

        Fails closed: an uncalibrated registry leaves session labels untouched.
        """
        registry = self.adapters.registry
        if registry is None or not self.config.voice_id.enabled:
            return
        for speaker in state.active:
            if speaker.prototype is None:
                continue
            embedding = SpeakerEmbedding(
                vector=speaker.prototype.centroid,
                model_version=speaker.prototype.model_version,
                quality=1.0,
                speech_duration_ms=state.minimum_speech_ms,
                origin="clean",
            )
            decision = registry.identify(self.tenant_id, embedding, ctx.child("voice_id"))
            state.apply_voice_id(speaker.session_speaker_id, decision)

    # -- helpers ------------------------------------------------------------- #

    def embed_buffer(
        self,
        buffer: AudioBuffer,
        interval: TimeInterval,
        ctx: CallContext,
        *,
        origin: EmbeddingOrigin = "separated",
        source_track: int | None = None,
        speech_intervals: list[TimeInterval] | None = None,
    ) -> SpeakerEmbedding | None:
        """Embed a slice of clean or separated speech, honouring the gate of spec 5.6."""
        try:
            crop = buffer.crop_ms(interval)
        except SasttError:
            return None
        speech = (
            speech_intervals
            if speech_intervals is not None
            else self.adapters.vad.detect(crop, ctx.child("vad"))
        )
        try:
            return self.adapters.embedder.embed(
                crop,
                ctx.child("embedding"),
                speech_intervals=speech,
                origin=origin,
                source_track=source_track,
            )
        except InsufficientSpeechForEmbeddingError:
            return None


def is_plausible_transcript(words: tuple[Word, ...], speech_intervals: list[TimeInterval]) -> bool:
    """Reject text impossible for the VAD-confirmed speech in an ASR input.

    This is intentionally independent of Whisper's raw probability: that score
    can be high for a language-model hallucination. It is also deliberately
    permissive for one-to-three-token acknowledgements, which may be genuine
    short speech.
    """
    speech_ms = sum(interval.duration_ms for interval in speech_intervals)
    if speech_ms < MIN_SOURCE_SPEECH_MS_WITH_TRANSCRIPT:
        return False
    token_count = sum(max(1, len(word.text.split())) for word in words)
    if token_count < MIN_TOKENS_FOR_DENSITY_GATE:
        return True
    minimum_ms = token_count * MIN_SOURCE_SPEECH_MS_PER_TOKEN
    word_span_ms = max(word.end_ms for word in words) - min(word.start_ms for word in words)
    return speech_ms >= minimum_ms and word_span_ms >= minimum_ms


def words_owned_by(words: list[Word], interval: TimeInterval) -> list[Word]:
    """Assign a word to exactly one region by its midpoint.

    Regions partition the timeline (non-overlap speech vs overlap regions), so
    midpoint ownership keeps a word that straddles a boundary from being emitted
    twice while never dropping it (spec 5.11).
    """
    return [word for word in words if interval.contains((word.start_ms + word.end_ms) // 2)]


def _source_buffer(batch: SeparatedBatch, index: int, crop: AudioBuffer) -> AudioBuffer:
    return AudioBuffer(
        samples=np.ascontiguousarray(batch.sources[index][np.newaxis, :].astype(np.float32)),
        sample_rate=batch.sample_rate,
        start_sample=batch.start_sample,
        channel_layout=("mono",),
        source_clock_hz=crop.source_clock_hz,
    )


def _merge_overlap_regions(
    diarization_regions: list[OverlapRegion],
    osd_regions: list[OverlapRegion],
    *,
    merge_gap_ms: int,
    min_duration_ms: int,
) -> list[OverlapRegion]:
    """Union of diarization and OSD evidence (spec 5.2)."""
    all_regions = [*diarization_regions, *osd_regions]
    merged = merge_intervals([region.interval for region in all_regions], merge_gap_ms=merge_gap_ms)
    out: list[OverlapRegion] = []
    for interval in merged:
        if interval.duration_ms < min_duration_ms:
            continue
        activations = [
            region.osd_activation
            for region in all_regions
            if region.osd_activation is not None and region.interval.intersects(interval)
        ]
        model_version = next(
            (
                region.model_version
                for region in all_regions
                if region.interval.intersects(interval)
            ),
            "",
        )
        out.append(
            OverlapRegion(
                interval=interval,
                osd_activation=max(activations) if activations else None,
                model_version=model_version,
            )
        )
    return out


def _subtract(intervals: list[TimeInterval], holes: list[TimeInterval]) -> list[TimeInterval]:
    """Interval difference in integer milliseconds."""
    remaining = list(merge_intervals(intervals))
    for hole in merge_intervals(holes):
        next_round: list[TimeInterval] = []
        for interval in remaining:
            if not interval.intersects(hole):
                next_round.append(interval)
                continue
            if interval.start_ms < hole.start_ms:
                next_round.append(TimeInterval(interval.start_ms, hole.start_ms))
            if hole.end_ms < interval.end_ms:
                next_round.append(TimeInterval(hole.end_ms, interval.end_ms))
        remaining = next_round
    return remaining


def _split_crop(
    padded: TimeInterval,
    region: TimeInterval,
    *,
    max_crop_ms: int,
    stitch_ms: int,
) -> list[tuple[TimeInterval, TimeInterval]]:
    """Split a long crop into overlapping windows — spec 5.4.

    Returns ``(window, owned)`` pairs: ``window`` is separated (with stitch
    context), ``owned`` is the slice of the original region this window is
    responsible for, so stitched windows never emit a word twice.
    """
    if padded.duration_ms <= max_crop_ms:
        return [(padded, region)]

    step = max(max_crop_ms - stitch_ms, 1)
    windows: list[tuple[TimeInterval, TimeInterval]] = []
    start = padded.start_ms
    owned_start = region.start_ms
    while start < padded.end_ms:
        end = min(start + max_crop_ms, padded.end_ms)
        owned_end = min(owned_start + step, region.end_ms)
        if owned_end > owned_start:
            windows.append((TimeInterval(start, end), TimeInterval(owned_start, owned_end)))
        if end >= padded.end_ms:
            break
        start += step
        owned_start = owned_end
    return windows


__all__ = [
    "OfflinePipeline",
    "OfflineResult",
    "PipelineAdapters",
]
