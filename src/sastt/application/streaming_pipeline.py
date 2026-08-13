"""Near-realtime pipeline — spec 4.2, 8.2, FR-011.

Two lanes, as in the spec 4.2 sequence diagram: a fast lane that emits
``transcript.provisional`` as soon as a speech chunk is endpointed, and a slower
speaker lane (rolling diarization/OSD, separation, linking) that issues
``transcript.revision``. Finalization runs the offline second pass over the same
session state, so temporary identities created at the start of a session are
reconciled and published as ``transcript.final`` events.

A client that reconnects replays from its ``last_sequence_number`` and never
receives a duplicated final event (spec 8.2, 15).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sastt.application.fusion import FusionEngine, NullConfidenceCalibrator, WordGroup
from sastt.application.offline_pipeline import (
    OfflinePipeline,
    PipelineAdapters,
    _merge_overlap_regions,
)
from sastt.application.overlap_router import CountingEvidence, estimate_source_count, route_overlap
from sastt.application.session_state import SessionSpeakerState
from sastt.config import SasttConfig
from sastt.domain.audio import (
    CANONICAL_SAMPLE_RATE,
    AudioBuffer,
    FloatArray,
    TimeInterval,
    ms_to_samples,
    samples_to_ms,
    seconds_to_ms,
)
from sastt.domain.errors import SasttError, SessionClockDiscontinuityError
from sastt.domain.events import (
    Clock,
    EventType,
    ServerEvent,
    SessionEventLog,
    SessionState,
    SystemClock,
)
from sastt.domain.transcript import ModelVersions, TranscriptSegment, sort_segments
from sastt.observability import CallContext


@dataclass
class _Emitted:
    """A provisional segment already delivered to the client."""

    event_id: str
    start_ms: int
    end_ms: int
    session_speaker_id: str
    source_track: int | None


@dataclass
class StreamingSession:
    """One realtime session over a WebSocket-style frame stream."""

    session_id: str
    config: SasttConfig
    adapters: PipelineAdapters
    tenant_id: str = "tenant_local"
    clock: Clock = field(default_factory=SystemClock)
    calibrator: NullConfidenceCalibrator = field(default_factory=NullConfidenceCalibrator)
    keep_session_audio: bool = True

    def __post_init__(self) -> None:
        self.state_machine: SessionState = SessionState.CREATED
        self.log = SessionEventLog(self.session_id)
        self.speakers = SessionSpeakerState(
            session_id=self.session_id,
            config=self.config,
            embedding_model_version=self.adapters.embedder.model_version,
        )
        self.pipeline = OfflinePipeline(
            self.config, self.adapters, tenant_id=self.tenant_id, calibrator=self.calibrator
        )
        self.model_versions = ModelVersions(
            diarization=self.adapters.diarizer.model_version,
            embedding=self.adapters.embedder.model_version,
            asr=self.adapters.recognizer.model_version,
            calibration=self.calibrator.calibration_version,
        )
        self._samples = np.zeros(0, dtype=np.float32)
        self._emitted: list[_Emitted] = []
        self._emitted_until_ms = 0
        self._last_process_ms = 0
        self._finalized = False

    # -- configuration derived values --------------------------------------- #

    @property
    def sample_rate(self) -> int:
        return self.config.audio.canonical_sample_rate

    @property
    def hop_ms(self) -> int:
        return seconds_to_ms(self.config.streaming.diarization_hop_seconds)

    @property
    def window_ms(self) -> int:
        return seconds_to_ms(self.config.streaming.diarization_window_seconds)

    @property
    def ring_ms(self) -> int:
        return seconds_to_ms(self.config.streaming.ring_buffer_seconds)

    @property
    def silence_ms(self) -> int:
        return seconds_to_ms(self.config.streaming.finalize_after_silence_seconds)

    @property
    def now_ms(self) -> int:
        return samples_to_ms(self._samples.size, self.sample_rate)

    # -- lifecycle ----------------------------------------------------------- #

    def start(self) -> ServerEvent:
        if self.state_machine is not SessionState.CREATED:
            raise SasttError("session already started")
        self.state_machine = SessionState.STREAMING
        return self.log.append(
            event_type=EventType.SESSION_STARTED,
            clock=self.clock,
            payload={
                "session_id": self.session_id,
                "sample_rate": self.sample_rate,
                "frame_ms": self.config.streaming.frame_ms,
                "max_session_speakers": self.config.product.max_session_speakers,
            },
            model_versions=self.model_versions.to_dict(),
            config_version=self.config.config_version,
        )

    def push_pcm(self, payload: bytes) -> list[ServerEvent]:
        """Ingest PCM s16le mono frames (spec 1.1 stream V1)."""
        if self.state_machine is not SessionState.STREAMING:
            raise SessionClockDiscontinuityError(
                "audio pushed outside the streaming state",
                details={"state": self.state_machine.value},
            )
        if len(payload) % 2 != 0:
            raise SasttError("PCM payload is not 16-bit aligned")
        frame = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        self._samples = np.concatenate([self._samples, frame])
        return self._maybe_process()

    def push_audio(self, buffer: AudioBuffer) -> list[ServerEvent]:
        if buffer.sample_rate != self.sample_rate:
            raise SasttError("streaming input must already be at the canonical sample rate")
        return self.push_pcm(_to_pcm16(buffer.to_mono().samples[0]))

    def finalize(self) -> list[ServerEvent]:
        """Close input, run the final pass, emit finals (spec 8.2, 5.9 step 6)."""
        if self._finalized:
            return []
        self.state_machine = SessionState.FINALIZING
        events = self._process(force_close=True)
        events.extend(self._final_pass())
        self.state_machine = SessionState.FINALIZED
        self._finalized = True
        events.append(
            self.log.append(
                event_type=EventType.SESSION_FINALIZED,
                clock=self.clock,
                payload={"segments": len(self.log.final_events())},
                model_versions=self.model_versions.to_dict(),
                config_version=self.config.config_version,
            )
        )
        return events

    def fail(self, error_code: str, message: str) -> ServerEvent:
        self.state_machine = SessionState.FAILED
        return self.log.append(
            event_type=EventType.SESSION_FAILED,
            clock=self.clock,
            payload={"error_code": error_code, "message": message},
            config_version=self.config.config_version,
        )

    def replay(self, last_sequence_number: int) -> list[ServerEvent]:
        """Reconnect replay — spec 8.2."""
        return self.log.replay_from(last_sequence_number)

    def result(self) -> list[TranscriptSegment]:
        """Canonical transcript of the session (``GET /v1/sessions/{id}/result``)."""
        return sort_segments(
            [
                _segment_from_payload(event.payload)
                for event in self.log.final_events()
                if event.event_type is EventType.TRANSCRIPT_FINAL
            ]
        )

    # -- processing ---------------------------------------------------------- #

    def _maybe_process(self) -> list[ServerEvent]:
        if self.now_ms - self._last_process_ms < self.hop_ms:
            return []
        return self._process()

    def _window_buffer(self) -> AudioBuffer | None:
        if self._samples.size == 0:
            return None
        window_start_ms = max(0, self.now_ms - max(self.window_ms, self.ring_ms))
        start = ms_to_samples(window_start_ms, self.sample_rate)
        chunk = self._samples[start:]
        if chunk.size == 0:
            return None
        return AudioBuffer(
            samples=np.ascontiguousarray(chunk[np.newaxis, :]),
            sample_rate=self.sample_rate,
            start_sample=start,
            channel_layout=("mono",),
            source_clock_hz=self.sample_rate,
        )

    def _process(self, *, force_close: bool = False) -> list[ServerEvent]:
        """One rolling step: fast ASR lane plus the speaker lane (spec 4.2)."""
        self._last_process_ms = self.now_ms
        buffer = self._window_buffer()
        if buffer is None:
            return []
        ctx = CallContext(stage="streaming", session_id=self.session_id)

        diarization = self.adapters.diarizer.diarize(
            buffer,
            ctx.child("diarization"),
            min_speakers=self.config.diarization.min_speakers,
            max_speakers=self.config.diarization.max_speakers,
        )
        osd = self.adapters.overlap_detector.detect(buffer, ctx.child("osd"))
        overlaps = _merge_overlap_regions(
            diarization.overlap_regions,
            osd,
            merge_gap_ms=seconds_to_ms(self.config.overlap_detection.merge_gap_seconds),
            min_duration_ms=seconds_to_ms(self.config.overlap_detection.min_duration_seconds),
        )
        self.pipeline.build_global_speakers(self.speakers, diarization, overlaps, buffer, ctx)

        speech = self.adapters.vad.detect(buffer, ctx.child("vad"))
        closed = [
            interval
            for interval in speech
            if interval.end_ms > self._emitted_until_ms
            and (force_close or self.now_ms - interval.end_ms >= self.silence_ms)
        ]
        if not closed:
            return self._drain_label_revisions()

        groups: list[WordGroup] = []
        for interval in closed:
            pending = TimeInterval(max(interval.start_ms, self._emitted_until_ms), interval.end_ms)
            groups.extend(self.pipeline.transcribe_non_overlap(buffer, [pending], overlaps, ctx))
            for region in overlaps:
                clipped = region.interval.clamp(pending)
                if clipped is None:
                    continue
                if not force_close and self.now_ms - clipped.end_ms < self.silence_ms:
                    continue  # region not closed yet
                count = estimate_source_count(CountingEvidence(), self.config)
                decision = route_overlap(region, count, self.config, osd_positive=True)
                region_groups, _, _ = self.pipeline.handle_overlap_region(
                    buffer, region, decision, self.speakers, ctx
                )
                groups.extend(region_groups)

        events = self._emit_provisional(groups, diarization.regular_tracks)
        self._emitted_until_ms = max(
            [self._emitted_until_ms, *[interval.end_ms for interval in closed]]
        )
        events.extend(self._drain_label_revisions())
        return events

    def _emit_provisional(
        self,
        groups: list[WordGroup],
        turns: list,  # type: ignore[type-arg]
    ) -> list[ServerEvent]:
        if not groups:
            return []
        fusion = FusionEngine(
            session_id=self.session_id,
            state=self.speakers,
            config=self.config,
            model_versions=self.model_versions,
            calibrator=self.calibrator,
        )
        events: list[ServerEvent] = []
        for segment in sort_segments(fusion.fuse(groups, turns, is_final=False)):
            event = self.log.append(
                event_type=EventType.TRANSCRIPT_PROVISIONAL,
                clock=self.clock,
                payload=segment.to_public_dict(),
                model_versions=self.model_versions.to_dict(),
                config_version=self.config.config_version,
            )
            self._emitted.append(
                _Emitted(
                    event_id=event.event_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    session_speaker_id=segment.session_speaker_id,
                    source_track=segment.source_track,
                )
            )
            events.append(event)
        return events

    def _drain_label_revisions(self) -> list[ServerEvent]:
        """Publish label changes as revisions — never rewrite a delivered event."""
        events: list[ServerEvent] = []
        for change in self.speakers.drain_label_changes():
            events.append(
                self.log.append(
                    event_type=EventType.TRANSCRIPT_REVISION,
                    clock=self.clock,
                    revision=2,
                    payload={
                        "session_speaker_id": change.session_speaker_id,
                        "previous_label": change.previous_label,
                        "speaker_label": change.new_label,
                        "reason": change.reason,
                    },
                    model_versions=self.model_versions.to_dict(),
                    config_version=self.config.config_version,
                )
            )
        return events

    def _final_pass(self) -> list[ServerEvent]:
        """Offline second pass over the whole session (spec 5.9 step 6, 4.1)."""
        events: list[ServerEvent] = []
        if self._samples.size == 0:
            return events
        payload = _to_pcm16(self._samples)
        ctx = CallContext(stage="finalization", session_id=self.session_id)
        result = self.pipeline.run(
            payload,
            ctx,
            session_id=self.session_id,
            state=self.speakers,
        )
        # Reconciliation may have merged temporary identities: publish the label
        # changes as revisions before the finals (spec 5.9 step 5, FR-011).
        events.extend(self._drain_label_revisions())

        superseded_ids: set[str] = set()
        for segment in result.segments:
            superseded = self._superseded_event_id(segment, superseded_ids)
            if superseded:
                superseded_ids.add(superseded)
            events.append(
                self.log.append(
                    event_type=EventType.TRANSCRIPT_FINAL,
                    clock=self.clock,
                    revision=2 if superseded else 1,
                    supersedes_event_id=superseded,
                    is_final=True,
                    payload=segment.to_public_dict(),
                    model_versions=result.model_versions.to_dict(),
                    config_version=result.config_version,
                    dedup_key=_final_dedup_key(segment),
                )
            )
        if result.warnings:
            events.append(
                self.log.append(
                    event_type=EventType.PIPELINE_WARNING,
                    clock=self.clock,
                    payload={"warnings": result.warnings, "degraded_mode": result.degraded},
                    config_version=result.config_version,
                )
            )
        return events

    def _superseded_event_id(
        self, segment: TranscriptSegment, already_used: set[str]
    ) -> str | None:
        """Which provisional event this final replaces (spec FR-011).

        Each provisional is superseded at most once, and a provisional carrying
        the same source track is preferred, so two concurrent speakers do not
        both claim the same earlier event.
        """
        interval = TimeInterval(segment.start_ms, segment.end_ms)
        best: _Emitted | None = None
        best_key = (0, 0)
        for emitted in self._emitted:
            if emitted.event_id in already_used:
                continue
            overlap = TimeInterval(emitted.start_ms, emitted.end_ms).intersection_ms(interval)
            if overlap <= 0:
                continue
            same_track = int(emitted.source_track == segment.source_track)
            key = (same_track, overlap)
            if key > best_key:
                best_key = key
                best = emitted
        return best.event_id if best else None


def _final_dedup_key(segment: TranscriptSegment) -> str:
    return (
        f"{segment.start_ms}:{segment.end_ms}:{segment.session_speaker_id}:{segment.source_track}"
    )


def _to_pcm16(samples: FloatArray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _segment_from_payload(payload: dict[str, Any]) -> TranscriptSegment:
    """Rebuild a segment from an event payload (canonical result assembly)."""
    from sastt.domain.speakers import IdentityStatus
    from sastt.domain.transcript import Confidences

    return TranscriptSegment(
        session_id=str(payload["session_id"]),
        event_id=str(payload["event_id"]),
        start_ms=int(payload["start_ms"]),
        end_ms=int(payload["end_ms"]),
        text=str(payload["text"]),
        speaker_id=str(payload["speaker_id"]),
        session_speaker_id=str(payload["session_speaker_id"]),
        identity_status=IdentityStatus(str(payload["identity_status"])),
        revision=int(payload["revision"]),
        supersedes_event_id=payload.get("supersedes_event_id"),
        registry_speaker_id=payload.get("registry_speaker_id"),
        speaker_label=str(payload.get("speaker_label") or ""),
        speaker_name=payload.get("speaker_name"),
        is_overlap=bool(payload.get("is_overlap")),
        estimated_concurrent_speakers=payload.get("estimated_concurrent_speakers"),
        count_confidence=payload.get("count_confidence"),
        source_track=payload.get("source_track"),
        separation_backend=payload.get("separation_backend"),
        confidences=Confidences(status="uncalibrated"),
        raw_scores=dict(payload.get("raw_scores") or {}),
        quality_flags=tuple(payload.get("quality_flags") or ()),
        degraded_mode=bool(payload.get("degraded_mode")),
        is_final=bool(payload.get("is_final")),
    )


CANONICAL_STREAM_SAMPLE_RATE = CANONICAL_SAMPLE_RATE

__all__ = [
    "CANONICAL_STREAM_SAMPLE_RATE",
    "StreamingSession",
]
