"""Transcript types and the public output contract v2 — spec 5.5, 5.11, 7."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from sastt.domain.audio import TimeInterval, ms_to_samples
from sastt.domain.errors import SchemaInvariantError
from sastt.domain.speakers import IdentityStatus

SCHEMA_VERSION = "2.0"

#: Rounding used by the stable word ID of spec 5.5 (dedup across revisions).
WORD_ID_SAMPLE_ROUNDING = 160  # 10 ms at 16 kHz

ConfidenceStatus = Literal["uncalibrated", "calibrated"]


def stable_word_id(
    *,
    session_id: str,
    start_sample: int,
    text: str,
    source_track: int | None,
    sample_rounding: int = WORD_ID_SAMPLE_ROUNDING,
) -> str:
    """Stable word ID from ``(session_id, rounded_start_sample, text_hash, source_track)``.

    Realtime dedup uses this ID plus an alignment tolerance — never a bare
    string comparison (spec 5.5).
    """
    rounded = (start_sample // sample_rounding) * sample_rounding
    text_hash = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]
    payload = (
        f"{session_id}|{rounded}|{text_hash}|{source_track if source_track is not None else '-'}"
    )
    return "wrd_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class Word:
    """One recognised word in absolute session time."""

    text: str
    start_ms: int
    end_ms: int
    raw_probability: float | None = None
    source_track: int | None = None
    word_id: str | None = None

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise SchemaInvariantError(
                f"word interval invalid: {self.start_ms}..{self.end_ms} ({self.text!r})"
            )

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.start_ms, self.end_ms)

    def shifted(self, offset_ms: int) -> Word:
        """Move a model-local word onto the absolute session timeline (spec 5.11.1)."""
        return Word(
            text=self.text,
            start_ms=self.start_ms + offset_ms,
            end_ms=self.end_ms + offset_ms,
            raw_probability=self.raw_probability,
            source_track=self.source_track,
            word_id=self.word_id,
        )

    def with_stable_id(self, session_id: str, sample_rate: int = 16_000) -> Word:
        return Word(
            text=self.text,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            raw_probability=self.raw_probability,
            source_track=self.source_track,
            word_id=stable_word_id(
                session_id=session_id,
                start_sample=ms_to_samples(self.start_ms, sample_rate),
                text=self.text,
                source_track=self.source_track,
            ),
        )


@dataclass(frozen=True)
class ASRResult:
    """Return type of the ``SpeechRecognizer`` port — spec 5.5.

    ``raw_scores`` holds raw model scores (e.g. Whisper word probability).
    They are NOT calibrated ASR confidence.
    """

    words: list[Word]
    detected_language: str
    language_score: float | None
    model_version: str
    raw_scores: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()


@dataclass(frozen=True)
class ModelVersions:
    """Model/config provenance attached to every output (spec FR-013, 7)."""

    diarization: str | None = None
    embedding: str | None = None
    separation: str | None = None
    asr: str | None = None
    calibration: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "diarization": self.diarization,
            "embedding": self.embedding,
            "separation": self.separation,
            "asr": self.asr,
            "calibration": self.calibration,
        }


@dataclass(frozen=True)
class Confidences:
    """Component confidences — spec 0.3 and 7.

    Every value stays ``None`` until a calibrator version exists; the pipeline
    MUST NOT emit probability-looking numbers before calibration.
    """

    asr: float | None = None
    diarization: float | None = None
    linking: float | None = None
    voice_id: float | None = None
    overlap: float | None = None
    overall: float | None = None
    status: ConfidenceStatus = "uncalibrated"

    def __post_init__(self) -> None:
        for name in ("asr", "diarization", "linking", "voice_id", "overlap", "overall"):
            value = getattr(self, name)
            if value is None:
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise SchemaInvariantError(f"{name}_confidence must be within [0,1], got {value}")
        if self.status == "uncalibrated" and self.overall is not None:
            raise SchemaInvariantError("overall_confidence requires a calibrator version")


@dataclass(frozen=True)
class TranscriptSegment:
    """Public output contract v2 — spec 7.

    Two segments may carry overlapping timestamps: the second speaker MUST NOT
    be dropped to make the timeline exclusive (spec 0.1.7).
    """

    session_id: str
    event_id: str
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str
    session_speaker_id: str
    identity_status: IdentityStatus
    revision: int = 1
    supersedes_event_id: str | None = None
    registry_speaker_id: str | None = None
    speaker_label: str = ""
    speaker_name: str | None = None
    is_overlap: bool = False
    estimated_concurrent_speakers: int | None = None
    count_confidence: float | None = None
    source_track: int | None = None
    separation_backend: str | None = None
    confidences: Confidences = field(default_factory=Confidences)
    raw_scores: dict[str, float] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()
    degraded_mode: bool = False
    is_final: bool = False
    model_versions: ModelVersions = field(default_factory=ModelVersions)
    words: tuple[Word, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    # -- invariants of spec 7 ------------------------------------------------ #
    def _validate(self) -> None:
        if not 0 <= self.start_ms < self.end_ms:
            raise SchemaInvariantError(
                f"invariant 0 <= start_ms < end_ms violated: {self.start_ms}..{self.end_ms}"
            )
        if self.revision < 1:
            raise SchemaInvariantError(f"revision must be >= 1, got {self.revision}")
        if self.separation_backend is not None and self.source_track is None:
            raise SchemaInvariantError("source_track is required once separation has run")
        if self.count_confidence is not None and not 0.0 <= self.count_confidence <= 1.0:
            raise SchemaInvariantError("count_confidence must be within [0,1]")
        if not isinstance(self.identity_status, IdentityStatus):
            raise SchemaInvariantError(f"unknown identity_status {self.identity_status!r}")
        if self.identity_status is IdentityStatus.ENROLLED and (
            not self.registry_speaker_id or not self.speaker_name
        ):
            raise SchemaInvariantError(
                "registry_speaker_id and speaker_name are required when enrolled"
            )
        if (
            self.identity_status is not IdentityStatus.ENROLLED
            and self.speaker_id != self.session_speaker_id
        ):
            raise SchemaInvariantError(
                "speaker_id must fall back to session_speaker_id when not enrolled"
            )

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.start_ms, self.end_ms)

    @property
    def sort_key(self) -> tuple[int, str, int]:
        """Final ordering key of spec 7: ``(start_ms, session_speaker_id, source_track)``."""
        return (self.start_ms, self.session_speaker_id, self.source_track or 0)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialise exactly the field set of the spec 7 example."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "revision": self.revision,
            "supersedes_event_id": self.supersedes_event_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "session_speaker_id": self.session_speaker_id,
            "registry_speaker_id": self.registry_speaker_id,
            "speaker_label": self.speaker_label,
            "speaker_name": self.speaker_name,
            "identity_status": self.identity_status.value,
            "is_overlap": self.is_overlap,
            "estimated_concurrent_speakers": self.estimated_concurrent_speakers,
            "count_confidence": self.count_confidence,
            "source_track": self.source_track,
            "separation_backend": self.separation_backend,
            "asr_confidence": self.confidences.asr,
            "diarization_confidence": self.confidences.diarization,
            "linking_confidence": self.confidences.linking,
            "voice_id_confidence": self.confidences.voice_id,
            "overlap_confidence": self.confidences.overlap,
            "overall_confidence": self.confidences.overall,
            "confidence_status": self.confidences.status,
            "raw_scores": dict(self.raw_scores),
            "quality_flags": list(self.quality_flags),
            "degraded_mode": self.degraded_mode,
            "is_final": self.is_final,
            "model_versions": self.model_versions.to_dict(),
        }


def sort_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Order by ``(start_ms, session_speaker_id, source_track)`` keeping concurrency (spec 7)."""
    return sorted(segments, key=lambda segment: segment.sort_key)


def _format_timestamp(ms: int) -> str:
    minutes, remainder = divmod(ms, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def render_segment(segment: TranscriptSegment) -> str:
    """Render one line of the spec 7 text format."""
    if segment.identity_status is IdentityStatus.ENROLLED:
        who = f"{segment.speaker_name} [{segment.registry_speaker_id}]"
    else:
        who = segment.speaker_label or segment.session_speaker_id
    start = _format_timestamp(segment.start_ms)
    end = _format_timestamp(segment.end_ms)
    return f"{start}–{end} — {who}: “{segment.text}”"


def render_transcript(segments: list[TranscriptSegment]) -> str:
    return "\n".join(render_segment(segment) for segment in sort_segments(segments))


__all__ = [
    "SCHEMA_VERSION",
    "ASRResult",
    "ConfidenceStatus",
    "Confidences",
    "ModelVersions",
    "TranscriptSegment",
    "Word",
    "render_segment",
    "render_transcript",
    "sort_segments",
    "stable_word_id",
]
