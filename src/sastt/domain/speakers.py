"""Speaker domain types and the identity state machine — spec 5.2–5.10 and 6."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from sastt.domain.audio import TimeInterval
from sastt.domain.errors import InvalidStateTransitionError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float32]
else:  # pragma: no cover - runtime alias
    FloatArray = np.ndarray


# --------------------------------------------------------------------------- #
# Identity state machine (spec 6)
# --------------------------------------------------------------------------- #


class IdentityState(str, Enum):
    """Internal speaker identity state — spec 6 diagram."""

    PROVISIONAL = "PROVISIONAL"
    SESSION_ANONYMOUS = "SESSION_ANONYMOUS"
    ENROLLED = "ENROLLED"
    MERGED = "MERGED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class IdentityStatus(str, Enum):
    """Public ``identity_status`` values — spec 7 invariants."""

    PROVISIONAL = "provisional"
    ENROLLED = "enrolled"
    ANONYMOUS = "anonymous"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


#: Entry points of the state machine (spec 6).
INITIAL_STATES: frozenset[IdentityState] = frozenset(
    {IdentityState.PROVISIONAL, IdentityState.SESSION_ANONYMOUS}
)

#: Allowed transitions, transcribed one-to-one from the spec 6 diagram.
ALLOWED_TRANSITIONS: dict[IdentityState, frozenset[IdentityState]] = {
    IdentityState.PROVISIONAL: frozenset(
        {IdentityState.SESSION_ANONYMOUS, IdentityState.ENROLLED, IdentityState.UNKNOWN}
    ),
    IdentityState.SESSION_ANONYMOUS: frozenset({IdentityState.ENROLLED, IdentityState.MERGED}),
    IdentityState.MERGED: frozenset({IdentityState.SESSION_ANONYMOUS}),
    IdentityState.ENROLLED: frozenset({IdentityState.AMBIGUOUS}),
    IdentityState.AMBIGUOUS: frozenset({IdentityState.ENROLLED, IdentityState.UNKNOWN}),
    IdentityState.UNKNOWN: frozenset(),
}

_PUBLIC_STATUS: dict[IdentityState, IdentityStatus] = {
    IdentityState.PROVISIONAL: IdentityStatus.PROVISIONAL,
    IdentityState.SESSION_ANONYMOUS: IdentityStatus.ANONYMOUS,
    IdentityState.MERGED: IdentityStatus.ANONYMOUS,
    IdentityState.ENROLLED: IdentityStatus.ENROLLED,
    IdentityState.AMBIGUOUS: IdentityStatus.AMBIGUOUS,
    IdentityState.UNKNOWN: IdentityStatus.UNKNOWN,
}


def public_status(state: IdentityState) -> IdentityStatus:
    """Map an internal state onto the public ``identity_status`` (spec 6, 7)."""
    return _PUBLIC_STATUS[state]


@dataclass(frozen=True)
class StateTransition:
    from_state: IdentityState
    to_state: IdentityState
    reason: str
    revision: int


class SpeakerIdentityStateMachine:
    """Guards the transitions of spec 6 and records an auditable history."""

    def __init__(self, initial: IdentityState) -> None:
        if initial not in INITIAL_STATES:
            raise InvalidStateTransitionError(
                f"{initial.value} is not a valid initial state",
                details={"allowed": sorted(s.value for s in INITIAL_STATES)},
            )
        self._state = initial
        self._history: list[StateTransition] = []

    @property
    def state(self) -> IdentityState:
        return self._state

    @property
    def status(self) -> IdentityStatus:
        return public_status(self._state)

    @property
    def history(self) -> tuple[StateTransition, ...]:
        return tuple(self._history)

    @property
    def revision(self) -> int:
        """Revisions start at 1 and increase with every accepted transition."""
        return len(self._history) + 1

    def can_transition(self, to_state: IdentityState) -> bool:
        return to_state in ALLOWED_TRANSITIONS[self._state]

    def transition(self, to_state: IdentityState, reason: str) -> StateTransition:
        if not self.can_transition(to_state):
            raise InvalidStateTransitionError(
                f"illegal transition {self._state.value} -> {to_state.value}",
                details={
                    "from": self._state.value,
                    "to": to_state.value,
                    "allowed": sorted(s.value for s in ALLOWED_TRANSITIONS[self._state]),
                },
            )
        transition = StateTransition(
            from_state=self._state,
            to_state=to_state,
            reason=reason,
            revision=self.revision + 1,
        )
        self._state = to_state
        self._history.append(transition)
        return transition


# --------------------------------------------------------------------------- #
# Diarization / OSD (spec 5.2)
# --------------------------------------------------------------------------- #

TrackKind = Literal["regular", "exclusive"]


@dataclass(frozen=True)
class SpeakerTurn:
    """One diarization turn attributed to a global cluster key."""

    cluster_id: str
    interval: TimeInterval
    kind: TrackKind = "regular"
    score: float | None = None

    @property
    def start_ms(self) -> int:
        return self.interval.start_ms

    @property
    def end_ms(self) -> int:
        return self.interval.end_ms


@dataclass(frozen=True)
class OverlapRegion:
    """A region where more than one speaker is active.

    ``is_overlap`` only means ``K > 1``: segmentation-3.0 models at most two
    active speakers per frame, so an exact ``K = 2`` MUST NOT be inferred
    from it (spec 5.2).
    """

    interval: TimeInterval
    osd_activation: float | None = None
    model_version: str = ""
    calibration_version: str | None = None

    @property
    def start_ms(self) -> int:
        return self.interval.start_ms

    @property
    def end_ms(self) -> int:
        return self.interval.end_ms


@dataclass(frozen=True)
class DiarizationResult:
    """Return type of the ``Diarizer`` port — spec 5.2."""

    turns: list[SpeakerTurn]
    regular_tracks: list[SpeakerTurn]
    exclusive_tracks: list[SpeakerTurn] | None
    overlap_regions: list[OverlapRegion]
    estimated_session_speakers: int
    model_version: str

    def cluster_ids(self) -> list[str]:
        seen: dict[str, int] = {}
        for turn in sorted(self.regular_tracks, key=lambda t: t.start_ms):
            seen.setdefault(turn.cluster_id, turn.start_ms)
        return list(seen)

    def overlap_at(self, interval: TimeInterval) -> OverlapRegion | None:
        for region in self.overlap_regions:
            if region.interval.intersects(interval):
                return region
        return None


# --------------------------------------------------------------------------- #
# Concurrent speaker counting (spec 5.3)
# --------------------------------------------------------------------------- #

CountMethod = Literal[
    "fixed_two",
    "ts_vad",
    "multichannel_activity",
    "multidecoder_research",
    "unknown",
]


@dataclass(frozen=True)
class SourceCountEstimate:
    """Concurrent source count estimate — spec 5.3.

    ``count_uncertain`` marks the V1 fallback ``K=2`` taken without evidence
    (spec 5.3 rule 4), which must be quality-checked after separation.
    """

    count: int | None
    confidence: float | None
    method: CountMethod
    count_uncertain: bool = False


# --------------------------------------------------------------------------- #
# Separation (spec 5.4)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceQuality:
    """Per-source diagnostics; not a calibrated probability (spec 5.4)."""

    speech_duration_ms: int
    energy_ratio: float | None = None
    leakage_similarity: float | None = None
    residual_speech_ratio: float | None = None
    snr_db: float | None = None
    passed_gate: bool = True
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "speech_duration_ms": self.speech_duration_ms,
            "energy_ratio": self.energy_ratio,
            "leakage_similarity": self.leakage_similarity,
            "residual_speech_ratio": self.residual_speech_ratio,
            "snr_db": self.snr_db,
            "passed_gate": self.passed_gate,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SeparatedBatch:
    """Return type of the ``SpeechSeparator`` port — spec 5.4.

    ``sources`` has shape ``[K, samples]``. The source index is local to this
    crop and MUST NOT be used as a cross-chunk identity (spec 5.4).
    """

    sources: FloatArray
    sample_rate: int
    requested_source_count: int
    estimated_source_count: int | None
    source_quality: list[SourceQuality]
    separator_version: str
    start_sample: int = 0

    def __post_init__(self) -> None:
        if self.sources.ndim != 2:
            raise ValueError("sources must have shape [K, samples]")
        if len(self.source_quality) != int(self.sources.shape[0]):
            raise ValueError("source_quality must have one entry per source")

    @property
    def source_count(self) -> int:
        return int(self.sources.shape[0])


# --------------------------------------------------------------------------- #
# Embeddings and prototypes (spec 5.6)
# --------------------------------------------------------------------------- #

EmbeddingOrigin = Literal["clean", "separated", "enrollment"]


@dataclass(frozen=True)
class SpeakerEmbedding:
    """An L2-normalised speaker embedding with lineage — spec 5.6.

    Embeddings from different models/versions MUST NOT be compared.
    """

    vector: FloatArray
    model_version: str
    quality: float
    speech_duration_ms: int
    origin: EmbeddingOrigin
    interval: TimeInterval | None = None
    source_track: int | None = None

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError("embedding vector must be 1-D")
        norm = float(np.linalg.norm(self.vector))
        if norm == 0.0:
            raise ValueError("embedding vector must not be all-zero")
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"embedding vector must be L2-normalised, norm={norm:.6f}")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be within [0,1], got {self.quality}")


def l2_normalize(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalise an all-zero vector")
    return (vector / norm).astype(np.float32)


def cosine_similarity(a: SpeakerEmbedding, b: SpeakerEmbedding) -> float:
    """Cosine similarity between two embeddings of the *same* model version."""
    if a.model_version != b.model_version:
        raise ValueError(
            "refusing to compare embeddings across model versions: "
            f"{a.model_version!r} vs {b.model_version!r}"
        )
    return float(np.dot(a.vector, b.vector))


@dataclass(frozen=True)
class SpeakerPrototype:
    """Quality-weighted centroid of one speaker — spec 5.6.

    ``c_new = normalize(sum(q_i * e_i) / sum(q_i))``. Updates are versioned and
    reversible: every version keeps the accumulator it was built from.
    """

    speaker_key: str
    centroid: FloatArray
    weight_sum: float
    model_version: str
    version: int = 1
    contributing_segments: tuple[TimeInterval, ...] = ()

    @classmethod
    def from_embedding(cls, speaker_key: str, embedding: SpeakerEmbedding) -> SpeakerPrototype:
        return cls(
            speaker_key=speaker_key,
            centroid=l2_normalize(embedding.vector * embedding.quality),
            weight_sum=embedding.quality,
            model_version=embedding.model_version,
            version=1,
            contributing_segments=(embedding.interval,) if embedding.interval else (),
        )

    def updated_with(self, embedding: SpeakerEmbedding) -> SpeakerPrototype:
        if embedding.model_version != self.model_version:
            raise ValueError("cannot update a prototype with another model version")
        accumulated = self.centroid * self.weight_sum + embedding.vector * embedding.quality
        weight_sum = self.weight_sum + embedding.quality
        segments = self.contributing_segments
        if embedding.interval is not None:
            segments = (*segments, embedding.interval)
        return replace(
            self,
            centroid=l2_normalize(accumulated / weight_sum),
            weight_sum=weight_sum,
            version=self.version + 1,
            contributing_segments=segments,
        )

    def similarity(self, embedding: SpeakerEmbedding) -> float:
        if embedding.model_version != self.model_version:
            raise ValueError(
                "refusing to compare an embedding against a prototype of another model version"
            )
        return float(np.dot(self.centroid, embedding.vector))


# --------------------------------------------------------------------------- #
# Linking and Voice ID decisions (spec 5.8, 5.10)
# --------------------------------------------------------------------------- #

LinkStatus = Literal["linked", "unknown", "ambiguous", "uncalibrated"]


@dataclass(frozen=True)
class LinkingDecision:
    """Result of assigning one separated source to a session speaker."""

    source_track: int
    session_speaker_id: str | None
    status: LinkStatus
    score: float | None = None
    margin: float | None = None
    reason: str = ""


VoiceIdStatus = Literal["enrolled", "unknown", "ambiguous", "uncalibrated"]


@dataclass(frozen=True)
class VoiceIdDecision:
    """Open-set Voice ID decision — spec 5.10. Never forced onto a person."""

    status: VoiceIdStatus
    registry_speaker_id: str | None = None
    speaker_name: str | None = None
    best_score: float | None = None
    margin: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class EnrollmentClipReport:
    """Per-clip verdict of an enrollment request — spec 5.10, 8.3."""

    clip_index: int
    accepted: bool
    speech_duration_ms: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnrollmentReport:
    """Enrollment returns a quality report, not just an HTTP success (spec 8.3)."""

    identity_id: str
    accepted_clips: int
    rejected_clips: int
    total_speech_ms: int
    prototype_count: int
    embedding_model_version: str
    meets_policy: bool
    clips: tuple[EnrollmentClipReport, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_id": self.identity_id,
            "accepted_clips": self.accepted_clips,
            "rejected_clips": self.rejected_clips,
            "total_speech_ms": self.total_speech_ms,
            "prototype_count": self.prototype_count,
            "embedding_model_version": self.embedding_model_version,
            "meets_policy": self.meets_policy,
            "reasons": list(self.reasons),
            "clips": [
                {
                    "clip_index": clip.clip_index,
                    "accepted": clip.accepted,
                    "speech_duration_ms": clip.speech_duration_ms,
                    "reasons": list(clip.reasons),
                }
                for clip in self.clips
            ],
        }


@dataclass
class SessionSpeaker:
    """A speaker inside one session — spec 6 identity rules.

    ``session_speaker_id`` is stable for the whole session and is never reused;
    ``registry_speaker_id`` may stay null; the display label may be revised.
    """

    session_speaker_id: str
    machine: SpeakerIdentityStateMachine
    display_label: str
    cluster_id: str | None = None
    registry_speaker_id: str | None = None
    speaker_name: str | None = None
    prototype: SpeakerPrototype | None = None
    prototype_history: list[SpeakerPrototype] = field(default_factory=list)
    merged_into: str | None = None
    cannot_link: set[str] = field(default_factory=set)

    @property
    def state(self) -> IdentityState:
        return self.machine.state

    @property
    def identity_status(self) -> IdentityStatus:
        return self.machine.status

    @property
    def public_speaker_id(self) -> str:
        """Backward-compatible ``speaker_id``: registry ID when enrolled (spec 6)."""
        return self.registry_speaker_id or self.session_speaker_id

    def rollback_prototype(self) -> None:
        """Prototype updates are reversible within the session (spec 5.6)."""
        if self.prototype_history:
            self.prototype = self.prototype_history.pop()


__all__ = [
    "ALLOWED_TRANSITIONS",
    "INITIAL_STATES",
    "CountMethod",
    "DiarizationResult",
    "EmbeddingOrigin",
    "EnrollmentClipReport",
    "EnrollmentReport",
    "IdentityState",
    "IdentityStatus",
    "LinkStatus",
    "LinkingDecision",
    "OverlapRegion",
    "SeparatedBatch",
    "SessionSpeaker",
    "SourceCountEstimate",
    "SourceQuality",
    "SpeakerEmbedding",
    "SpeakerIdentityStateMachine",
    "SpeakerPrototype",
    "SpeakerTurn",
    "StateTransition",
    "TrackKind",
    "VoiceIdDecision",
    "VoiceIdStatus",
    "cosine_similarity",
    "l2_normalize",
    "public_status",
]
