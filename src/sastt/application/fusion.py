"""Word/turn fusion into the public contract — spec 5.11, 7.

Fusion never collapses concurrency: two speakers active at the same time produce
two segments with overlapping timestamps (spec 0.1.7). Confidence values stay
``None`` until a calibrator release exists (spec 0.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sastt.application.session_state import SessionSpeakerState
from sastt.config import SasttConfig
from sastt.domain.audio import TimeInterval, seconds_to_ms
from sastt.domain.errors import SchemaInvariantError
from sastt.domain.events import new_id
from sastt.domain.speakers import IdentityStatus, SpeakerTurn
from sastt.domain.transcript import (
    Confidences,
    ModelVersions,
    TranscriptSegment,
    Word,
)

SENTENCE_FINAL = (".", "?", "!", "…")


class NullConfidenceCalibrator:
    """``ConfidenceCalibrator`` used while no calibration release exists.

    Returns every component confidence as ``None`` with
    ``confidence_status="uncalibrated"`` — the pipeline must not invent
    probability-looking numbers (spec 0.3, 7).
    """

    @property
    def calibration_version(self) -> str | None:
        return None

    def calibrate(self, raw_scores: dict[str, float]) -> Confidences:
        return Confidences(status="uncalibrated")


@dataclass(frozen=True)
class WordGroup:
    """Words from one ASR call plus their provenance.

    ``session_speaker_id`` is set for overlap groups by source linking; for
    non-overlap groups it may be ``None`` and is resolved against the
    diarization turns (spec 5.11.2).
    """

    words: tuple[Word, ...]
    interval: TimeInterval
    is_overlap: bool = False
    source_track: int | None = None
    separation_backend: str | None = None
    session_speaker_id: str | None = None
    estimated_concurrent_speakers: int | None = None
    count_confidence: float | None = None
    raw_scores: dict[str, float] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()
    degraded_mode: bool = False


@dataclass
class _Utterance:
    speaker_id: str
    words: list[Word]
    group: WordGroup

    @property
    def start_ms(self) -> int:
        return min(word.start_ms for word in self.words)

    @property
    def end_ms(self) -> int:
        return max(word.end_ms for word in self.words)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()


class FusionEngine:
    """Turns word groups into transcript segments (spec 5.11)."""

    def __init__(
        self,
        session_id: str,
        state: SessionSpeakerState,
        config: SasttConfig,
        model_versions: ModelVersions,
        *,
        calibrator: NullConfidenceCalibrator | None = None,
        max_pause_ms: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.state = state
        self.config = config
        self.model_versions = model_versions
        self.calibrator = calibrator or NullConfidenceCalibrator()
        # Utterance boundary reuses the configured endpoint silence (spec 4.2, 5.11.4).
        self.max_pause_ms = (
            max_pause_ms
            if max_pause_ms is not None
            else seconds_to_ms(config.streaming.finalize_after_silence_seconds)
        )

    # -- attribution --------------------------------------------------------- #

    def attribute_word(
        self,
        word: Word,
        turns: list[SpeakerTurn],
        *,
        previous_speaker_id: str | None = None,
    ) -> str | None:
        """Non-overlap attribution — spec 5.11.2.

        The turn with the largest temporal intersection wins; a small continuity
        prior only breaks exact ties.
        """
        best_cluster: str | None = None
        best_overlap = 0
        for turn in turns:
            overlap = turn.interval.intersection_ms(word.interval)
            if overlap <= 0:
                continue
            speaker = self.state.by_cluster(turn.cluster_id)
            candidate_id = speaker.session_speaker_id if speaker else None
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = turn.cluster_id
            elif (
                overlap == best_overlap
                and previous_speaker_id is not None
                and candidate_id == previous_speaker_id
            ):
                best_cluster = turn.cluster_id
        if best_cluster is None:
            return None
        return self.state.ensure_cluster_speaker(best_cluster).session_speaker_id

    # -- coalescing ---------------------------------------------------------- #

    def _coalesce(self, groups: list[WordGroup], turns: list[SpeakerTurn]) -> list[_Utterance]:
        utterances: list[_Utterance] = []
        for group in groups:
            previous_speaker: str | None = None
            current: _Utterance | None = None
            for word in sorted(group.words, key=lambda w: (w.start_ms, w.end_ms)):
                if group.is_overlap or group.session_speaker_id is not None:
                    speaker_id = group.session_speaker_id
                else:
                    speaker_id = self.attribute_word(
                        word, turns, previous_speaker_id=previous_speaker
                    )
                if speaker_id is None:
                    # No turn intersects this word: keep the text, drop nothing,
                    # and let the caller see it as unattributed (spec FR-012).
                    speaker_id = self._fallback_speaker_id(word)
                canonical = self.state.get(speaker_id).session_speaker_id

                if (
                    current is not None
                    and current.speaker_id == canonical
                    and word.start_ms - current.end_ms <= self.max_pause_ms
                    and not current.words[-1].text.endswith(SENTENCE_FINAL)
                ):
                    current.words.append(word)
                else:
                    current = _Utterance(speaker_id=canonical, words=[word], group=group)
                    utterances.append(current)
                previous_speaker = canonical
        return utterances

    def _fallback_speaker_id(self, word: Word) -> str:
        """Attach unattributed words to a provisional speaker instead of dropping them."""
        provisional = self.state.provisional_speakers()
        if provisional:
            return provisional[0].session_speaker_id
        return self.state.create_temporary_speaker().session_speaker_id

    # -- segments ------------------------------------------------------------ #

    def fuse(
        self,
        groups: list[WordGroup],
        turns: list[SpeakerTurn],
        *,
        is_final: bool = False,
        revision: int = 1,
        supersedes_event_id: str | None = None,
    ) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for utterance in self._coalesce(groups, turns):
            speaker = self.state.get(utterance.speaker_id)
            group = utterance.group
            raw_scores = dict(group.raw_scores)
            probabilities = [
                w.raw_probability for w in utterance.words if w.raw_probability is not None
            ]
            if probabilities:
                raw_scores.setdefault(
                    "asr_word_probability", sum(probabilities) / len(probabilities)
                )
            confidences = self.calibrator.calibrate(raw_scores)
            status = speaker.identity_status
            model_versions = self.model_versions
            if group.separation_backend and not model_versions.separation:
                model_versions = ModelVersions(
                    diarization=model_versions.diarization,
                    embedding=model_versions.embedding,
                    separation=group.separation_backend,
                    asr=model_versions.asr,
                    calibration=model_versions.calibration,
                )

            segment = TranscriptSegment(
                session_id=self.session_id,
                event_id=new_id("evt"),
                start_ms=utterance.start_ms,
                end_ms=utterance.end_ms,
                text=utterance.text,
                speaker_id=speaker.public_speaker_id,
                session_speaker_id=speaker.session_speaker_id,
                identity_status=status,
                revision=revision,
                supersedes_event_id=supersedes_event_id,
                registry_speaker_id=speaker.registry_speaker_id,
                speaker_label=speaker.display_label,
                speaker_name=speaker.speaker_name,
                is_overlap=group.is_overlap,
                estimated_concurrent_speakers=group.estimated_concurrent_speakers,
                count_confidence=group.count_confidence,
                source_track=group.source_track,
                separation_backend=group.separation_backend,
                confidences=confidences,
                raw_scores=raw_scores,
                quality_flags=group.quality_flags,
                degraded_mode=group.degraded_mode,
                is_final=is_final,
                model_versions=model_versions,
                words=tuple(utterance.words),
            )
            segments.append(segment)

        _assert_concurrency_preserved(groups, segments)
        return segments


def _assert_concurrency_preserved(
    groups: list[WordGroup], segments: list[TranscriptSegment]
) -> None:
    """Guard against silently losing a concurrent speaker (spec 0.1.7, 3)."""
    spoken = sum(len(group.words) for group in groups)
    emitted = sum(len(segment.words) for segment in segments)
    if emitted != spoken:
        raise SchemaInvariantError(
            f"fusion lost words: {spoken} in, {emitted} out",
            details={"words_in": spoken, "words_out": emitted},
        )


def dedup_words(existing: list[Word], incoming: list[Word], tolerance_ms: int = 200) -> list[Word]:
    """Realtime dedup by stable word ID plus an alignment tolerance (spec 5.5).

    Words are compared by ID first; identical text within ``tolerance_ms`` of an
    existing word is treated as the same word, not a new one.
    """
    known_ids = {word.word_id for word in existing if word.word_id}
    result: list[Word] = []
    for word in incoming:
        if word.word_id and word.word_id in known_ids:
            continue
        duplicate = any(
            other.text == word.text and abs(other.start_ms - word.start_ms) <= tolerance_ms
            for other in existing
        )
        if duplicate:
            continue
        result.append(word)
    return result


def identity_status_of(state: SessionSpeakerState, session_speaker_id: str) -> IdentityStatus:
    return state.get(session_speaker_id).identity_status


__all__ = [
    "FusionEngine",
    "NullConfidenceCalibrator",
    "WordGroup",
    "dedup_words",
    "identity_status_of",
]
