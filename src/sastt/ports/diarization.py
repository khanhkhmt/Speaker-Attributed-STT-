"""VAD, diarization, overlap detection and concurrent counting ports — spec 5.2, 5.3, 9."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.audio import AudioBuffer, TimeInterval
from sastt.domain.speakers import DiarizationResult, OverlapRegion, SourceCountEstimate
from sastt.observability import CallContext


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Speech activity over a buffer, in absolute session time."""

    def detect(self, buffer: AudioBuffer, ctx: CallContext) -> list[TimeInterval]: ...


@runtime_checkable
class Diarizer(Protocol):
    """Global diarization — spec 5.2.

    ``regular_tracks`` is the source of truth and preserves overlap;
    ``exclusive_tracks`` may only be used to align words in non-overlap regions
    and MUST NOT delete a second track inside overlap.
    """

    @property
    def model_version(self) -> str: ...

    def diarize(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ) -> DiarizationResult: ...


@runtime_checkable
class OverlapDetector(Protocol):
    """Overlapped speech detection — spec 5.2.

    A positive region only means ``K > 1``; it MUST NOT be read as ``K == 2``.
    """

    @property
    def model_version(self) -> str: ...

    def detect(self, buffer: AudioBuffer, ctx: CallContext) -> list[OverlapRegion]: ...


@runtime_checkable
class ConcurrentSpeakerCounter(Protocol):
    """Concurrent source count estimation — spec 5.3 evidence order."""

    def estimate(
        self,
        buffer: AudioBuffer,
        region: OverlapRegion,
        ctx: CallContext,
        *,
        active_speaker_hint: int | None = None,
    ) -> SourceCountEstimate: ...


__all__ = [
    "ConcurrentSpeakerCounter",
    "Diarizer",
    "OverlapDetector",
    "VoiceActivityDetector",
]
