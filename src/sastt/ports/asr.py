"""Speech recognition port — spec 5.5, 9."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.audio import AudioBuffer
from sastt.domain.transcript import ASRResult
from sastt.observability import CallContext


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Word-level ASR.

    Word timestamps are returned in absolute session time. Whisper word
    probabilities are raw model scores and MUST NOT be published as calibrated
    ASR confidence (spec 5.5).
    """

    @property
    def model_version(self) -> str: ...

    def transcribe(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        language: str | None = None,
        source_track: int | None = None,
    ) -> ASRResult: ...


__all__ = ["SpeechRecognizer"]
