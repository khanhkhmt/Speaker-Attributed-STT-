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

    def detect_language(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
    ) -> tuple[str, float]:
        """Identify the language of ``buffer`` without transcribing it.

        Separate from :meth:`transcribe` because identification is a session
        decision made once on pooled speech, while transcription runs per crop.
        Returns the ISO code and the backend's own probability for it; callers
        decide whether that probability is high enough to act on (spec 5.5).
        """
        ...


__all__ = ["SpeechRecognizer"]
