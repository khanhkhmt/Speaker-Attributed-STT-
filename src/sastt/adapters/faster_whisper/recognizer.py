"""faster-whisper ASR and Silero VAD adapters — spec 5.5, 9.

``large-v3-turbo`` with Whisper language auto-detection and word timestamps is the
default (spec 0.2, 5.5). Whisper's word probability is a **raw model score** and
is published as such in ``raw_scores``; it never becomes a calibrated ASR
confidence (spec 5.5, 0.3).

Word timestamps are model-local, so every word is shifted onto absolute session
time using the buffer's ``start_ms`` (spec 5.11.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sastt.domain.audio import AudioBuffer, TimeInterval
from sastt.domain.errors import ModelNotReadyError, SasttError
from sastt.domain.transcript import ASRResult, Word
from sastt.observability import CallContext

DEFAULT_COMPUTE_TYPE = "int8_float16"


class FasterWhisperRecognizer:
    """``SpeechRecognizer`` port backed by CTranslate2 faster-whisper."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_version: str,
        device: str = "auto",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str | None = None,
        beam_size: int = 5,
        min_word_ms: int = 20,
    ) -> None:
        self.model_path = Path(model_path)
        self._model_version = model_version
        self.language = language
        self.beam_size = beam_size
        self.min_word_ms = min_word_ms
        if not self.model_path.exists():
            raise ModelNotReadyError(
                f"ASR weights are not staged at {self.model_path} "
                "(spec 11.2 pre-stages them at build time)",
                details={"model_path": str(self.model_path)},
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ModelNotReadyError("faster-whisper is not installed") from exc

        try:
            self._model = WhisperModel(
                str(self.model_path), device=device, compute_type=compute_type
            )
        except Exception as exc:  # noqa: BLE001 - map any backend failure to a domain error
            raise ModelNotReadyError(
                f"could not load the ASR model: {exc}", details={"model_path": str(self.model_path)}
            ) from exc

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect_language(self, buffer: AudioBuffer, ctx: CallContext) -> tuple[str, float]:
        """Identify the language from up to ~90 s of the buffer — spec 5.5.

        Runs the encoder and the language head only, so it costs a fraction of a
        decode. ``language_detection_segments`` counts 30 s windows; three of
        them vote, which keeps one unusual window from deciding the session.
        """
        ctx.check()
        mono = buffer.to_mono().samples[0]
        try:
            language, probability, _all_probabilities = self._model.detect_language(
                mono, language_detection_segments=3
            )
        except Exception as exc:  # noqa: BLE001 - never leak a backend exception
            raise SasttError(f"language identification failed: {exc}") from exc
        return str(language), float(probability)

    def transcribe(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        language: str | None = None,
        source_track: int | None = None,
    ) -> ASRResult:
        ctx.check()
        mono = buffer.to_mono().samples[0]
        offset_ms = buffer.start_ms
        try:
            segments, info = self._model.transcribe(
                mono,
                language=language or self.language,
                word_timestamps=True,
                beam_size=self.beam_size,
                # Upstream VAD has already cut the speech; a second pass is only
                # justified once a benchmark shows it helps (spec 5.5).
                vad_filter=False,
            )
            words: list[Word] = []
            probabilities: list[float] = []
            for segment in segments:
                for word in segment.words or []:
                    start = offset_ms + int(round(word.start * 1000))
                    end = offset_ms + int(round(word.end * 1000))
                    if end - start < self.min_word_ms:
                        end = start + self.min_word_ms
                    text = word.word.strip()
                    if not text:
                        continue
                    probability = float(getattr(word, "probability", 0.0) or 0.0)
                    probabilities.append(probability)
                    words.append(
                        Word(
                            text=text,
                            start_ms=max(0, start),
                            end_ms=max(1, end),
                            raw_probability=probability,
                            source_track=source_track,
                        )
                    )
        except SasttError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a backend exception
            raise SasttError(f"ASR failed: {exc}") from exc

        raw_scores: dict[str, float] = {}
        if probabilities:
            # Raw Whisper score, deliberately not called a confidence (spec 5.5).
            raw_scores["asr_word_probability"] = sum(probabilities) / len(probabilities)
        return ASRResult(
            words=words,
            detected_language=str(getattr(info, "language", language or self.language)),
            language_score=None,
            model_version=self._model_version,
            raw_scores=raw_scores,
        )


class SileroVoiceActivityDetector:
    """``VoiceActivityDetector`` port using the Silero VAD bundled with faster-whisper."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 100,
    ) -> None:
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ModelNotReadyError("faster-whisper is not installed") from exc
        self._get_speech_timestamps = get_speech_timestamps
        self._options_factory: Any = VadOptions
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

    def detect(self, buffer: AudioBuffer, ctx: CallContext) -> list[TimeInterval]:
        ctx.check()
        mono = buffer.to_mono().samples[0]
        options = self._options_factory(
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        try:
            chunks = self._get_speech_timestamps(mono, options)
        except Exception as exc:  # noqa: BLE001 - map to a domain error
            raise SasttError(f"VAD failed: {exc}") from exc

        rate = buffer.sample_rate
        intervals: list[TimeInterval] = []
        for chunk in chunks:
            start = buffer.start_ms + int(round(chunk["start"] * 1000 / rate))
            end = buffer.start_ms + int(round(chunk["end"] * 1000 / rate))
            if end > start:
                intervals.append(TimeInterval(start, end))
        return intervals


__all__ = ["FasterWhisperRecognizer", "SileroVoiceActivityDetector"]
