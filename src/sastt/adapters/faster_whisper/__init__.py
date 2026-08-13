"""faster-whisper ASR and Silero VAD — spec 5.5."""

from sastt.adapters.faster_whisper.recognizer import (
    FasterWhisperRecognizer,
    SileroVoiceActivityDetector,
)

__all__ = ["FasterWhisperRecognizer", "SileroVoiceActivityDetector"]
