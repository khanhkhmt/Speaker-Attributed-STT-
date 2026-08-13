"""Deterministic fake adapters and synthetic scenarios (spec 16.1.3, 18 M0)."""

from sastt.adapters.fake.models import (
    EMBEDDING_DIMENSION,
    FakeAudioDecoder,
    FakeDiarizer,
    FakeOverlapDetector,
    FakeSpeakerEmbedder,
    FakeSpeechRecognizer,
    FakeSpeechSeparator,
    FakeVoiceActivityDetector,
    FakeVoiceRegistry,
)
from sastt.adapters.fake.scenario import Scenario, ScriptedTurn, SourceOrder

__all__ = [
    "EMBEDDING_DIMENSION",
    "FakeAudioDecoder",
    "FakeDiarizer",
    "FakeOverlapDetector",
    "FakeSpeakerEmbedder",
    "FakeSpeechRecognizer",
    "FakeSpeechSeparator",
    "FakeVoiceActivityDetector",
    "FakeVoiceRegistry",
    "Scenario",
    "ScriptedTurn",
    "SourceOrder",
]
