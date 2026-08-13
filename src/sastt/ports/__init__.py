"""Domain ports — spec 9.

Every model and every piece of infrastructure sits behind one of these typed
protocols so a backend can be swapped by configuration (spec 0.3). Adapters map
their exceptions onto :mod:`sastt.domain.errors`; framework exceptions MUST NOT
leak to the public API, and every call takes a
:class:`sastt.observability.CallContext` carrying timeout, cancellation and
metrics context.

Spec 17 names six port modules; ``audio`` and ``fusion`` are added here because
spec 9 mandates ``AudioDecoder``, ``ConfidenceCalibrator`` and ``FusionEngine``
as ports too.
"""

from sastt.ports.asr import SpeechRecognizer
from sastt.ports.audio import AudioDecoder
from sastt.ports.diarization import (
    ConcurrentSpeakerCounter,
    Diarizer,
    OverlapDetector,
    VoiceActivityDetector,
)
from sastt.ports.embedding import SessionClusterer, SourceLinker, SpeakerEmbedder
from sastt.ports.fusion import ConfidenceCalibrator, FusionEngine
from sastt.ports.registry import VoiceRegistry
from sastt.ports.separation import SpeechSeparator
from sastt.ports.storage import EventStore, JobStore, ObjectStore

__all__ = [
    "AudioDecoder",
    "ConcurrentSpeakerCounter",
    "ConfidenceCalibrator",
    "Diarizer",
    "EventStore",
    "FusionEngine",
    "JobStore",
    "ObjectStore",
    "OverlapDetector",
    "SessionClusterer",
    "SourceLinker",
    "SpeakerEmbedder",
    "SpeechRecognizer",
    "SpeechSeparator",
    "VoiceActivityDetector",
    "VoiceRegistry",
]
