"""Domain errors and the standard error codes of spec 8.4.

Adapter exceptions MUST be mapped onto these types; framework/model exceptions
MUST NOT leak out of the public API (spec 9).
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes — spec 8.4."""

    UNSUPPORTED_AUDIO_FORMAT = "UNSUPPORTED_AUDIO_FORMAT"
    INVALID_CHANNEL_LAYOUT = "INVALID_CHANNEL_LAYOUT"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    MODEL_LICENSE_DISABLED = "MODEL_LICENSE_DISABLED"
    UNSUPPORTED_CONCURRENCY = "UNSUPPORTED_CONCURRENCY"
    SEPARATION_FAILED = "SEPARATION_FAILED"
    VOICE_ID_UNCALIBRATED = "VOICE_ID_UNCALIBRATED"
    INSUFFICIENT_SPEECH_FOR_EMBEDDING = "INSUFFICIENT_SPEECH_FOR_EMBEDDING"
    QUEUE_OVERLOADED = "QUEUE_OVERLOADED"
    TENANT_ACCESS_DENIED = "TENANT_ACCESS_DENIED"
    SESSION_CLOCK_DISCONTINUITY = "SESSION_CLOCK_DISCONTINUITY"


class SasttError(Exception):
    """Base class for every error crossing a domain boundary."""

    code: ErrorCode | None = None

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "error_code": self.code.value if self.code else None,
            "message": self.message,
            "details": self.details,
        }


class UnsupportedAudioFormatError(SasttError):
    code = ErrorCode.UNSUPPORTED_AUDIO_FORMAT


class InvalidChannelLayoutError(SasttError):
    code = ErrorCode.INVALID_CHANNEL_LAYOUT


class AudioTooLongError(SasttError):
    code = ErrorCode.AUDIO_TOO_LONG


class ModelNotReadyError(SasttError):
    code = ErrorCode.MODEL_NOT_READY


class ModelLicenseDisabledError(SasttError):
    """A checkpoint is disabled for production by manifest/licence gate (spec 20)."""

    code = ErrorCode.MODEL_LICENSE_DISABLED


class UnsupportedConcurrencyError(SasttError):
    """More concurrent speakers than the committed capability (spec 1.2, 5.3)."""

    code = ErrorCode.UNSUPPORTED_CONCURRENCY


class SeparationFailedError(SasttError):
    code = ErrorCode.SEPARATION_FAILED


class VoiceIdUncalibratedError(SasttError):
    """Voice ID asked for while thresholds/calibrator are null (spec 5.10)."""

    code = ErrorCode.VOICE_ID_UNCALIBRATED


class InsufficientSpeechForEmbeddingError(SasttError):
    code = ErrorCode.INSUFFICIENT_SPEECH_FOR_EMBEDDING


class QueueOverloadedError(SasttError):
    code = ErrorCode.QUEUE_OVERLOADED


class TenantAccessDeniedError(SasttError):
    code = ErrorCode.TENANT_ACCESS_DENIED


class SessionClockDiscontinuityError(SasttError):
    code = ErrorCode.SESSION_CLOCK_DISCONTINUITY


class ConfigurationError(SasttError):
    """Startup-time configuration/manifest violation.

    Not part of the public API surface: the process must refuse to start
    instead of serving traffic with an invalid configuration (spec 12).
    """


class InvalidStateTransitionError(SasttError):
    """Speaker identity state machine violation (spec 6)."""


class SchemaInvariantError(SasttError):
    """Public output contract invariant violation (spec 7)."""


__all__ = [
    "AudioTooLongError",
    "ConfigurationError",
    "ErrorCode",
    "InsufficientSpeechForEmbeddingError",
    "InvalidChannelLayoutError",
    "InvalidStateTransitionError",
    "ModelLicenseDisabledError",
    "ModelNotReadyError",
    "QueueOverloadedError",
    "SasttError",
    "SchemaInvariantError",
    "SeparationFailedError",
    "SessionClockDiscontinuityError",
    "TenantAccessDeniedError",
    "UnsupportedAudioFormatError",
    "UnsupportedConcurrencyError",
    "VoiceIdUncalibratedError",
]
