"""Audio decoding port — spec 5.1, 9."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.audio import AudioAsset
from sastt.observability import CallContext


@runtime_checkable
class AudioDecoder(Protocol):
    """Decode an input once, preserving the original and its channel map.

    Implementations MUST reject corrupt input, negative duration, NaN/Inf and a
    channel count outside 1-8, and MUST return the mono 16 kHz derivative
    alongside the untouched original (spec 5.1).
    """

    def decode(
        self, payload: bytes, ctx: CallContext, *, container_hint: str | None = None
    ) -> AudioAsset:
        """Decode raw bytes into an :class:`AudioAsset`.

        Raises:
            sastt.domain.errors.UnsupportedAudioFormatError: undecodable input.
            sastt.domain.errors.InvalidChannelLayoutError: channel count out of range.
            sastt.domain.errors.AudioTooLongError: longer than the configured limit.
        """
        ...


__all__ = ["AudioDecoder"]
