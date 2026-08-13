"""Speech separation port — spec 5.4, 9."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.audio import AudioBuffer
from sastt.domain.speakers import SeparatedBatch
from sastt.observability import CallContext


@runtime_checkable
class SpeechSeparator(Protocol):
    """Separate a mixture crop into ``K`` waveforms.

    The separator returns waveforms, never speaker identities, and the source
    index is local to the crop (spec 5.4).
    """

    @property
    def backend(self) -> str: ...

    @property
    def separator_version(self) -> str: ...

    @property
    def sample_rate(self) -> int:
        """Native rate of the backend (16 kHz MossFormer2, 8 kHz SepFormer)."""
        ...

    @property
    def supported_source_counts(self) -> tuple[int, ...]: ...

    def separate(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        requested_source_count: int,
    ) -> SeparatedBatch:
        """Raises :class:`sastt.domain.errors.SeparationFailedError` on model failure."""
        ...


__all__ = ["SpeechSeparator"]
