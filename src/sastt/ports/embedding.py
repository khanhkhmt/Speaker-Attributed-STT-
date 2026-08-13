"""Speaker embedding, clustering and source-linking ports — spec 5.6, 5.7, 5.8, 9."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.audio import AudioBuffer, TimeInterval
from sastt.domain.speakers import (
    EmbeddingOrigin,
    LinkingDecision,
    SpeakerEmbedding,
    SpeakerPrototype,
)
from sastt.observability import CallContext


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """Speaker embedding extraction — spec 5.6.

    Embeddings come from VAD-ed speech, are L2-normalised and carry their model
    version; embeddings of different versions MUST NOT be compared.
    """

    @property
    def model_version(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        speech_intervals: list[TimeInterval] | None = None,
        origin: EmbeddingOrigin = "clean",
        source_track: int | None = None,
    ) -> SpeakerEmbedding:
        """Raises :class:`~sastt.domain.errors.InsufficientSpeechForEmbeddingError`
        when the clean speech is shorter than the configured minimum."""
        ...


@runtime_checkable
class SessionClusterer(Protocol):
    """Incremental session clustering — spec 5.7.

    Cosine similarity plus temporal constraints, capped at the maximum number of
    session speakers. Overlapping activity creates a cannot-link constraint.
    """

    def assign(
        self,
        embedding: SpeakerEmbedding,
        ctx: CallContext,
        *,
        interval: TimeInterval | None = None,
    ) -> str | None: ...


@runtime_checkable
class SourceLinker(Protocol):
    """One-to-one assignment of separated sources to session speakers — spec 5.8."""

    def link(
        self,
        embeddings: list[SpeakerEmbedding | None],
        prototypes: list[SpeakerPrototype],
        ctx: CallContext,
        *,
        previous_mapping: dict[int, str] | None = None,
    ) -> list[LinkingDecision]: ...


__all__ = ["SessionClusterer", "SourceLinker", "SpeakerEmbedder"]
