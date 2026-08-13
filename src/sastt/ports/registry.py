"""Voice registry port — spec 5.10, 8.3, 10, 14.

Every operation is tenant-scoped: a cross-tenant read MUST raise
:class:`~sastt.domain.errors.TenantAccessDeniedError` (spec 14.2, S15). Meeting
audio never updates an enrollment automatically (spec 1.3, 5.10).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.speakers import EnrollmentReport, SpeakerEmbedding, VoiceIdDecision
from sastt.observability import CallContext


@runtime_checkable
class VoiceRegistry(Protocol):
    """Open-set speaker registry."""

    @property
    def embedding_model_version(self) -> str: ...

    def enroll(
        self,
        tenant_id: str,
        identity_id: str,
        embeddings: list[SpeakerEmbedding],
        ctx: CallContext,
        *,
        display_name: str | None = None,
        consent_ref: str | None = None,
    ) -> EnrollmentReport:
        """Store enrollment prototypes and return a quality report (spec 8.3)."""
        ...

    def identify(
        self,
        tenant_id: str,
        embedding: SpeakerEmbedding,
        ctx: CallContext,
    ) -> VoiceIdDecision:
        """Open-set decision: accept, reject as unknown, or report ambiguous.

        While thresholds/calibrator are null the implementation MUST fail closed
        and return ``status="uncalibrated"`` rather than guessing (spec 5.10).
        """
        ...

    def delete_identity(self, tenant_id: str, identity_id: str, ctx: CallContext) -> bool:
        """Delete identity, templates, index entries and cached prototypes (spec 10.3)."""
        ...

    def identity_exists(self, tenant_id: str, identity_id: str) -> bool: ...


__all__ = ["VoiceRegistry"]
