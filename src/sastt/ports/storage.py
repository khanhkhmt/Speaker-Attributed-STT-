"""Persistence ports — spec 8.1, 10.

Constraints mirrored from spec 10.2: ``(tenant_id, idempotency_key)`` on jobs and
``(session_id, sequence_number)`` / ``(session_id, event_id, revision)`` on events.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.events import JobRecord, JobState, ServerEvent


@runtime_checkable
class JobStore(Protocol):
    """Job persistence with idempotent creation (spec 8.1, FR-001)."""

    def create_or_get(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        input_hash: str,
        config_version: str | None = None,
    ) -> tuple[JobRecord, bool]:
        """Return ``(job, created)``.

        A retry with the same ``(tenant_id, idempotency_key)`` and the same input
        returns the existing job with ``created=False``. The same key with a
        different ``input_hash`` is a conflict and MUST raise.
        """
        ...

    def get(self, tenant_id: str, job_id: str) -> JobRecord:
        """Raises :class:`~sastt.domain.errors.TenantAccessDeniedError` across tenants."""
        ...

    def update_state(self, tenant_id: str, job_id: str, state: JobState) -> JobRecord: ...

    def set_error(self, tenant_id: str, job_id: str, error_code: str) -> JobRecord: ...


@runtime_checkable
class EventStore(Protocol):
    """Durable/ephemeral transcript event log used for reconnect replay (spec 8.2)."""

    def append(self, event: ServerEvent) -> ServerEvent: ...

    def replay(self, session_id: str, last_sequence_number: int) -> list[ServerEvent]: ...

    def last_sequence_number(self, session_id: str) -> int: ...


@runtime_checkable
class ObjectStore(Protocol):
    """Encrypted object storage for input/derived audio (spec 10.1, 10.3)."""

    def put(self, tenant_id: str, key: str, payload: bytes) -> str: ...

    def get(self, tenant_id: str, key: str) -> bytes: ...

    def delete(self, tenant_id: str, key: str) -> bool: ...


__all__ = ["EventStore", "JobStore", "ObjectStore"]
