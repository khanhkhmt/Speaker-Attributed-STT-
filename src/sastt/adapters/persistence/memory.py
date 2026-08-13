"""In-memory persistence adapters — spec 8.1, 10.2.

They enforce the same constraints the database will:
``(tenant_id, idempotency_key)`` on jobs, ``(session_id, sequence_number)`` and
``(session_id, event_id, revision)`` on events, and tenant isolation on reads.
"""

from __future__ import annotations

from sastt.domain.errors import SasttError, TenantAccessDeniedError
from sastt.domain.events import JobRecord, JobState, ServerEvent, new_id


class IdempotencyConflictError(SasttError):
    """Same idempotency key, different input — spec 8.1."""


class InMemoryJobStore:
    """``JobStore`` port implementation."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._by_key: dict[tuple[str, str], str] = {}

    def create_or_get(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        input_hash: str,
        config_version: str | None = None,
    ) -> tuple[JobRecord, bool]:
        key = (tenant_id, idempotency_key)
        existing_id = self._by_key.get(key)
        if existing_id is not None:
            existing = self._jobs[existing_id]
            if existing.input_hash != input_hash:
                raise IdempotencyConflictError(
                    "idempotency key reused with a different input",
                    details={"idempotency_key": idempotency_key},
                )
            return existing, False

        job = JobRecord(
            job_id=new_id("job"),
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            input_hash=input_hash,
            state=JobState.QUEUED,
            config_version=config_version,
        )
        self._jobs[job.job_id] = job
        self._by_key[key] = job.job_id
        return job, True

    def get(self, tenant_id: str, job_id: str) -> JobRecord:
        job = self._jobs.get(job_id)
        if job is None or job.tenant_id != tenant_id:
            raise TenantAccessDeniedError(
                "job not found for this tenant", details={"job_id": job_id}
            )
        return job

    def update_state(self, tenant_id: str, job_id: str, state: JobState) -> JobRecord:
        job = self.get(tenant_id, job_id)
        job.transition(state)
        return job


class InMemoryEventStore:
    """``EventStore`` port implementation with a bounded replay window."""

    def __init__(self, retention: int = 10_000) -> None:
        self.retention = retention
        self._events: dict[str, list[ServerEvent]] = {}
        self._keys: set[tuple[str, str, int]] = set()

    def append(self, event: ServerEvent) -> ServerEvent:
        events = self._events.setdefault(event.session_id, [])
        if events and event.sequence_number <= events[-1].sequence_number:
            raise SasttError(
                "sequence_number must increase monotonically per session",
                details={"session_id": event.session_id},
            )
        key = (event.session_id, event.event_id, event.revision)
        if key in self._keys:
            raise SasttError(
                "duplicate (session_id, event_id, revision)", details={"key": str(key)}
            )
        self._keys.add(key)
        events.append(event)
        if len(events) > self.retention:
            del events[: len(events) - self.retention]
        return event

    def replay(self, session_id: str, last_sequence_number: int) -> list[ServerEvent]:
        events = self._events.get(session_id, [])
        if events and last_sequence_number < events[0].sequence_number - 1:
            raise SasttError(
                "requested sequence is outside the retention window",
                details={"session_id": session_id},
            )
        return [event for event in events if event.sequence_number > last_sequence_number]

    def last_sequence_number(self, session_id: str) -> int:
        events = self._events.get(session_id, [])
        return events[-1].sequence_number if events else 0


class InMemoryObjectStore:
    """``ObjectStore`` port implementation, tenant-scoped (spec 10.1)."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put(self, tenant_id: str, key: str, payload: bytes) -> str:
        self._objects[(tenant_id, key)] = payload
        return f"mem://{tenant_id}/{key}"

    def get(self, tenant_id: str, key: str) -> bytes:
        try:
            return self._objects[(tenant_id, key)]
        except KeyError as exc:
            raise TenantAccessDeniedError("object not found for this tenant") from exc

    def delete(self, tenant_id: str, key: str) -> bool:
        return self._objects.pop((tenant_id, key), None) is not None


__all__ = [
    "IdempotencyConflictError",
    "InMemoryEventStore",
    "InMemoryJobStore",
    "InMemoryObjectStore",
]
