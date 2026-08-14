"""PostgreSQL persistence adapters — spec 10.1, 10.2.

The same ports as the in-memory stores, with the constraints of spec 10.2
enforced by the database rather than by Python dictionaries:

* ``jobs (tenant_id, idempotency_key)`` unique, so a retry cannot create a
  second job even when two API replicas race (spec 8.1, FR-001);
* ``transcript_events (session_id, sequence_number)`` and
  ``(session_id, event_id, revision)`` unique, so a reconnect replay can never
  deliver a duplicate final (spec 8.2, S12);
* every read is tenant-scoped, because spec 14.2 forbids trusting a client
  ``tenant_id`` and requires isolation on every query.

Connections come from a pool: a worker holds one only for the length of a
statement, so a long GPU stage never pins a database connection.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sastt.adapters.persistence.memory import IdempotencyConflictError
from sastt.domain.errors import SasttError, TenantAccessDeniedError
from sastt.domain.events import EventType, JobRecord, JobState, ServerEvent, new_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psycopg_pool import ConnectionPool

DEFAULT_DSN = "postgresql://sastt:sastt_dev@127.0.0.1:5432/sastt"


def build_pool(dsn: str | None = None, *, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    """Open a connection pool. Raises a domain error when psycopg is missing."""
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise SasttError(
            "PostgreSQL persistence needs psycopg: pip install 'psycopg[binary,pool]'"
        ) from exc
    pool = ConnectionPool(dsn or DEFAULT_DSN, min_size=min_size, max_size=max_size, open=True)
    pool.wait(timeout=10.0)
    return pool


class PostgresJobStore:
    """``JobStore`` port backed by the ``jobs`` table."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def create_or_get(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        input_hash: str,
        config_version: str | None = None,
    ) -> tuple[JobRecord, bool]:
        job_id = new_id("job")
        with self._pool.connection() as connection, connection.cursor() as cursor:
            # ON CONFLICT DO NOTHING makes the insert the arbiter: whoever wins
            # the unique index owns the job, and the loser reads it back. Two
            # concurrent retries therefore cannot produce two jobs (spec 8.1).
            cursor.execute(
                """
                INSERT INTO jobs (id, tenant_id, idempotency_key, state, input_hash,
                                  config_version)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, idempotency_key) DO NOTHING
                RETURNING id
                """,
                (
                    job_id,
                    tenant_id,
                    idempotency_key,
                    JobState.QUEUED.value,
                    input_hash,
                    config_version,
                ),
            )
            inserted = cursor.fetchone()
            if inserted is not None:
                return (
                    JobRecord(
                        job_id=job_id,
                        tenant_id=tenant_id,
                        idempotency_key=idempotency_key,
                        input_hash=input_hash,
                        state=JobState.QUEUED,
                        config_version=config_version,
                    ),
                    True,
                )

            cursor.execute(
                """
                SELECT id, tenant_id, idempotency_key, input_hash, state, config_version,
                       error_code, warnings, degraded_mode
                FROM jobs
                WHERE tenant_id = %s AND idempotency_key = %s
                """,
                (tenant_id, idempotency_key),
            )
            row = cursor.fetchone()
        if row is None:  # pragma: no cover - only on a concurrent delete
            raise SasttError("job vanished between insert and read")
        existing = _job_from_row(row)
        if existing.input_hash != input_hash:
            raise IdempotencyConflictError(
                "idempotency key reused with a different input",
                details={"idempotency_key": idempotency_key},
            )
        return existing, False

    def get(self, tenant_id: str, job_id: str) -> JobRecord:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, tenant_id, idempotency_key, input_hash, state, config_version,
                       error_code, warnings, degraded_mode
                FROM jobs
                WHERE id = %s AND tenant_id = %s
                """,
                (job_id, tenant_id),
            )
            row = cursor.fetchone()
        if row is None:
            # Same error whether the job belongs to another tenant or does not
            # exist: the difference itself would leak information (spec 14.2).
            raise TenantAccessDeniedError(
                "job not found for this tenant", details={"job_id": job_id}
            )
        return _job_from_row(row)

    def update_state(self, tenant_id: str, job_id: str, state: JobState) -> JobRecord:
        job = self.get(tenant_id, job_id)
        job.transition(state)  # domain guard first, so an illegal move never persists
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE jobs
                SET state = %s, error_code = %s, warnings = %s, degraded_mode = %s,
                    updated_at = now()
                WHERE id = %s AND tenant_id = %s
                """,
                (
                    job.state.value,
                    job.error_code,
                    json.dumps(job.warnings),
                    job.degraded,
                    job_id,
                    tenant_id,
                ),
            )
        return job

    def save_result(
        self, tenant_id: str, job_id: str, segments: list[dict[str, Any]], session_id: str
    ) -> int:
        """Persist the canonical final segments (spec 10.2 ``transcript_segments``)."""
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM transcript_segments WHERE job_id = %s AND tenant_id = %s",
                (job_id, tenant_id),
            )
            for segment in segments:
                cursor.execute(
                    """
                    INSERT INTO transcript_segments
                        (id, job_id, session_id, tenant_id, start_ms, end_ms,
                         session_speaker_id, source_track, is_overlap, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        segment.get("event_id") or new_id("seg"),
                        job_id,
                        session_id,
                        tenant_id,
                        int(segment["start_ms"]),
                        int(segment["end_ms"]),
                        segment.get("session_speaker_id"),
                        segment.get("source_track"),
                        bool(segment.get("is_overlap", False)),
                        json.dumps(segment),
                    ),
                )
        return len(segments)

    def load_result(self, tenant_id: str, job_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM transcript_segments
                WHERE job_id = %s AND tenant_id = %s
                ORDER BY start_ms, session_speaker_id NULLS FIRST, source_track NULLS FIRST
                """,
                (job_id, tenant_id),
            )
            return [row[0] for row in cursor.fetchall()]


class PostgresEventStore:
    """``EventStore`` port backed by the append-only ``transcript_events`` table."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def append(self, event: ServerEvent) -> ServerEvent:
        tenant_id = str(event.payload.get("tenant_id") or "unknown")
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT COALESCE(MAX(sequence_number), 0) FROM transcript_events WHERE session_id = %s",
                (event.session_id,),
            )
            row = cursor.fetchone()
            highest = int(row[0]) if row else 0
            if event.sequence_number <= highest:
                raise SasttError(
                    "sequence_number must increase monotonically per session",
                    details={"session_id": event.session_id},
                )
            try:
                cursor.execute(
                    """
                    INSERT INTO transcript_events
                        (id, session_id, tenant_id, sequence_number, revision, event_type,
                         supersedes_event_id, is_final, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.event_id,
                        event.session_id,
                        tenant_id,
                        event.sequence_number,
                        event.revision,
                        event.event_type.value,
                        event.supersedes_event_id,
                        event.is_final,
                        # to_dict() is the wire shape and omits dedup_key, which
                        # realtime needs after a reconnect to suppress a repeated
                        # final (spec 8.2, S12). Store it alongside.
                        json.dumps({**event.to_dict(), "dedup_key": event.dedup_key}),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - unique violation -> domain error
                raise SasttError(
                    "duplicate (session_id, event_id, revision)",
                    details={"event_id": event.event_id},
                ) from exc
        return event

    def replay(self, session_id: str, last_sequence_number: int) -> list[ServerEvent]:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT COALESCE(MIN(sequence_number), 0) FROM transcript_events WHERE session_id = %s",
                (session_id,),
            )
            row = cursor.fetchone()
            lowest = int(row[0]) if row else 0
            if lowest and last_sequence_number < lowest - 1:
                raise SasttError(
                    "requested sequence is outside the retention window",
                    details={"session_id": session_id},
                )
            cursor.execute(
                """
                SELECT payload FROM transcript_events
                WHERE session_id = %s AND sequence_number > %s
                ORDER BY sequence_number
                """,
                (session_id, last_sequence_number),
            )
            return [_event_from_payload(row[0]) for row in cursor.fetchall()]

    def last_sequence_number(self, session_id: str) -> int:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT COALESCE(MAX(sequence_number), 0) FROM transcript_events WHERE session_id = %s",
                (session_id,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0


def _job_from_row(row: tuple[Any, ...]) -> JobRecord:
    return JobRecord(
        job_id=row[0],
        tenant_id=row[1],
        idempotency_key=row[2],
        input_hash=row[3],
        state=JobState(row[4]),
        config_version=row[5],
        error_code=row[6],
        warnings=list(row[7] or []),
        degraded=bool(row[8]),
    )


def _event_from_payload(payload: dict[str, Any]) -> ServerEvent:
    return ServerEvent(
        event_id=str(payload["event_id"]),
        session_id=str(payload["session_id"]),
        sequence_number=int(payload["sequence_number"]),
        event_type=EventType(payload["type"]),
        server_time_ms=int(payload.get("server_time_ms", 0)),
        revision=int(payload.get("revision", 1)),
        supersedes_event_id=payload.get("supersedes_event_id"),
        is_final=bool(payload.get("is_final", False)),
        payload=dict(payload.get("payload", {})),
        model_versions=dict(payload.get("model_versions", {})),
        config_version=payload.get("config_version"),
        dedup_key=payload.get("dedup_key"),
    )


__all__ = [
    "DEFAULT_DSN",
    "PostgresEventStore",
    "PostgresJobStore",
    "build_pool",
]
