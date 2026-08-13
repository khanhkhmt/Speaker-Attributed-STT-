"""Event model, session/job lifecycle and idempotency primitives — spec 8, 10.2, 15.

Design rules enforced here:

* every event carries ``event_id``, a monotonic ``sequence_number``, ``revision``,
  ``server_time`` and model/config versions (spec 8.2);
* a client reconnect replays from ``last_sequence_number`` and MUST NOT receive a
  duplicated final event (spec 8.2, 15);
* a retry with the same ``Idempotency-Key`` MUST NOT produce a second final
  transcript (spec 3, 8.1).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class Clock(Protocol):
    """Injectable wall clock; tests use a deterministic implementation."""

    def now_ms(self) -> int: ...


class SystemClock:
    """Default clock backed by the OS."""

    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _encode_crockford(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid(now_ms: int | None = None) -> str:
    """ULID-style lexicographically sortable identifier (spec 7 example IDs)."""
    timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10), "big")
    return _encode_crockford(timestamp, 10) + _encode_crockford(randomness, 16)


def new_id(prefix: str, now_ms: int | None = None) -> str:
    """Prefixed identifier, e.g. ``ses_01J...`` / ``evt_01J...`` / ``job_01J...``."""
    return f"{prefix}_{new_ulid(now_ms)}"


# --------------------------------------------------------------------------- #
# Lifecycle (spec 8.1)
# --------------------------------------------------------------------------- #


class JobState(str, Enum):
    """Offline job states — spec 8.1."""

    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    DIARIZING = "DIARIZING"
    TRANSCRIBING = "TRANSCRIBING"
    SEPARATING = "SEPARATING"
    LINKING = "LINKING"
    FUSING = "FUSING"
    SUCCEEDED = "SUCCEEDED"
    DEGRADED_SUCCEEDED = "DEGRADED_SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_JOB_STATES: frozenset[JobState] = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.DEGRADED_SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
)


class SessionState(str, Enum):
    """Realtime session states backing the event stream (spec 8.2)."""

    CREATED = "CREATED"
    STREAMING = "STREAMING"
    FINALIZING = "FINALIZING"
    FINALIZED = "FINALIZED"
    FAILED = "FAILED"


class EventType(str, Enum):
    """Server events — spec 8.2."""

    SESSION_STARTED = "session.started"
    TRANSCRIPT_PROVISIONAL = "transcript.provisional"
    TRANSCRIPT_REVISION = "transcript.revision"
    TRANSCRIPT_FINAL = "transcript.final"
    PIPELINE_WARNING = "pipeline.warning"
    SESSION_FINALIZED = "session.finalized"
    SESSION_FAILED = "session.failed"


@dataclass(frozen=True)
class ServerEvent:
    """One event delivered to the client (spec 8.2)."""

    event_id: str
    session_id: str
    sequence_number: int
    event_type: EventType
    server_time_ms: int
    revision: int = 1
    supersedes_event_id: str | None = None
    is_final: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str | None] = field(default_factory=dict)
    config_version: str | None = None
    dedup_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "type": self.event_type.value,
            "server_time_ms": self.server_time_ms,
            "revision": self.revision,
            "supersedes_event_id": self.supersedes_event_id,
            "is_final": self.is_final,
            "payload": self.payload,
            "model_versions": self.model_versions,
            "config_version": self.config_version,
        }


class SessionEventLog:
    """Append-only event log with replay — spec 8.2, 10.2, 15.

    Unique constraints mirrored from spec 10.2:
    ``(session_id, sequence_number)`` and ``(session_id, event_id, revision)``.
    """

    def __init__(self, session_id: str, *, retention: int = 10_000) -> None:
        self.session_id = session_id
        self.retention = retention
        self._events: list[ServerEvent] = []
        self._sequence = 0
        self._keys: set[tuple[str, int]] = set()
        self._final_by_dedup_key: dict[str, ServerEvent] = {}

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[ServerEvent, ...]:
        return tuple(self._events)

    @property
    def last_sequence_number(self) -> int:
        return self._sequence

    def next_sequence_number(self) -> int:
        return self._sequence + 1

    def append(
        self,
        *,
        event_type: EventType,
        clock: Clock,
        revision: int = 1,
        supersedes_event_id: str | None = None,
        is_final: bool = False,
        payload: dict[str, Any] | None = None,
        model_versions: dict[str, str | None] | None = None,
        config_version: str | None = None,
        dedup_key: str | None = None,
        event_id: str | None = None,
    ) -> ServerEvent:
        """Append an event, or return the existing final event for ``dedup_key``.

        Returning the existing event (instead of appending a second one) is what
        keeps reconnect replay and job retries free of duplicated finals.
        """
        if is_final and dedup_key is not None and dedup_key in self._final_by_dedup_key:
            return self._final_by_dedup_key[dedup_key]
        if revision < 1:
            raise ValueError("revision must be >= 1")

        now_ms = clock.now_ms()
        event = ServerEvent(
            event_id=event_id or new_id("evt", now_ms),
            session_id=self.session_id,
            sequence_number=self._sequence + 1,
            event_type=event_type,
            server_time_ms=now_ms,
            revision=revision,
            supersedes_event_id=supersedes_event_id,
            is_final=is_final,
            payload=dict(payload or {}),
            model_versions=dict(model_versions or {}),
            config_version=config_version,
            dedup_key=dedup_key,
        )
        key = (event.event_id, event.revision)
        if key in self._keys:
            raise ValueError(f"duplicate (event_id, revision) {key} in session {self.session_id}")

        self._keys.add(key)
        self._sequence = event.sequence_number
        self._events.append(event)
        if is_final and dedup_key is not None:
            self._final_by_dedup_key[dedup_key] = event
        if len(self._events) > self.retention:
            self._events = self._events[-self.retention :]
        return event

    def replay_from(self, last_sequence_number: int) -> list[ServerEvent]:
        """Events strictly after ``last_sequence_number`` (client reconnect, spec 8.2).

        Raises when the requested point has already fallen out of the retention
        window, so the client can restart instead of silently losing events.
        """
        if last_sequence_number < 0:
            raise ValueError("last_sequence_number must be >= 0")
        if last_sequence_number > self._sequence:
            raise ValueError(
                f"last_sequence_number {last_sequence_number} is ahead of the server "
                f"({self._sequence})"
            )
        if self._events and last_sequence_number < self._events[0].sequence_number - 1:
            raise ValueError("requested sequence is outside the retention window")
        return [e for e in self._events if e.sequence_number > last_sequence_number]

    def final_events(self) -> list[ServerEvent]:
        return [e for e in self._events if e.is_final]


# --------------------------------------------------------------------------- #
# Idempotency (spec 3, 8.1, 10.2)
# --------------------------------------------------------------------------- #


def idempotency_fingerprint(*, tenant_id: str, idempotency_key: str, input_hash: str) -> str:
    """Deterministic fingerprint for the ``(tenant_id, idempotency_key)`` constraint.

    ``input_hash`` is folded in so the same key replayed with different audio is
    detected as a conflict rather than silently returning the first job.
    """
    payload = f"{tenant_id}|{idempotency_key}|{input_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class JobRecord:
    """Persisted job — mirrors ``jobs(...)`` of spec 10.2."""

    job_id: str
    tenant_id: str
    idempotency_key: str
    input_hash: str
    state: JobState = JobState.QUEUED
    config_version: str | None = None
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def fingerprint(self) -> str:
        return idempotency_fingerprint(
            tenant_id=self.tenant_id,
            idempotency_key=self.idempotency_key,
            input_hash=self.input_hash,
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_JOB_STATES

    def transition(self, state: JobState) -> None:
        if self.is_terminal and state is not self.state:
            raise ValueError(f"job {self.job_id} is terminal in {self.state.value}")
        self.state = state


__all__ = [
    "TERMINAL_JOB_STATES",
    "Clock",
    "EventType",
    "JobRecord",
    "JobState",
    "ServerEvent",
    "SessionEventLog",
    "SessionState",
    "SystemClock",
    "idempotency_fingerprint",
    "new_id",
    "new_ulid",
]
