"""Event log, replay and idempotency primitives — spec 3, 8.1, 8.2, 10.2, 15."""

from __future__ import annotations

import pytest

from sastt.adapters.persistence import InMemoryEventStore, InMemoryJobStore
from sastt.adapters.persistence.memory import IdempotencyConflictError
from sastt.domain.errors import TenantAccessDeniedError
from sastt.domain.events import (
    Clock,
    EventType,
    JobState,
    SessionEventLog,
    idempotency_fingerprint,
    new_id,
    new_ulid,
)

pytestmark = pytest.mark.unit


class TestIdentifiers:
    def test_ulids_are_sortable_by_time(self) -> None:
        early = new_ulid(1_700_000_000_000)
        late = new_ulid(1_700_000_001_000)
        assert early < late

    def test_prefixed_ids(self) -> None:
        assert new_id("evt").startswith("evt_")
        assert new_id("ses").startswith("ses_")


class TestSessionEventLog:
    def test_sequence_numbers_are_monotonic(self, clock: Clock) -> None:
        log = SessionEventLog("ses_1")
        events = [
            log.append(event_type=EventType.SESSION_STARTED, clock=clock),
            log.append(event_type=EventType.TRANSCRIPT_PROVISIONAL, clock=clock),
            log.append(event_type=EventType.TRANSCRIPT_REVISION, clock=clock, revision=2),
        ]
        assert [event.sequence_number for event in events] == [1, 2, 3]
        assert log.last_sequence_number == 3

    def test_replay_returns_only_newer_events(self, clock: Clock) -> None:
        log = SessionEventLog("ses_1")
        for _ in range(4):
            log.append(event_type=EventType.TRANSCRIPT_PROVISIONAL, clock=clock)
        replayed = log.replay_from(2)
        assert [event.sequence_number for event in replayed] == [3, 4]

    def test_replay_rejects_a_sequence_ahead_of_the_server(self, clock: Clock) -> None:
        log = SessionEventLog("ses_1")
        log.append(event_type=EventType.SESSION_STARTED, clock=clock)
        with pytest.raises(ValueError):
            log.replay_from(5)

    def test_replay_outside_the_retention_window_is_an_error(self, clock: Clock) -> None:
        log = SessionEventLog("ses_1", retention=2)
        for _ in range(4):
            log.append(event_type=EventType.TRANSCRIPT_PROVISIONAL, clock=clock)
        with pytest.raises(ValueError):
            log.replay_from(0)

    def test_a_final_event_is_never_duplicated(self, clock: Clock) -> None:
        log = SessionEventLog("ses_1")
        first = log.append(
            event_type=EventType.TRANSCRIPT_FINAL, clock=clock, is_final=True, dedup_key="seg-1"
        )
        second = log.append(
            event_type=EventType.TRANSCRIPT_FINAL, clock=clock, is_final=True, dedup_key="seg-1"
        )
        assert first.event_id == second.event_id
        assert len(log.final_events()) == 1

    def test_revision_must_be_positive(self, clock: Clock) -> None:
        log = SessionEventLog("ses_1")
        with pytest.raises(ValueError):
            log.append(event_type=EventType.TRANSCRIPT_REVISION, clock=clock, revision=0)


class TestIdempotency:
    def test_fingerprint_is_deterministic(self) -> None:
        args = {"tenant_id": "t1", "idempotency_key": "k1", "input_hash": "h1"}
        assert idempotency_fingerprint(**args) == idempotency_fingerprint(**args)
        assert idempotency_fingerprint(**{**args, "tenant_id": "t2"}) != idempotency_fingerprint(
            **args
        )

    def test_retry_returns_the_same_job(self) -> None:
        store = InMemoryJobStore()
        first, created = store.create_or_get(tenant_id="t1", idempotency_key="k1", input_hash="h1")
        second, created_again = store.create_or_get(
            tenant_id="t1", idempotency_key="k1", input_hash="h1"
        )
        assert created is True and created_again is False
        assert first.job_id == second.job_id

    def test_same_key_with_other_audio_is_a_conflict(self) -> None:
        store = InMemoryJobStore()
        store.create_or_get(tenant_id="t1", idempotency_key="k1", input_hash="h1")
        with pytest.raises(IdempotencyConflictError):
            store.create_or_get(tenant_id="t1", idempotency_key="k1", input_hash="h2")

    def test_keys_are_scoped_per_tenant(self) -> None:
        store = InMemoryJobStore()
        first, _ = store.create_or_get(tenant_id="t1", idempotency_key="k1", input_hash="h1")
        second, created = store.create_or_get(tenant_id="t2", idempotency_key="k1", input_hash="h1")
        assert created is True
        assert first.job_id != second.job_id

    def test_cross_tenant_read_is_denied(self) -> None:
        store = InMemoryJobStore()
        job, _ = store.create_or_get(tenant_id="t1", idempotency_key="k1", input_hash="h1")
        with pytest.raises(TenantAccessDeniedError):
            store.get("t2", job.job_id)

    def test_terminal_jobs_do_not_move_backwards(self) -> None:
        store = InMemoryJobStore()
        job, _ = store.create_or_get(tenant_id="t1", idempotency_key="k1", input_hash="h1")
        store.update_state("t1", job.job_id, JobState.SUCCEEDED)
        with pytest.raises(ValueError):
            store.update_state("t1", job.job_id, JobState.DIARIZING)


class TestEventStore:
    def test_replay_from_store(self, clock: Clock) -> None:
        log = SessionEventLog("ses_1")
        store = InMemoryEventStore()
        for _ in range(3):
            store.append(log.append(event_type=EventType.TRANSCRIPT_PROVISIONAL, clock=clock))
        assert store.last_sequence_number("ses_1") == 3
        assert [event.sequence_number for event in store.replay("ses_1", 1)] == [2, 3]
