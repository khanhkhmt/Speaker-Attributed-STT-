"""PostgreSQL + pgvector adapters against a real database — spec 10, 14.

Marked ``db``: skipped unless ``SASTT_TEST_DATABASE_URL`` points at a migrated
database, so ordinary CI needs no PostgreSQL (spec 16.3). These assert the
behaviour the in-memory stores can only imitate — real unique constraints, real
tenant isolation and real cosine search.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from sastt.domain.errors import TenantAccessDeniedError
from sastt.domain.events import EventType, JobState, ServerEvent
from sastt.domain.speakers import SpeakerEmbedding, l2_normalize
from sastt.observability import CallContext

pytestmark = pytest.mark.db

DIMENSION = 192
MODEL_VERSION = "campplus@test"


@pytest.fixture(scope="module")
def pool():
    url = os.environ.get("SASTT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set SASTT_TEST_DATABASE_URL to run the PostgreSQL contract tests")
    try:
        from sastt.adapters.persistence import build_pool
    except Exception as exc:  # noqa: BLE001 - psycopg missing
        pytest.skip(f"psycopg unavailable: {exc}")
    try:
        created = build_pool(url)
    except Exception as exc:  # noqa: BLE001 - database unreachable
        pytest.skip(f"database unreachable: {exc}")
    yield created
    created.close()


@pytest.fixture
def tenant() -> str:
    return f"tenant_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def ctx() -> CallContext:
    return CallContext(stage="db_contract")


def embedding(seed: int, *, speech_ms: int = 5_000, origin: str = "clean") -> SpeakerEmbedding:
    rng = np.random.default_rng(seed)
    return SpeakerEmbedding(
        vector=l2_normalize(rng.normal(size=DIMENSION).astype(np.float32)),
        model_version=MODEL_VERSION,
        quality=0.9,
        speech_duration_ms=speech_ms,
        origin=origin,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Jobs (spec 8.1, 10.2, FR-001)
# --------------------------------------------------------------------------- #


class TestJobStore:
    def test_same_idempotency_key_returns_one_job(self, pool, tenant) -> None:
        from sastt.adapters.persistence import PostgresJobStore

        store = PostgresJobStore(pool)
        first, created_first = store.create_or_get(
            tenant_id=tenant, idempotency_key="key-1", input_hash="hash-1"
        )
        second, created_second = store.create_or_get(
            tenant_id=tenant, idempotency_key="key-1", input_hash="hash-1"
        )

        assert created_first is True
        assert created_second is False
        assert first.job_id == second.job_id

    def test_same_key_with_a_different_input_is_a_conflict(self, pool, tenant) -> None:
        from sastt.adapters.persistence import IdempotencyConflictError, PostgresJobStore

        store = PostgresJobStore(pool)
        store.create_or_get(tenant_id=tenant, idempotency_key="key-2", input_hash="hash-a")

        with pytest.raises(IdempotencyConflictError):
            store.create_or_get(tenant_id=tenant, idempotency_key="key-2", input_hash="hash-b")

    def test_another_tenant_cannot_read_the_job(self, pool, tenant) -> None:
        """Spec 14.2, S15: isolation is enforced by the query, not by the caller."""
        from sastt.adapters.persistence import PostgresJobStore

        store = PostgresJobStore(pool)
        job, _ = store.create_or_get(tenant_id=tenant, idempotency_key="key-3", input_hash="hash-3")

        with pytest.raises(TenantAccessDeniedError):
            store.get("tenant_someone_else", job.job_id)

    def test_state_transitions_persist(self, pool, tenant) -> None:
        from sastt.adapters.persistence import PostgresJobStore

        store = PostgresJobStore(pool)
        job, _ = store.create_or_get(tenant_id=tenant, idempotency_key="key-4", input_hash="hash-4")
        store.update_state(tenant, job.job_id, JobState.PREPROCESSING)

        assert store.get(tenant, job.job_id).state is JobState.PREPROCESSING


# --------------------------------------------------------------------------- #
# Event log (spec 8.2, S12)
# --------------------------------------------------------------------------- #


class TestEventStore:
    def _event(self, session_id: str, sequence: int, *, final: bool = False) -> ServerEvent:
        return ServerEvent(
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            sequence_number=sequence,
            event_type=EventType.TRANSCRIPT_FINAL if final else EventType.TRANSCRIPT_PROVISIONAL,
            server_time_ms=sequence * 100,
            is_final=final,
        )

    def test_replay_returns_only_events_after_the_cursor(self, pool) -> None:
        from sastt.adapters.persistence import PostgresEventStore

        store = PostgresEventStore(pool)
        session = f"ses_{uuid.uuid4().hex[:16]}"
        for sequence in (1, 2, 3):
            store.append(self._event(session, sequence))

        replayed = store.replay(session, last_sequence_number=1)

        assert [event.sequence_number for event in replayed] == [2, 3]
        assert store.last_sequence_number(session) == 3

    def test_sequence_numbers_must_increase(self, pool) -> None:
        from sastt.adapters.persistence import PostgresEventStore
        from sastt.domain.errors import SasttError

        store = PostgresEventStore(pool)
        session = f"ses_{uuid.uuid4().hex[:16]}"
        store.append(self._event(session, 5))

        with pytest.raises(SasttError):
            store.append(self._event(session, 5))


# --------------------------------------------------------------------------- #
# Voice registry on pgvector (spec 5.10, 8.3, 10.3, S15)
# --------------------------------------------------------------------------- #


class TestPgVectorRegistry:
    def _registry(self, pool, *, calibrated: bool):
        from sastt.adapters.persistence import PgVectorVoiceRegistry

        return PgVectorVoiceRegistry(
            pool,
            embedding_model_version=MODEL_VERSION,
            accept_threshold=0.55 if calibrated else None,
            ambiguous_margin=0.05 if calibrated else None,
        )

    def test_uncalibrated_registry_fails_closed(self, pool, tenant, ctx) -> None:
        """Spec 5.10, 18 rule 7: no thresholds means no identification, ever."""
        registry = self._registry(pool, calibrated=False)
        identity = f"emp_{uuid.uuid4().hex[:10]}"
        registry.enroll(tenant, identity, [embedding(1), embedding(1), embedding(1)], ctx)

        decision = registry.identify(tenant, embedding(1), ctx)

        assert decision.status == "uncalibrated"
        assert decision.registry_speaker_id is None

    def test_enrollment_reports_rejected_clips(self, pool, tenant, ctx) -> None:
        """Spec 8.3: enrollment returns a quality report, not just success."""
        registry = self._registry(pool, calibrated=True)
        identity = f"emp_{uuid.uuid4().hex[:10]}"

        report = registry.enroll(
            tenant,
            identity,
            [
                embedding(2),
                embedding(3, speech_ms=1_000),  # too short
                embedding(4, origin="separated"),  # overlapped source
            ],
            ctx,
        )

        assert report.accepted_clips == 1
        assert report.rejected_clips == 2
        assert report.meets_policy is False
        assert "fewer_than_minimum_clips" in report.reasons

    def test_enrolled_voice_is_identified(self, pool, tenant, ctx) -> None:
        registry = self._registry(pool, calibrated=True)
        identity = f"emp_{uuid.uuid4().hex[:10]}"
        registry.enroll(
            tenant,
            identity,
            [embedding(10), embedding(11), embedding(12)],
            ctx,
            display_name="Nguyen Van B",
        )

        decision = registry.identify(tenant, embedding(10), ctx)

        assert decision.status == "enrolled"
        assert decision.registry_speaker_id == identity
        assert decision.speaker_name == "Nguyen Van B"

    def test_unseen_voice_is_rejected(self, pool, tenant, ctx) -> None:
        """S05: an unenrolled speaker must be rejected, not forced onto someone."""
        registry = self._registry(pool, calibrated=True)
        identity = f"emp_{uuid.uuid4().hex[:10]}"
        registry.enroll(tenant, identity, [embedding(20), embedding(21), embedding(22)], ctx)

        decision = registry.identify(tenant, embedding(999), ctx)

        assert decision.status == "unknown"
        assert decision.registry_speaker_id is None

    def test_templates_of_another_tenant_are_invisible(self, pool, ctx) -> None:
        """S15: the same voice enrolled elsewhere must not match here."""
        registry = self._registry(pool, calibrated=True)
        owner, intruder = f"t_{uuid.uuid4().hex[:8]}", f"t_{uuid.uuid4().hex[:8]}"
        identity = f"emp_{uuid.uuid4().hex[:10]}"
        registry.enroll(owner, identity, [embedding(30), embedding(31), embedding(32)], ctx)

        decision = registry.identify(intruder, embedding(30), ctx)

        assert decision.status == "unknown"
        assert registry.identity_exists(intruder, identity) is False
        with pytest.raises(TenantAccessDeniedError):
            registry.prototypes_of(intruder, identity)

    def test_a_different_embedding_version_never_matches(self, pool, tenant, ctx) -> None:
        """Spec 5.6: vectors from another model version are not comparable."""
        registry = self._registry(pool, calibrated=True)
        identity = f"emp_{uuid.uuid4().hex[:10]}"
        registry.enroll(tenant, identity, [embedding(40), embedding(41), embedding(42)], ctx)

        other = embedding(40)
        object.__setattr__(other, "model_version", "campplus@other")
        decision = registry.identify(tenant, other, ctx)

        assert decision.status == "unknown"
        assert decision.reason == "embedding_model_mismatch"

    def test_deleting_an_identity_removes_its_templates(self, pool, tenant, ctx) -> None:
        """Spec 10.3, FR-014: deletion cascades and is audited."""
        registry = self._registry(pool, calibrated=True)
        identity = f"emp_{uuid.uuid4().hex[:10]}"
        registry.enroll(tenant, identity, [embedding(50), embedding(51), embedding(52)], ctx)

        assert registry.delete_identity(tenant, identity, ctx) is True
        assert registry.identity_exists(tenant, identity) is False
        assert registry.identify(tenant, embedding(50), ctx).status == "unknown"

        with pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM voice_templates WHERE identity_id = %s", (identity,)
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                """
                SELECT count(*) FROM audit_events
                WHERE subject_id = %s AND action = 'voice_identity.delete'
                """,
                (identity,),
            )
            assert cursor.fetchone()[0] == 1

    def test_another_tenant_cannot_delete_the_identity(self, pool, ctx) -> None:
        registry = self._registry(pool, calibrated=True)
        owner, intruder = f"t_{uuid.uuid4().hex[:8]}", f"t_{uuid.uuid4().hex[:8]}"
        identity = f"emp_{uuid.uuid4().hex[:10]}"
        registry.enroll(owner, identity, [embedding(60), embedding(61), embedding(62)], ctx)

        assert registry.delete_identity(intruder, identity, ctx) is False
        assert registry.identity_exists(owner, identity) is True
