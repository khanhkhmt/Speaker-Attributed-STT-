"""Redis work queues against a real server — spec 11.3, 3.

Marked ``db``: skipped unless ``SASTT_TEST_REDIS_URL`` is set, so ordinary CI
needs no Redis (spec 16.3).

The properties under test are the ones that decide whether audio can be lost:
priority between realtime and batch, backpressure instead of silent acceptance,
and recovery of tasks a worker was holding when it died.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def client():
    url = os.environ.get("SASTT_TEST_REDIS_URL")
    if not url:
        pytest.skip("set SASTT_TEST_REDIS_URL to run the Redis contract tests")
    try:
        from sastt.adapters.queue.redis_queue import build_client
    except Exception as exc:  # noqa: BLE001 - redis client missing
        pytest.skip(f"redis unavailable: {exc}")
    try:
        return build_client(url)
    except Exception as exc:  # noqa: BLE001 - server unreachable
        pytest.skip(f"redis unreachable: {exc}")


@pytest.fixture
def queue(client):
    """A namespace per test, so runs cannot see each other's tasks."""
    from sastt.adapters.queue.redis_queue import RedisTaskQueue

    namespace = f"sasttest:{uuid.uuid4().hex[:12]}"
    created = RedisTaskQueue(client, namespace=namespace)
    yield created
    for key in client.scan_iter(f"{namespace}:*"):
        client.delete(key)


class TestQueueBasics:
    def test_a_reserved_task_round_trips(self, queue) -> None:
        from sastt.adapters.queue.redis_queue import QUEUE_SPEAKER_BATCH

        queue.enqueue(QUEUE_SPEAKER_BATCH, tenant_id="t1", job_id="job-1", payload={"a": 1})
        task = queue.reserve([QUEUE_SPEAKER_BATCH], "worker-1", timeout_seconds=1)

        assert task is not None
        assert task.job_id == "job-1"
        assert task.payload == {"a": 1}
        assert queue.ack(task, "worker-1") is True

    def test_unknown_queue_is_refused(self, queue) -> None:
        from sastt.domain.errors import SasttError

        with pytest.raises(SasttError):
            queue.enqueue("not.a.queue", tenant_id="t1", job_id="job-x")

    def test_realtime_is_served_before_batch(self, queue) -> None:
        """Spec 11.3: realtime outranks batch, so a long file cannot block a call."""
        from sastt.adapters.queue.redis_queue import QUEUE_ASR_BATCH, QUEUE_ASR_REALTIME

        queue.enqueue(QUEUE_ASR_BATCH, tenant_id="t1", job_id="batch-job")
        queue.enqueue(QUEUE_ASR_REALTIME, tenant_id="t1", job_id="realtime-job")

        first = queue.reserve([QUEUE_ASR_REALTIME, QUEUE_ASR_BATCH], "worker-1", timeout_seconds=1)

        assert first is not None
        assert first.job_id == "realtime-job"


class TestBackpressure:
    def test_a_full_queue_is_refused_not_accepted(self, client) -> None:
        """Spec 11.3 ladder: reject with QUEUE_OVERLOADED rather than drop frames."""
        from sastt.adapters.queue.redis_queue import (
            QUEUE_SPEAKER_BATCH,
            QueueOverloadedError,
            RedisTaskQueue,
        )

        namespace = f"sasttest:{uuid.uuid4().hex[:12]}"
        bounded = RedisTaskQueue(client, namespace=namespace, max_depth=2)
        try:
            bounded.enqueue(QUEUE_SPEAKER_BATCH, tenant_id="t1", job_id="a")
            bounded.enqueue(QUEUE_SPEAKER_BATCH, tenant_id="t1", job_id="b")

            with pytest.raises(QueueOverloadedError):
                bounded.enqueue(QUEUE_SPEAKER_BATCH, tenant_id="t1", job_id="c")
        finally:
            for key in client.scan_iter(f"{namespace}:*"):
                client.delete(key)


class TestRecovery:
    def test_an_unacked_task_survives_a_dead_worker(self, queue) -> None:
        """Spec 3: a worker restart must not lose an acknowledged job."""
        from sastt.adapters.queue.redis_queue import QUEUE_SPEAKER_BATCH

        queue.enqueue(QUEUE_SPEAKER_BATCH, tenant_id="t1", job_id="job-crash")
        reserved = queue.reserve([QUEUE_SPEAKER_BATCH], "doomed-worker", timeout_seconds=1)
        assert reserved is not None
        assert queue.depth(QUEUE_SPEAKER_BATCH) == 0  # held, not lost

        queue.visibility_timeout_ms = 0  # pretend the lease expired
        recovered = queue.requeue_stale(QUEUE_SPEAKER_BATCH, "doomed-worker")

        assert recovered == 1
        assert queue.depth(QUEUE_SPEAKER_BATCH) == 1

    def test_a_task_is_retried_then_parked(self, queue) -> None:
        from sastt.adapters.queue.redis_queue import QUEUE_SPEAKER_BATCH

        queue.enqueue(QUEUE_SPEAKER_BATCH, tenant_id="t1", job_id="job-retry")
        task = queue.reserve([QUEUE_SPEAKER_BATCH], "worker-1", timeout_seconds=1)
        assert task is not None

        assert queue.retry(task, "worker-1", max_attempts=2) is True
        again = queue.reserve([QUEUE_SPEAKER_BATCH], "worker-1", timeout_seconds=1)
        assert again is not None
        assert again.attempts == 1

        # Second failure exhausts the budget: parked, not looped forever.
        assert queue.retry(again, "worker-1", max_attempts=2) is False
        assert queue.depth(QUEUE_SPEAKER_BATCH) == 0


class TestObservability:
    def test_stats_expose_depth_and_age(self, queue) -> None:
        """Spec 13.1 autoscales on queue age, so it has to be observable."""
        from sastt.adapters.queue.redis_queue import QUEUE_FUSION

        queue.enqueue(QUEUE_FUSION, tenant_id="t1", job_id="job-1")
        stats = queue.stats()

        assert stats[QUEUE_FUSION]["depth"] == 1
        assert stats[QUEUE_FUSION]["oldest_age_ms"] >= 0
