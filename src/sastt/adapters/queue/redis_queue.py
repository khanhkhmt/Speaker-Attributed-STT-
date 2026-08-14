"""Redis work queues — spec 11.3.

One queue per stage, so ASR and speaker work scale independently and a realtime
burst cannot be stuck behind a four-hour batch job (spec 11.3):

    asr.realtime  asr.batch  speaker.realtime  speaker.batch
    separation.two_source  separation.beta  gss.batch  fusion

Delivery is at-least-once. ``BRPOPLPUSH`` moves a task to a per-consumer
processing list, so a worker that dies mid-task leaves the task recoverable
instead of losing it — spec 3 requires that a worker restart never drops an
acknowledged job. Completion is an explicit ``ack``; a crash without ack leaves
the task for :meth:`RedisTaskQueue.requeue_stale`.

Realtime queues are polled before batch ones and every task carries its tenant,
which is what the per-tenant quota of spec 11.3 needs.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sastt.domain.errors import SasttError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from redis import Redis

DEFAULT_URL = "redis://127.0.0.1:6379/0"

#: Spec 11.3 queues, realtime before batch so priority is a list order.
QUEUE_ASR_REALTIME = "asr.realtime"
QUEUE_ASR_BATCH = "asr.batch"
QUEUE_SPEAKER_REALTIME = "speaker.realtime"
QUEUE_SPEAKER_BATCH = "speaker.batch"
QUEUE_SEPARATION_TWO_SOURCE = "separation.two_source"
QUEUE_SEPARATION_BETA = "separation.beta"
QUEUE_GSS_BATCH = "gss.batch"
QUEUE_FUSION = "fusion"

ALL_QUEUES: tuple[str, ...] = (
    QUEUE_ASR_REALTIME,
    QUEUE_SPEAKER_REALTIME,
    QUEUE_ASR_BATCH,
    QUEUE_SPEAKER_BATCH,
    QUEUE_SEPARATION_TWO_SOURCE,
    QUEUE_SEPARATION_BETA,
    QUEUE_GSS_BATCH,
    QUEUE_FUSION,
)


class QueueOverloadedError(SasttError):
    """Backpressure: the queue is past its bound — spec 8.4 ``QUEUE_OVERLOADED``."""


@dataclass(frozen=True)
class Task:
    """One unit of queued work."""

    task_id: str
    queue: str
    tenant_id: str
    job_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    enqueued_ms: int = 0
    attempts: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "task_id": self.task_id,
                "queue": self.queue,
                "tenant_id": self.tenant_id,
                "job_id": self.job_id,
                "payload": self.payload,
                "enqueued_ms": self.enqueued_ms,
                "attempts": self.attempts,
            }
        )

    @staticmethod
    def from_json(raw: str | bytes) -> Task:
        data = json.loads(raw)
        return Task(
            task_id=str(data["task_id"]),
            queue=str(data["queue"]),
            tenant_id=str(data["tenant_id"]),
            job_id=str(data["job_id"]),
            payload=dict(data.get("payload", {})),
            enqueued_ms=int(data.get("enqueued_ms", 0)),
            attempts=int(data.get("attempts", 0)),
        )

    @property
    def age_ms(self) -> int:
        return max(0, _now_ms() - self.enqueued_ms)


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_client(url: str | None = None) -> Redis:
    try:
        from redis import Redis
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise SasttError("the Redis queue needs redis: pip install redis") from exc
    client: Redis = Redis.from_url(url or DEFAULT_URL, decode_responses=True)
    client.ping()
    return client


class RedisTaskQueue:
    """Reliable at-least-once queue over Redis lists."""

    def __init__(
        self,
        client: Redis,
        *,
        namespace: str = "sastt",
        max_depth: int = 10_000,
        visibility_timeout_ms: int = 300_000,
    ) -> None:
        self._client = client
        self.namespace = namespace
        self.max_depth = max_depth
        self.visibility_timeout_ms = visibility_timeout_ms

    def key(self, queue: str) -> str:
        return f"{self.namespace}:queue:{queue}"

    def processing_key(self, queue: str, consumer: str) -> str:
        return f"{self.namespace}:processing:{queue}:{consumer}"

    # -- producing ----------------------------------------------------------- #

    def enqueue(
        self,
        queue: str,
        *,
        tenant_id: str,
        job_id: str,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Push a task, refusing rather than accepting work it cannot hold.

        Spec 11.3's degradation ladder ends at "reject a new session with
        QUEUE_OVERLOADED; do not accept it and then drop frames".
        """
        if queue not in ALL_QUEUES:
            raise SasttError(f"unknown queue {queue!r}", details={"queues": list(ALL_QUEUES)})
        depth = self.depth(queue)
        if depth >= self.max_depth:
            raise QueueOverloadedError(
                f"queue {queue!r} is at its bound of {self.max_depth}",
                details={"queue": queue, "depth": depth},
            )
        task = Task(
            task_id=f"tsk_{uuid.uuid4().hex[:20]}",
            queue=queue,
            tenant_id=tenant_id,
            job_id=job_id,
            payload=payload or {},
            enqueued_ms=_now_ms(),
        )
        self._client.lpush(self.key(queue), task.to_json())
        return task

    # -- consuming ----------------------------------------------------------- #

    def reserve(self, queues: list[str], consumer: str, *, timeout_seconds: int = 5) -> Task | None:
        """Reserve the next task, highest-priority queue first.

        The task lands in this consumer's processing list, so it survives a crash
        and can be requeued instead of vanishing (spec 3 recovery).
        """
        for queue in queues:
            raw = self._client.rpoplpush(self.key(queue), self.processing_key(queue, consumer))
            if raw:
                return Task.from_json(raw)
        # Nothing ready: block on the first queue so an idle worker does not spin.
        raw = self._client.brpoplpush(
            self.key(queues[0]), self.processing_key(queues[0], consumer), timeout_seconds
        )
        return Task.from_json(raw) if raw else None

    def ack(self, task: Task, consumer: str) -> bool:
        """Remove a finished task from the processing list."""
        removed = self._client.lrem(self.processing_key(task.queue, consumer), 1, task.to_json())
        return bool(removed)

    def retry(self, task: Task, consumer: str, *, max_attempts: int = 3) -> bool:
        """Return a failed task to its queue; ``False`` once attempts run out."""
        self.ack(task, consumer)
        if task.attempts + 1 >= max_attempts:
            self._client.lpush(
                f"{self.namespace}:dead:{task.queue}",
                Task(**{**task.__dict__, "attempts": task.attempts + 1}).to_json(),
            )
            return False
        retried = Task(**{**task.__dict__, "attempts": task.attempts + 1})
        self._client.lpush(self.key(task.queue), retried.to_json())
        return True

    def requeue_stale(self, queue: str, consumer: str) -> int:
        """Return tasks a dead consumer was holding — spec 3 worker restart."""
        processing = self.processing_key(queue, consumer)
        moved = 0
        while True:
            popped = self._client.rpop(processing)
            if not popped:
                break
            # decode_responses=True means a single pop is always a string; the
            # list form of the signature only applies when a count is passed.
            raw = popped if isinstance(popped, str) else str(popped)
            task = Task.from_json(raw)
            if task.age_ms < self.visibility_timeout_ms:
                # Still within its lease: put it back untouched.
                self._client.lpush(processing, raw)
                break
            self._client.lpush(self.key(queue), raw)
            moved += 1
        return moved

    # -- observation (spec 13.1) --------------------------------------------- #

    def depth(self, queue: str) -> int:
        return int(self._client.llen(self.key(queue)))

    def oldest_age_ms(self, queue: str) -> int:
        """``sastt_queue_age_seconds`` needs the age of the oldest waiting task."""
        raw = self._client.lindex(self.key(queue), -1)
        return Task.from_json(raw).age_ms if raw else 0

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            queue: {"depth": self.depth(queue), "oldest_age_ms": self.oldest_age_ms(queue)}
            for queue in ALL_QUEUES
        }

    def purge(self, queue: str) -> int:
        removed = self.depth(queue)
        self._client.delete(self.key(queue))
        return removed


__all__ = [
    "ALL_QUEUES",
    "DEFAULT_URL",
    "QUEUE_ASR_BATCH",
    "QUEUE_ASR_REALTIME",
    "QUEUE_FUSION",
    "QUEUE_GSS_BATCH",
    "QUEUE_SEPARATION_BETA",
    "QUEUE_SEPARATION_TWO_SOURCE",
    "QUEUE_SPEAKER_BATCH",
    "QUEUE_SPEAKER_REALTIME",
    "QueueOverloadedError",
    "RedisTaskQueue",
    "Task",
    "build_client",
]
