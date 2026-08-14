"""Batch transcription worker — spec 11.1, 11.3.

Runs as its own process so a GPU stage never blocks the API event loop, which is
the whole point of the topology in spec 11.1. It reserves a task, walks the job
through the states of spec 8.1, writes the canonical segments to PostgreSQL and
acknowledges.

Failure handling follows spec 3 and 15: an unacknowledged task is retried, and a
task that has exhausted its attempts is parked on a dead list and its job marked
FAILED rather than left QUEUED forever.

Usage:
    python -m sastt.workers.offline_worker --queues speaker.batch asr.batch
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from sastt.adapters.fake.scenario import Scenario

# Imported from the submodules, not the packages: the package __getattr__ that
# keeps psycopg/redis optional hides the real types from the type checker.
from sastt.adapters.persistence.postgres import PostgresJobStore, build_pool
from sastt.adapters.queue.redis_queue import (
    QUEUE_ASR_BATCH,
    QUEUE_SPEAKER_BATCH,
    RedisTaskQueue,
    Task,
    build_client,
)
from sastt.adapters.storage import S3ObjectStore
from sastt.config import load_config
from sastt.domain.errors import SasttError
from sastt.domain.events import JobState
from sastt.observability import CallContext, InMemoryMetrics
from sastt.ports.storage import ObjectStore

LOG = logging.getLogger("sastt.worker")

# Absolute, because a worker is started from whatever directory the supervisor
# happens to use; a relative default only works when cwd is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
MANIFEST_DIR = REPO_ROOT / "model-manifests"


class OfflineWorker:
    """One consumer of the batch queues."""

    def __init__(
        self,
        *,
        queue: RedisTaskQueue,
        jobs: PostgresJobStore,
        queues: list[str],
        consumer: str,
        objects: ObjectStore,
        engine_name: str = "fake",
        config_path: str | None = None,
    ) -> None:
        self.queue = queue
        self.jobs = jobs
        self.queues = queues
        self.consumer = consumer
        self.objects = objects
        self.engine_name = engine_name
        self.config_path = config_path
        self.metrics = InMemoryMetrics()
        self._running = True
        self._engine: Any = None
        self._config: Any = None

    def stop(self, *_: object) -> None:
        """SIGTERM: finish the task in hand, then exit (spec 3 recovery)."""
        LOG.info("shutdown requested; finishing current task")
        self._running = False

    def adapters(self, scenario: Scenario | None = None) -> Any:
        """Load a real model bundle once; fake adapters retain their scenario."""
        if self._engine is None:
            from sastt.api.http import build_fake_engine, build_real_engine

            self._config = load_config(
                self.config_path or DEFAULT_CONFIG,
                environment="development",
                manifest_dir=MANIFEST_DIR,
                # Demo-only calibration-shaped override; production inference
                # keeps null thresholds and fails closed until calibrated.
                overrides={"source_linking": {"accept_threshold": 0.55, "ambiguous_margin": 0.10}},
            )
            builder = build_fake_engine if self.engine_name == "fake" else build_real_engine
            self._engine = builder(self._config)
        return self._engine.adapters_for(scenario)

    def run_forever(self) -> int:
        # A previous incarnation of this consumer may have died holding tasks.
        for queue in self.queues:
            recovered = self.queue.requeue_stale(queue, self.consumer)
            if recovered:
                LOG.warning("requeued %d stale task(s) from %s", recovered, queue)

        LOG.info("worker %s consuming %s", self.consumer, ", ".join(self.queues))
        while self._running:
            task = self.queue.reserve(self.queues, self.consumer, timeout_seconds=2)
            if task is None:
                continue
            started = time.monotonic()
            try:
                self.handle(task)
            except Exception as exc:  # noqa: BLE001 - a task must never kill the worker
                LOG.exception("task %s failed: %s", task.task_id, exc)
                if not self.queue.retry(task, self.consumer):
                    self._fail_job(task, str(exc))
            else:
                self.queue.ack(task, self.consumer)
                LOG.info(
                    "task %s done in %.2fs (queue age %.1fs)",
                    task.task_id,
                    time.monotonic() - started,
                    task.age_ms / 1000,
                )
        return 0

    def handle(self, task: Task) -> None:
        """Run one job to SUCCEEDED, or DEGRADED_SUCCEEDED when a stage degraded."""
        from sastt.application.fusion import load_confidence_calibrator
        from sastt.application.offline_pipeline import OfflinePipeline

        audio = self._load_audio(task)
        scenario_name = task.payload.get("scenario")
        scenario = (
            Scenario.load(REPO_ROOT / "tests" / "fixtures" / f"{scenario_name}.json")
            if scenario_name
            else None
        )
        ctx = CallContext(
            stage="offline_job",
            timeout_seconds=float(task.payload.get("timeout_seconds", 3600)),
            job_id=task.job_id,
            metrics=self.metrics,
        )
        language = task.payload.get("language")
        if language is not None and not isinstance(language, str):
            raise SasttError("task language must be a string or null")
        # adapters() is what populates self._config, so it has to run first.
        adapters = self.adapters(scenario)
        job_config = self._config.model_copy(
            update={"asr": self._config.asr.model_copy(update={"language": language})}
        )

        def update_stage(state: JobState) -> None:
            self.jobs.update_state(task.tenant_id, task.job_id, state)

        result = OfflinePipeline(
            job_config,
            adapters,
            calibrator=load_confidence_calibrator(job_config.confidence.calibration_path),
        ).run(
            audio,
            ctx,
            session_id=task.job_id,
            on_stage=update_stage,
        )

        segments = [segment.to_public_dict() for segment in result.segments]
        self.jobs.save_result(
            task.tenant_id,
            task.job_id,
            segments,
            task.job_id,
            warnings=result.warnings,
            degraded=result.degraded,
        )
        self.jobs.update_state(
            task.tenant_id,
            task.job_id,
            JobState.DEGRADED_SUCCEEDED if result.degraded else JobState.SUCCEEDED,
        )

    def _load_audio(self, task: Task) -> bytes:
        """Audio comes from tenant-scoped object storage, never a local path."""
        key = task.payload.get("audio_key")
        if not isinstance(key, str) or not key:
            raise SasttError("task has no audio_key", details={"task_id": task.task_id})
        return self.objects.get(task.tenant_id, key)

    def _fail_job(self, task: Task, message: str) -> None:
        try:
            error_code = "SEPARATION_FAILED" if "separation" in message else "MODEL_NOT_READY"
            self.jobs.set_error(task.tenant_id, task.job_id, error_code)
            self.jobs.update_state(task.tenant_id, task.job_id, JobState.FAILED)
        except Exception:  # noqa: BLE001 - the job may already be gone
            LOG.exception("could not mark job %s failed", task.job_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queues", nargs="+", default=[QUEUE_SPEAKER_BATCH, QUEUE_ASR_BATCH])
    parser.add_argument("--engine", default=os.environ.get("SASTT_ENGINE", "fake"))
    parser.add_argument("--config", default=os.environ.get("SASTT_CONFIG"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL"))
    parser.add_argument("--once", action="store_true", help="handle one task and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    worker = OfflineWorker(
        queue=RedisTaskQueue(build_client(args.redis_url)),
        jobs=PostgresJobStore(build_pool(args.database_url)),
        queues=list(args.queues),
        consumer=f"{socket.gethostname()}:{os.getpid()}",
        objects=S3ObjectStore.from_environment(),
        engine_name=args.engine,
        config_path=args.config,
    )
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)

    if args.once:
        task = worker.queue.reserve(worker.queues, worker.consumer, timeout_seconds=5)
        if task is None:
            LOG.info("no task available")
            return 0
        worker.handle(task)
        worker.queue.ack(task, worker.consumer)
        return 0
    return worker.run_forever()


if __name__ == "__main__":
    sys.exit(main())
