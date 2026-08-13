"""Metrics, tracing context and call contracts — spec 9, 13.

Every model call carries a :class:`CallContext` (timeout, cancellation, metrics
context) as required by spec 9. Nothing here may record raw audio, embeddings or
full transcripts (spec 10.3, 13.2), and no metric label may carry a speaker name,
raw text or a registry ID (spec 13.1).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol

from sastt.domain.errors import SasttError

# --- Metric names (spec 13.1) ---------------------------------------------- #
METRIC_AUDIO_SECONDS = "sastt_audio_seconds_total"
METRIC_OVERLAP_SECONDS = "sastt_overlap_seconds_total"
METRIC_STAGE_DURATION = "sastt_stage_duration_seconds"
METRIC_STAGE_RTF = "sastt_stage_rtf"
METRIC_QUEUE_AGE = "sastt_queue_age_seconds"
METRIC_GPU_VRAM_BYTES = "sastt_gpu_vram_bytes"
METRIC_SPEAKER_COUNT_ESTIMATE = "sastt_speaker_count_estimate"
METRIC_SOURCE_LINK_UNKNOWN = "sastt_source_link_unknown_total"
METRIC_VOICE_ID_ACCEPT = "sastt_voice_id_accept_total"
METRIC_VOICE_ID_REJECT = "sastt_voice_id_reject_total"
METRIC_VOICE_ID_AMBIGUOUS = "sastt_voice_id_ambiguous_total"
METRIC_REVISION = "sastt_revision_total"
METRIC_DEGRADED_SESSION = "sastt_degraded_session_total"
METRIC_MODEL_ERROR = "sastt_model_error_total"

#: Label keys that MUST NOT appear on a metric (spec 13.1).
FORBIDDEN_METRIC_LABELS: frozenset[str] = frozenset(
    {"speaker_name", "text", "transcript", "registry_speaker_id", "external_id", "embedding"}
)


class MetricsSink(Protocol):
    """Minimal metrics surface; the real exporter lives in the worker images."""

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...

    def gauge(self, name: str, value: float, **labels: str) -> None: ...


def _check_labels(name: str, labels: dict[str, str]) -> None:
    forbidden = FORBIDDEN_METRIC_LABELS.intersection(labels)
    if forbidden:
        raise SasttError(
            f"metric {name!r} carries forbidden labels {sorted(forbidden)} (spec 13.1)"
        )


@dataclass
class InMemoryMetrics:
    """Collector used by tests and local runs."""

    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    observations: list[tuple[str, float, dict[str, str]]] = field(default_factory=list)
    gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        _check_labels(name, labels)
        key = (name, tuple(sorted(labels.items())))
        self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        _check_labels(name, labels)
        self.observations.append((name, value, dict(labels)))

    def gauge(self, name: str, value: float, **labels: str) -> None:
        _check_labels(name, labels)
        self.gauges[(name, tuple(sorted(labels.items())))] = value

    def counter_value(self, name: str, **labels: str) -> float:
        return self.counters.get((name, tuple(sorted(labels.items()))), 0.0)


class NullMetrics:
    """No-op sink."""

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        return None

    def observe(self, name: str, value: float, **labels: str) -> None:
        return None

    def gauge(self, name: str, value: float, **labels: str) -> None:
        return None


def tenant_hash(tenant_id: str) -> str:
    """Pseudonymous tenant identifier for traces and logs (spec 13.2)."""
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]


class CancelledError(SasttError):
    """Raised when a cancelled call is still being executed."""


class CancellationToken:
    """Cooperative cancellation for model calls (spec 9)."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError("operation cancelled")


@dataclass
class CallContext:
    """Timeout, cancellation and metrics context handed to every port call (spec 9)."""

    stage: str
    timeout_seconds: float = 60.0
    tenant_hash: str | None = None
    session_id: str | None = None
    job_id: str | None = None
    model_release_id: str | None = None
    audio_duration_ms: int | None = None
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    metrics: MetricsSink = field(default_factory=NullMetrics)
    started_monotonic: float = field(default_factory=time.monotonic)

    @property
    def deadline_exceeded(self) -> bool:
        return (time.monotonic() - self.started_monotonic) > self.timeout_seconds

    def child(self, stage: str, *, timeout_seconds: float | None = None) -> CallContext:
        return CallContext(
            stage=stage,
            timeout_seconds=timeout_seconds
            if timeout_seconds is not None
            else self.timeout_seconds,
            tenant_hash=self.tenant_hash,
            session_id=self.session_id,
            job_id=self.job_id,
            model_release_id=self.model_release_id,
            audio_duration_ms=self.audio_duration_ms,
            cancellation=self.cancellation,
            metrics=self.metrics,
        )

    def check(self) -> None:
        """Cooperative cancellation/timeout checkpoint."""
        self.cancellation.raise_if_cancelled()

    def trace_fields(self) -> dict[str, object]:
        """Trace attributes of spec 13.2 — never raw audio, embeddings or text."""
        return {
            "stage": self.stage,
            "tenant_hash": self.tenant_hash,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "model_release_id": self.model_release_id,
            "audio_duration_ms": self.audio_duration_ms,
        }


__all__ = [
    "FORBIDDEN_METRIC_LABELS",
    "METRIC_AUDIO_SECONDS",
    "METRIC_DEGRADED_SESSION",
    "METRIC_GPU_VRAM_BYTES",
    "METRIC_MODEL_ERROR",
    "METRIC_OVERLAP_SECONDS",
    "METRIC_QUEUE_AGE",
    "METRIC_REVISION",
    "METRIC_SOURCE_LINK_UNKNOWN",
    "METRIC_SPEAKER_COUNT_ESTIMATE",
    "METRIC_STAGE_DURATION",
    "METRIC_STAGE_RTF",
    "METRIC_VOICE_ID_ACCEPT",
    "METRIC_VOICE_ID_AMBIGUOUS",
    "METRIC_VOICE_ID_REJECT",
    "CallContext",
    "CancellationToken",
    "CancelledError",
    "InMemoryMetrics",
    "MetricsSink",
    "NullMetrics",
    "tenant_hash",
]
