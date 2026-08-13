"""Metrics, tracing and call context rules — spec 9, 13."""

from __future__ import annotations

import pytest

from sastt.domain.errors import SasttError
from sastt.observability import (
    METRIC_SOURCE_LINK_UNKNOWN,
    METRIC_STAGE_DURATION,
    CallContext,
    CancellationToken,
    CancelledError,
    InMemoryMetrics,
    NullMetrics,
    tenant_hash,
)

pytestmark = pytest.mark.unit


class TestMetricLabels:
    def test_speaker_name_is_not_allowed_as_a_label(self) -> None:
        metrics = InMemoryMetrics()
        with pytest.raises(SasttError):
            metrics.increment(METRIC_SOURCE_LINK_UNKNOWN, speaker_name="Nguyễn Văn B")

    def test_registry_id_and_text_are_not_allowed(self) -> None:
        metrics = InMemoryMetrics()
        for label in ({"registry_speaker_id": "EMP-042"}, {"text": "xin chào"}):
            with pytest.raises(SasttError):
                metrics.observe(METRIC_STAGE_DURATION, 1.0, **label)

    def test_allowed_labels_are_recorded(self) -> None:
        metrics = InMemoryMetrics()
        metrics.increment(METRIC_SOURCE_LINK_UNKNOWN, reason="low_margin")
        assert metrics.counter_value(METRIC_SOURCE_LINK_UNKNOWN, reason="low_margin") == 1.0

    def test_null_sink_accepts_everything(self) -> None:
        sink = NullMetrics()
        sink.increment("x")
        sink.observe("x", 1.0)
        sink.gauge("x", 1.0)


class TestCallContext:
    def test_child_inherits_context_and_cancellation(self) -> None:
        parent = CallContext(stage="pipeline", session_id="ses_1", tenant_hash="abc")
        child = parent.child("asr", timeout_seconds=5)
        assert child.session_id == "ses_1"
        assert child.tenant_hash == "abc"
        assert child.timeout_seconds == 5
        parent.cancellation.cancel()
        with pytest.raises(CancelledError):
            child.check()

    def test_cancellation_token_is_explicit(self) -> None:
        token = CancellationToken()
        assert token.is_cancelled is False
        token.raise_if_cancelled()
        token.cancel()
        assert token.is_cancelled is True

    def test_trace_fields_carry_no_payload(self) -> None:
        ctx = CallContext(stage="asr", session_id="ses_1", tenant_hash=tenant_hash("tenant-a"))
        fields = ctx.trace_fields()
        assert set(fields) == {
            "stage",
            "tenant_hash",
            "session_id",
            "job_id",
            "model_release_id",
            "audio_duration_ms",
        }
        assert fields["tenant_hash"] != "tenant-a"


class TestTenantHash:
    def test_is_stable_and_pseudonymous(self) -> None:
        assert tenant_hash("tenant-a") == tenant_hash("tenant-a")
        assert tenant_hash("tenant-a") != tenant_hash("tenant-b")
        assert "tenant-a" not in tenant_hash("tenant-a")
