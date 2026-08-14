"""Prometheus exposition — spec 13.1."""

from __future__ import annotations

from sastt.observability import METRIC_AUDIO_SECONDS, PrometheusMetrics


def test_prometheus_metrics_exports_counter_and_escapes_labels() -> None:
    metrics = PrometheusMetrics()
    metrics.increment(METRIC_AUDIO_SECONDS, 1.5, mode='batch"test')

    rendered = metrics.render()

    assert "# TYPE sastt_audio_seconds_total counter" in rendered
    assert 'mode="batch\\"test"' in rendered
    assert "sastt_audio_seconds_total{" in rendered
    assert "1.5" in rendered
