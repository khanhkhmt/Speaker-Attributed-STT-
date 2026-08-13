"""Load/soak tests — spec 16.1.6.

Excluded from the default run (marker ``load``). The SLOs of spec 3 — E2E RTF
p95 <= 0.50, provisional text p95 <= 2.5 s, attributed label p95 <= 5 s, VRAM
and GPU utilisation below 80 % — are only meaningful on the baseline hardware of
spec 19.2/19.3 with real weights, so they stay unimplemented until Milestone 3
rather than being faked here.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.load


def test_offline_rtf_slo() -> None:
    pytest.skip("requires baseline GPU + real models (spec 3, 19.2); Milestone 3/5")


def test_realtime_latency_slo() -> None:
    pytest.skip("requires baseline GPU + real models (spec 3, 19.3); Milestone 3")


def test_thirty_minute_stream_has_no_memory_leak() -> None:
    pytest.skip("soak test, spec 18 Milestone 3 DoD; needs the real worker image")


def test_worker_restart_mid_job_resumes() -> None:
    pytest.skip("needs the real queue/worker topology (spec 11.3); Milestone 3")
