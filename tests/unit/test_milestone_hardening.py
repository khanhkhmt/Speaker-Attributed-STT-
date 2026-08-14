"""Executable M3–M5 guardrails that do not pretend to be hardware evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from deploy.benchmark_report import build_report
from deploy.capacity_report import build_capacity_report
from deploy.generate_sbom import build_sbom

from sastt.adapters.fake import (
    FakeAudioDecoder,
    FakeDiarizer,
    FakeOverlapDetector,
    FakeSpeakerEmbedder,
    FakeSpeechRecognizer,
    FakeSpeechSeparator,
    FakeVoiceActivityDetector,
)
from sastt.adapters.fake.scenario import Scenario
from sastt.adapters.speechbrain.sepformer import _as_sources
from sastt.application.fusion import FileConfidenceCalibrator
from sastt.application.offline_pipeline import PipelineAdapters
from sastt.application.streaming_pipeline import StreamingSession
from sastt.config import StreamingConfig
from sastt.domain.errors import ConfigurationError
from sastt.observability import (
    METRIC_AUDIO_SECONDS,
    METRIC_STREAM_BUFFER_SECONDS,
    InMemoryMetrics,
)

pytestmark = pytest.mark.unit


class TestM3BoundedRealtimeMemory:
    def test_ring_buffer_stays_bounded_and_records_ingest_metrics(self, calibrated_config) -> None:
        config = calibrated_config.model_copy(
            update={
                "streaming": StreamingConfig(
                    ring_buffer_seconds=0.2,
                    diarization_window_seconds=0.2,
                    diarization_hop_seconds=0.2,
                    finalize_after_silence_seconds=0.2,
                )
            }
        )
        scenario = Scenario.load(
            Path(__file__).parents[1] / "fixtures" / "s02_two_speaker_overlap.json"
        )
        adapters = PipelineAdapters(
            decoder=FakeAudioDecoder(),
            vad=FakeVoiceActivityDetector(),
            diarizer=FakeDiarizer(scenario.speakers),
            overlap_detector=FakeOverlapDetector(scenario.speakers),
            recognizer=FakeSpeechRecognizer(scenario),
            embedder=FakeSpeakerEmbedder(scenario.speakers),
            separator=FakeSpeechSeparator(scenario.speakers, scenario=scenario),
        )
        metrics = InMemoryMetrics()
        session = StreamingSession(
            session_id="ses_bounded",
            config=config,
            adapters=adapters,
            metrics=metrics,
        )
        session.start()
        silent_frame = b"\0\0" * int(config.audio.canonical_sample_rate * 0.04)
        for _ in range(30):
            session.push_pcm(silent_frame)
        assert session.now_ms == 1200
        assert session._samples.size <= int(config.audio.canonical_sample_rate * 0.2)
        assert metrics.counter_value(METRIC_AUDIO_SECONDS, mode="streaming") == pytest.approx(1.2)
        assert (
            metrics.gauges[(METRIC_STREAM_BUFFER_SECONDS, (("session_mode", "streaming"),))] <= 0.2
        )


class TestM4SepformerContract:
    def test_sepformer_normalizes_time_source_layout(self) -> None:
        value = np.zeros((40, 3), dtype=np.float32)
        assert _as_sources(value, 3).shape == (3, 40)

    def test_sepformer_normalizes_source_time_layout(self) -> None:
        value = np.zeros((3, 40), dtype=np.float32)
        assert _as_sources(value, 3).shape == (3, 40)


class TestM5EvidenceTools:
    def test_calibration_release_maps_only_supplied_raw_evidence(self, tmp_path) -> None:
        path = tmp_path / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "release_id": "cal_2026_08",
                    "components": {"asr": {"raw_key": "asr_word_probability", "a": 4, "b": -2}},
                }
            ),
            encoding="utf-8",
        )
        result = FileConfidenceCalibrator.from_path(path).calibrate({"asr_word_probability": 0.9})
        assert result.status == "calibrated"
        assert result.asr is not None and 0.0 < result.asr < 1.0
        assert result.diarization is None

    def test_invalid_calibration_release_fails_closed(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"release_id": "bad", "components": {}}', encoding="utf-8")
        with pytest.raises(ConfigurationError):
            FileConfidenceCalibrator.from_path(path)

    def test_benchmark_and_capacity_do_not_pass_missing_evidence(self) -> None:
        benchmark = build_report([], release_id="bench_empty")
        capacity = build_capacity_report([])
        assert benchmark["evidence_status"] == "not_evaluated"
        assert benchmark["quality_gate"] == "pending_review"
        assert capacity["overall"] == "pending"
        assert set(capacity["gates"].values()) == {"not_evaluated"}

    def test_reports_aggregate_measured_evidence(self) -> None:
        benchmark = build_report(
            [
                {
                    "reference_text": "xin chao ban",
                    "hypothesis_text": "xin chao ban",
                    "speaker_attributed_correct": True,
                }
            ],
            release_id="bench_1",
        )
        capacity = build_capacity_report(
            [
                {
                    "rtf": 0.3,
                    "provisional_latency_s": 1.0,
                    "attributed_latency_s": 2.0,
                    "gpu_utilization": 0.7,
                }
            ]
        )
        assert benchmark["wer"] == 0.0
        assert capacity["overall"] == "pass"

    def test_sbom_includes_model_inventory(self) -> None:
        sbom = build_sbom(Path(__file__).parents[2] / "model-manifests")
        assert any(item["backend"] == "sepformer_libri3mix" for item in sbom["model_manifests"])
