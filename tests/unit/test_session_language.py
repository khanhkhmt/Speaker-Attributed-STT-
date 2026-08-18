"""The ASR language is a session decision, not a per-crop one — spec 5.5.

Whisper identifies the language from whatever audio it is handed. Handing it a
few hundred milliseconds of a separated overlap source makes identification a
coin flip, and a wrong-language decode is where the model emits memorised
subtitle credits instead of a transcription. These tests pin the contract: one
identification per session, from pooled speech, reused by every later call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sastt.application.offline_pipeline import (
    WARNING_LANGUAGE_UNCERTAIN,
    OfflinePipeline,
    _pool_speech,
)
from sastt.config import ConfigurationError, SasttConfig, load_config
from sastt.domain.audio import AudioBuffer, TimeInterval
from sastt.domain.transcript import ASRResult
from sastt.observability import CallContext

pytestmark = pytest.mark.unit

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
SCENARIO = "s02_two_speaker_overlap.json"


class RecordingRecognizer:
    """Wraps a recognizer and records what every call was asked to decode."""

    def __init__(self, inner: Any, *, detected: str = "vi", probability: float = 0.99) -> None:
        self.inner = inner
        self.detected = detected
        self.probability = probability
        self.languages: list[str | None] = []
        self.detect_calls = 0

    @property
    def model_version(self) -> str:
        return str(self.inner.model_version)

    def detect_language(self, buffer: AudioBuffer, ctx: CallContext) -> tuple[str, float]:
        self.detect_calls += 1
        return self.detected, self.probability

    def transcribe(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        language: str | None = None,
        source_track: int | None = None,
    ) -> ASRResult:
        self.languages.append(language)
        return self.inner.transcribe(buffer, ctx, language=language, source_track=source_track)


def config_with(**detection: Any) -> SasttConfig:
    """Default config with a small identification sample, so short fixtures qualify."""
    settings: dict[str, Any] = {"mode": "auto_once", "sample_seconds": 2, "min_probability": 0.6}
    settings.update(detection)
    return load_config(
        CONFIG_PATH,
        environment="development",
        manifest_dir=None,
        overrides={
            "asr": {"language_detection": settings},
            "source_linking": {"accept_threshold": 0.55, "ambiguous_margin": 0.10},
        },
    )


@pytest.fixture
def run_session(scenario_factory: Any, adapters_factory: Any, pcm_factory: Any, ctx: Any) -> Any:
    """Run one offline job and hand back the pipeline, the recorder and the result."""

    def _run(
        config: SasttConfig, *, probability: float = 0.99, detected: str = "vi"
    ) -> tuple[OfflinePipeline, RecordingRecognizer, Any]:
        scenario = scenario_factory(SCENARIO)
        adapters = adapters_factory(scenario)
        recorder = RecordingRecognizer(
            adapters.recognizer, detected=detected, probability=probability
        )
        adapters.recognizer = recorder
        pipeline = OfflinePipeline(config, adapters)
        result = pipeline.run(pcm_factory(scenario), ctx)
        return pipeline, recorder, result

    return _run


class TestSessionLanguageIsDecidedOnce:
    def test_every_transcribe_call_gets_the_same_language(self, run_session: Any) -> None:
        _pipeline, recorder, _result = run_session(config_with())

        assert recorder.languages, "the scenario should have produced ASR calls"
        assert set(recorder.languages) == {"vi"}

    def test_identification_runs_once_for_the_whole_session(self, run_session: Any) -> None:
        _pipeline, recorder, _result = run_session(config_with())

        assert recorder.detect_calls == 1
        # The overlap scenario alone makes several ASR calls; without pinning,
        # each of them would have identified the language again.
        assert len(recorder.languages) > 1

    def test_a_pinned_job_language_skips_identification(self, run_session: Any) -> None:
        config = load_config(
            CONFIG_PATH,
            environment="development",
            manifest_dir=None,
            overrides={"asr": {"language": "en"}},
        )
        _pipeline, recorder, _result = run_session(config)

        assert recorder.detect_calls == 0
        assert set(recorder.languages) == {"en"}

    def test_per_segment_mode_restores_backend_identification(self, run_session: Any) -> None:
        _pipeline, recorder, _result = run_session(config_with(mode="per_segment"))

        assert recorder.detect_calls == 0
        assert set(recorder.languages) == {None}


class TestUncertainIdentificationFailsOpen:
    def test_a_low_probability_result_is_not_pinned(self, run_session: Any) -> None:
        pipeline, recorder, result = run_session(config_with(min_probability=0.9), probability=0.2)

        assert pipeline.session_language is None
        assert set(recorder.languages) == {None}, "unpinned means the backend still decides"
        assert WARNING_LANGUAGE_UNCERTAIN in result.warnings

    def test_pinning_is_reported_with_its_probability(self, run_session: Any) -> None:
        pipeline, _recorder, _result = run_session(config_with(), probability=0.87)

        assert pipeline.session_language == "vi"
        assert pipeline.language_probability == pytest.approx(0.87)


class TestFinalPassInheritsTheDecision:
    """A streaming final pass must not re-decide and contradict what it already sent."""

    def test_pin_language_is_adopted(self, run_session: Any) -> None:
        pipeline, _recorder, _result = run_session(config_with())
        successor = OfflinePipeline(config_with(), pipeline.adapters)

        successor.pin_language(pipeline.session_language, pipeline.language_probability)

        assert successor.session_language == pipeline.session_language

    def test_pinning_nothing_leaves_the_successor_free(self, run_session: Any) -> None:
        pipeline, _recorder, _result = run_session(config_with())
        successor = OfflinePipeline(config_with(), pipeline.adapters)

        successor.pin_language(None)

        assert successor.session_language is None


class TestPooledSpeech:
    def _buffer(self, seconds: float, sample_rate: int = 16_000) -> AudioBuffer:
        samples = np.linspace(-0.5, 0.5, int(seconds * sample_rate), dtype=np.float32)
        return AudioBuffer(
            samples=samples[np.newaxis, :],
            sample_rate=sample_rate,
            start_sample=0,
            channel_layout=("mono",),
            source_clock_hz=sample_rate,
        )

    def test_silence_between_speech_is_dropped(self) -> None:
        buffer = self._buffer(10.0)
        speech = [TimeInterval(0, 2_000), TimeInterval(8_000, 10_000)]

        pooled = _pool_speech(buffer, speech, target_ms=4_000)

        assert pooled is not None
        assert pooled.duration_ms == 4_000, "the 6 s gap must not be pooled"

    def test_pooling_stops_at_the_target(self) -> None:
        pooled = _pool_speech(self._buffer(60.0), [TimeInterval(0, 60_000)], target_ms=30_000)

        assert pooled is not None
        assert pooled.duration_ms == 30_000

    def test_too_little_speech_defers_the_decision(self) -> None:
        # Below half the target: a streaming window waits rather than deciding
        # the session's language from a couple of seconds of audio.
        assert _pool_speech(self._buffer(10.0), [TimeInterval(0, 1_000)], target_ms=30_000) is None

    def test_no_speech_at_all_defers_the_decision(self) -> None:
        assert _pool_speech(self._buffer(10.0), [], target_ms=1_000) is None


class TestConfigGate:
    def test_fixed_mode_without_a_language_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="requires asr.language"):
            load_config(
                CONFIG_PATH,
                environment="development",
                manifest_dir=None,
                overrides={"asr": {"language_detection": {"mode": "fixed"}}},
            )

    def test_fixed_mode_with_a_language_is_accepted(self) -> None:
        config = load_config(
            CONFIG_PATH,
            environment="development",
            manifest_dir=None,
            overrides={"asr": {"language": "vi", "language_detection": {"mode": "fixed"}}},
        )

        assert config.asr.language == "vi"
