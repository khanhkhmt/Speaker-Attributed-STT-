"""Shared fixtures for the sastt test suite (spec 16.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sastt.adapters.fake import (
    FakeAudioDecoder,
    FakeDiarizer,
    FakeOverlapDetector,
    FakeSpeakerEmbedder,
    FakeSpeechRecognizer,
    FakeSpeechSeparator,
    FakeVoiceActivityDetector,
    FakeVoiceRegistry,
)
from sastt.adapters.fake.scenario import Scenario
from sastt.application.offline_pipeline import PipelineAdapters
from sastt.config import SasttConfig, load_config
from sastt.domain.events import Clock
from sastt.observability import CallContext, InMemoryMetrics

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class FrozenClock:
    """Deterministic clock (spec 16.1.1: no wall-clock flakiness in unit tests)."""

    def __init__(self, start_ms: int = 1_700_000_000_000, step_ms: int = 1) -> None:
        self.current = start_ms
        self.step = step_ms

    def now_ms(self) -> int:
        value = self.current
        self.current += self.step
        return value


@pytest.fixture
def clock() -> Clock:
    return FrozenClock()


@pytest.fixture
def metrics() -> InMemoryMetrics:
    return InMemoryMetrics()


@pytest.fixture
def ctx(metrics: InMemoryMetrics) -> CallContext:
    return CallContext(stage="test", metrics=metrics, tenant_hash="tenant_test")


@pytest.fixture
def base_config() -> SasttConfig:
    """The shipped default configuration — thresholds still null (spec 12)."""
    return load_config(CONFIG_PATH, environment="development", manifest_dir=None)


@pytest.fixture
def calibrated_config() -> SasttConfig:
    """A configuration with calibrated linking/Voice ID thresholds.

    Production ships them as ``null`` and fails closed; a calibration release
    (spec 21.3) is what makes identity decisions possible, so tests that assert
    linking behaviour must supply one explicitly.
    """
    return load_config(
        CONFIG_PATH,
        environment="development",
        manifest_dir=None,
        overrides={
            "source_linking": {"accept_threshold": 0.55, "ambiguous_margin": 0.10},
            "voice_id": {"accept_threshold": 0.70, "ambiguous_margin": 0.10},
        },
    )


def load_scenario(name: str) -> Scenario:
    return Scenario.load(FIXTURE_DIR / name)


@pytest.fixture
def scenario_factory() -> Any:
    return load_scenario


@pytest.fixture
def adapters_factory() -> Any:
    """``build_adapters`` as a fixture.

    Test modules under ``tests/unit`` cannot ``from conftest import ...``: pytest
    imports every ``conftest.py`` under the same basename, so which one that name
    resolves to depends on collection order. Fixtures are resolved by pytest
    itself and do not have that ambiguity.
    """
    return build_adapters


@pytest.fixture
def pcm_factory() -> Any:
    return scenario_pcm


def build_adapters(
    scenario: Scenario,
    *,
    alternate_order: bool = False,
    fail_first_separation: bool = False,
    registry: FakeVoiceRegistry | None = None,
    minimum_speech_ms: int = 1500,
) -> PipelineAdapters:
    """Wire the fake adapters for one scenario (spec 18 M0 deliverable)."""
    embedder = FakeSpeakerEmbedder(scenario.speakers, minimum_speech_ms=minimum_speech_ms)
    return PipelineAdapters(
        decoder=FakeAudioDecoder(sample_rate=scenario.sample_rate),
        vad=FakeVoiceActivityDetector(),
        diarizer=FakeDiarizer(scenario.speakers),
        overlap_detector=FakeOverlapDetector(scenario.speakers),
        recognizer=FakeSpeechRecognizer(scenario),
        embedder=embedder,
        separator=FakeSpeechSeparator(
            scenario.speakers,
            scenario=scenario,
            alternate_order=alternate_order,
            fail_first_call=fail_first_separation,
        ),
        registry=registry,
    )


def scenario_pcm(scenario: Scenario) -> bytes:
    """Render a scenario to PCM s16le, the format the stream API accepts."""
    samples = scenario.render().samples[0]
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
