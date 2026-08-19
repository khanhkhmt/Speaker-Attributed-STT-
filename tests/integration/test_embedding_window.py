"""Which slice of a separated source is embedded for linking — spec 5.6, 5.8.

Separation runs on the overlap region padded by ``audio.overlap_context_seconds``
on each side. ``embedding_window`` decides whether identity is judged from the
region alone or from that whole window. These tests pin the mechanics; whether
the wider window links *correctly* is an accuracy question and needs labelled
overlap audio (spec 18 rule 6, 19.1).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from sastt.application.offline_pipeline import OfflinePipeline
from sastt.config import SasttConfig, load_config
from sastt.domain.audio import AudioBuffer, seconds_to_ms
from sastt.observability import CallContext
from sastt.ports.embedding import SpeakerEmbedder

pytestmark = pytest.mark.integration

from conftest import CONFIG_PATH, build_adapters, load_scenario, scenario_pcm  # noqa: E402


class RecordingEmbedder:
    """Wraps the fake embedder and records how much audio each call received."""

    def __init__(self, inner: SpeakerEmbedder) -> None:
        self._inner = inner
        self.separated_durations_ms: list[int] = []

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    def embed(self, buffer: AudioBuffer, ctx: CallContext, **kwargs: Any) -> Any:
        if kwargs.get("origin") == "separated":
            self.separated_durations_ms.append(buffer.duration_ms)
        return self._inner.embed(buffer, ctx, **kwargs)


class RecordingRecognizer:
    """Records the exact span of audio handed to ASR on every call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.intervals: list[tuple[int, int]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def transcribe(self, buffer: AudioBuffer, ctx: CallContext, **kwargs: Any) -> Any:
        self.intervals.append((buffer.start_ms, buffer.end_ms))
        return self._inner.transcribe(buffer, ctx, **kwargs)


def config_with(window: str) -> SasttConfig:
    return load_config(
        CONFIG_PATH,
        environment="development",
        manifest_dir=None,
        overrides={
            "source_linking": {
                "accept_threshold": 0.55,
                "ambiguous_margin": 0.10,
                "embedding_window": window,
            },
            "voice_id": {"accept_threshold": 0.70, "ambiguous_margin": 0.10},
        },
    )


def run_and_record(window: str, ctx: CallContext) -> tuple[RecordingEmbedder, SasttConfig]:
    config = config_with(window)
    scenario = load_scenario("s02_two_speaker_overlap.json")
    adapters = build_adapters(scenario)
    spy = RecordingEmbedder(adapters.embedder)
    result = OfflinePipeline(config, replace(adapters, embedder=spy)).run(
        scenario_pcm(scenario), ctx
    )
    assert result.succeeded
    assert spy.separated_durations_ms, "the scenario must exercise the separation branch"
    return spy, config


def test_shipped_config_keeps_the_narrow_window(ctx: CallContext) -> None:
    """An unvalidated mechanism must be off in the configuration that ships."""
    shipped = load_config(CONFIG_PATH, environment="development", manifest_dir=None)
    assert shipped.source_linking.embedding_window == "owned"


def test_padded_window_embeds_more_audio_than_owned(ctx: CallContext) -> None:
    owned_spy, config = run_and_record("owned", ctx)
    padded_spy, _ = run_and_record("padded", ctx)

    context_ms = seconds_to_ms(config.audio.overlap_context_seconds)
    assert context_ms > 0
    assert len(padded_spy.separated_durations_ms) == len(owned_spy.separated_durations_ms)
    for padded_ms, owned_ms in zip(
        padded_spy.separated_durations_ms, owned_spy.separated_durations_ms, strict=True
    ):
        assert padded_ms > owned_ms
        assert padded_ms <= owned_ms + 2 * context_ms


def test_padded_window_never_widens_what_asr_reads(ctx: CallContext) -> None:
    """Identity may use the padding; ASR never may — those words are the neighbours'.

    This is the invariant the flag actually guarantees. The *final* transcript is
    not invariant: identity decides how words group into utterances and in what
    order segments come out, so a different linking decision reshuffles segment
    boundaries even though every ASR input is byte-for-byte the same. A run on
    20 minutes of real audio showed exactly that — the word sequence stayed 99.4%
    identical while segment boundaries moved.
    """
    scenario = load_scenario("s02_two_speaker_overlap.json")

    asr_inputs = []
    for window in ("owned", "padded"):
        adapters = build_adapters(scenario)
        spy = RecordingRecognizer(adapters.recognizer)
        OfflinePipeline(config_with(window), replace(adapters, recognizer=spy)).run(
            scenario_pcm(scenario), ctx
        )
        asr_inputs.append(spy.intervals)

    assert asr_inputs[0], "the scenario must reach the recogniser"
    assert asr_inputs[0] == asr_inputs[1]
