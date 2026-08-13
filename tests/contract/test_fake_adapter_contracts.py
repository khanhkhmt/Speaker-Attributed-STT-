"""Adapter contract tests — spec 16.1.2.

Every adapter must satisfy its port protocol and return domain types, without
loading model weights.
"""

from __future__ import annotations

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
from sastt.adapters.persistence import InMemoryEventStore, InMemoryJobStore, InMemoryObjectStore
from sastt.domain.audio import AudioAsset, AudioBuffer, TimeInterval
from sastt.domain.errors import (
    InsufficientSpeechForEmbeddingError,
    SeparationFailedError,
    TenantAccessDeniedError,
    UnsupportedAudioFormatError,
)
from sastt.domain.speakers import (
    DiarizationResult,
    EnrollmentReport,
    OverlapRegion,
    SeparatedBatch,
    SpeakerEmbedding,
    VoiceIdDecision,
)
from sastt.domain.transcript import ASRResult
from sastt.observability import CallContext
from sastt.ports import (
    AudioDecoder,
    Diarizer,
    EventStore,
    JobStore,
    ObjectStore,
    OverlapDetector,
    SpeakerEmbedder,
    SpeechRecognizer,
    SpeechSeparator,
    VoiceActivityDetector,
    VoiceRegistry,
)

pytestmark = pytest.mark.contract

from conftest import load_scenario, scenario_pcm  # noqa: E402  (pytest adds tests/ to sys.path)


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return load_scenario("s02_two_speaker_overlap.json")


@pytest.fixture(scope="module")
def session_audio(scenario: Scenario) -> AudioBuffer:
    return scenario.render()


class TestProtocolConformance:
    def test_every_fake_satisfies_its_port(self, scenario: Scenario) -> None:
        assert isinstance(FakeAudioDecoder(), AudioDecoder)
        assert isinstance(FakeVoiceActivityDetector(), VoiceActivityDetector)
        assert isinstance(FakeDiarizer(scenario.speakers), Diarizer)
        assert isinstance(FakeOverlapDetector(scenario.speakers), OverlapDetector)
        assert isinstance(FakeSpeechSeparator(scenario.speakers), SpeechSeparator)
        assert isinstance(FakeSpeechRecognizer(scenario), SpeechRecognizer)
        assert isinstance(FakeSpeakerEmbedder(scenario.speakers), SpeakerEmbedder)
        assert isinstance(
            FakeVoiceRegistry(embedding_model_version="fake-campplus@1"), VoiceRegistry
        )
        assert isinstance(InMemoryJobStore(), JobStore)
        assert isinstance(InMemoryEventStore(), EventStore)
        assert isinstance(InMemoryObjectStore(), ObjectStore)


class TestDecoder:
    def test_returns_an_asset_with_a_mono_derivative(
        self, scenario: Scenario, ctx: CallContext
    ) -> None:
        asset = FakeAudioDecoder().decode(scenario_pcm(scenario), ctx)
        assert isinstance(asset, AudioAsset)
        assert asset.mono_16k.sample_rate == 16_000
        assert asset.mono_16k.is_mono
        assert len(asset.input_sha256) == 64
        assert asset.channel_map == ("mono",)

    def test_multichannel_input_keeps_its_channels(self, ctx: CallContext) -> None:
        interleaved = np.zeros(1600 * 2, dtype="<i2")
        asset = FakeAudioDecoder(channels=2).decode(interleaved.tobytes(), ctx)
        assert asset.original.num_channels == 2  # spec 5.1.4
        assert asset.is_multichannel is True
        assert asset.mono_16k.num_channels == 1

    def test_broken_payload_maps_to_a_domain_error(self, ctx: CallContext) -> None:
        with pytest.raises(UnsupportedAudioFormatError):
            FakeAudioDecoder().decode(b"\x00", ctx)
        with pytest.raises(UnsupportedAudioFormatError):
            FakeAudioDecoder().decode(b"", ctx)


class TestSpeakerStack:
    def test_diarizer_returns_domain_result(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        result = FakeDiarizer(scenario.speakers).diarize(session_audio, ctx)
        assert isinstance(result, DiarizationResult)
        assert result.estimated_session_speakers == 2
        assert result.regular_tracks and result.exclusive_tracks is not None
        assert all(turn.kind == "regular" for turn in result.regular_tracks)
        assert result.model_version

    def test_overlap_detector_returns_regions(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        regions = FakeOverlapDetector(scenario.speakers).detect(session_audio, ctx)
        assert regions and all(isinstance(region, OverlapRegion) for region in regions)
        assert all(region.interval.duration_ms >= 300 for region in regions)

    def test_vad_returns_intervals(self, session_audio: AudioBuffer, ctx: CallContext) -> None:
        speech = FakeVoiceActivityDetector().detect(session_audio, ctx)
        assert speech and all(isinstance(interval, TimeInterval) for interval in speech)

    def test_separator_returns_waveforms_not_identities(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        crop = session_audio.crop_ms(TimeInterval(12_000, 15_000))
        batch = FakeSpeechSeparator(scenario.speakers).separate(crop, ctx, requested_source_count=2)
        assert isinstance(batch, SeparatedBatch)
        assert batch.sources.shape[0] == 2
        assert len(batch.source_quality) == 2
        assert batch.separator_version

    def test_separator_rejects_unsupported_source_counts(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        crop = session_audio.crop_ms(TimeInterval(12_000, 15_000))
        with pytest.raises(SeparationFailedError):
            FakeSpeechSeparator(scenario.speakers).separate(crop, ctx, requested_source_count=3)

    def test_embedder_returns_normalised_vectors(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        crop = session_audio.crop_ms(TimeInterval(0, 4000))
        embedding = FakeSpeakerEmbedder(scenario.speakers).embed(crop, ctx)
        assert isinstance(embedding, SpeakerEmbedding)
        assert embedding.vector.shape == (192,)
        assert float(np.linalg.norm(embedding.vector)) == pytest.approx(1.0, abs=1e-4)
        assert embedding.model_version

    def test_embedder_refuses_speech_below_the_minimum(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        crop = session_audio.crop_ms(TimeInterval(0, 800))
        with pytest.raises(InsufficientSpeechForEmbeddingError):
            FakeSpeakerEmbedder(scenario.speakers).embed(crop, ctx)


class TestRecognizer:
    def test_returns_absolute_word_timestamps(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        crop = session_audio.crop_ms(TimeInterval(5500, 10_000))
        result = FakeSpeechRecognizer(scenario).transcribe(crop, ctx, language="vi")
        assert isinstance(result, ASRResult)
        assert result.detected_language == "vi"
        assert result.words
        assert all(word.start_ms >= 5000 for word in result.words)
        assert result.language_score is None  # raw score, never a calibrated number


class TestRegistry:
    def _embeddings(self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext) -> list:
        embedder = FakeSpeakerEmbedder(scenario.speakers)
        # Four 4 s clips: >= 3 clips and >= 15 s of speech, per the policy of spec 5.10.
        return [
            embedder.embed(session_audio.crop_ms(TimeInterval(start, start + 4000)), ctx)
            for start in (0, 1000, 5500, 6500)
        ]

    def test_enrollment_returns_a_quality_report(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        registry = FakeVoiceRegistry(embedding_model_version="fake-campplus@1")
        report = registry.enroll(
            "tenant_a",
            "EMP-042",
            self._embeddings(scenario, session_audio, ctx),
            ctx,
            display_name="Nguyễn Văn B",
            consent_ref="consent-1",
        )
        assert isinstance(report, EnrollmentReport)
        assert report.accepted_clips == 4
        assert report.meets_policy is True
        assert report.to_dict()["prototype_count"] == 4

    def test_identify_fails_closed_while_uncalibrated(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        registry = FakeVoiceRegistry(embedding_model_version="fake-campplus@1")
        embeddings = self._embeddings(scenario, session_audio, ctx)
        registry.enroll("tenant_a", "EMP-042", embeddings, ctx, display_name="B")
        decision = registry.identify("tenant_a", embeddings[0], ctx)
        assert isinstance(decision, VoiceIdDecision)
        assert decision.status == "uncalibrated"
        assert decision.registry_speaker_id is None

    def test_templates_are_tenant_scoped(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        registry = FakeVoiceRegistry(embedding_model_version="fake-campplus@1")
        registry.enroll("tenant_a", "EMP-042", self._embeddings(scenario, session_audio, ctx), ctx)
        assert registry.identity_exists("tenant_a", "EMP-042") is True
        assert registry.identity_exists("tenant_b", "EMP-042") is False
        with pytest.raises(TenantAccessDeniedError):
            registry.prototypes_of("tenant_b", "EMP-042")

    def test_deletion_removes_templates(
        self, scenario: Scenario, session_audio: AudioBuffer, ctx: CallContext
    ) -> None:
        registry = FakeVoiceRegistry(embedding_model_version="fake-campplus@1")
        registry.enroll("tenant_a", "EMP-042", self._embeddings(scenario, session_audio, ctx), ctx)
        assert registry.delete_identity("tenant_a", "EMP-042", ctx) is True
        assert registry.identity_exists("tenant_a", "EMP-042") is False
