"""Model smoke tests on real weights and real speech — spec 16.1.4, 18 M1.

Marked ``model``: excluded from the default run, so ordinary CI never downloads
weights and never needs a Hub token (spec 16.3). Each test skips with a reason
when its checkpoint is not staged — it never falls back to a fake adapter and
reports a pass (spec 18 rule 6).

These are smoke tests. They assert that an adapter runs, honours its contract
and behaves sanely on real audio. They do **not** measure accuracy: the DER,
SI-SDRi, WER and Voice ID gates live in the benchmark of spec 16.4/16.5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sastt.config import load_config, load_manifests
from sastt.domain.audio import TimeInterval
from sastt.domain.errors import ModelNotReadyError
from sastt.domain.speakers import SpeakerPrototype, cosine_similarity
from sastt.observability import CallContext

pytestmark = pytest.mark.model

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = load_manifests(REPO_ROOT / "model-manifests")


@pytest.fixture(scope="module")
def config():
    return load_config(
        REPO_ROOT / "configs" / "default.yaml", environment="development", manifest_dir=None
    )


@pytest.fixture
def ctx() -> CallContext:
    return CallContext(stage="model_smoke")


def require(path: str | None, what: str) -> Path:
    resolved = Path(path or "")
    if not path or not resolved.exists():
        pytest.skip(f"{what} weights not staged at {path} (spec 11.2 pre-stages them)")
    return resolved


def version_of(backend: str) -> str:
    manifest = MANIFESTS.get(backend)
    return manifest.release_id if manifest else f"{backend}@unpinned"


# --------------------------------------------------------------------------- #
# Decoder (spec 5.1)
# --------------------------------------------------------------------------- #


class TestDecoder:
    def test_decodes_real_audio_and_derives_mono_16k(self, clean_speech_wav, ctx) -> None:
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder

        asset = FfmpegAudioDecoder().decode(clean_speech_wav, ctx)
        assert asset.mono_16k.sample_rate == 16_000
        assert asset.mono_16k.is_mono
        assert asset.original.num_channels == 1
        assert len(asset.input_sha256) == 64
        assert 4_000 < asset.duration_ms < 6_000
        assert asset.quality.clipping_ratio < 0.01

    def test_rejects_a_non_audio_payload(self, ctx) -> None:
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder
        from sastt.domain.errors import UnsupportedAudioFormatError

        with pytest.raises(UnsupportedAudioFormatError):
            FfmpegAudioDecoder().decode(b"not audio at all" * 100, ctx)


# --------------------------------------------------------------------------- #
# VAD + ASR (spec 5.5)
# --------------------------------------------------------------------------- #


class TestAsr:
    @pytest.fixture(scope="class")
    def recognizer(self):
        from sastt.adapters.faster_whisper import FasterWhisperRecognizer

        path = require("/models/faster-whisper-large-v3-turbo", "ASR")
        try:
            return FasterWhisperRecognizer(
                path, model_version=version_of("faster_whisper"), language="en"
            )
        except ModelNotReadyError as exc:
            pytest.skip(str(exc))

    def test_vad_finds_speech(self, clean_speech_wav, ctx) -> None:
        from sastt.adapters.faster_whisper import SileroVoiceActivityDetector
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder

        asset = FfmpegAudioDecoder().decode(clean_speech_wav, ctx)
        speech = SileroVoiceActivityDetector().detect(asset.mono_16k, ctx)
        assert speech
        assert sum(interval.duration_ms for interval in speech) > 1_000

    def test_transcribes_with_absolute_word_timestamps(
        self, recognizer, clean_speech_wav, ctx
    ) -> None:
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder

        asset = FfmpegAudioDecoder().decode(clean_speech_wav, ctx)
        crop = asset.mono_16k.crop_ms(TimeInterval(1_000, 5_000))
        result = recognizer.transcribe(crop, ctx, language="en")
        assert result.words, "no words recognised from real speech"
        assert all(word.start_ms >= 900 for word in result.words)
        assert all(word.end_ms > word.start_ms for word in result.words)
        assert result.model_version
        # Whisper word probability is a raw score, never a calibrated confidence.
        assert "asr_word_probability" in result.raw_scores
        assert result.language_score is None

    def test_identifies_the_language_of_real_speech(
        self, recognizer, clean_speech_wav, ctx
    ) -> None:
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder

        asset = FfmpegAudioDecoder().decode(clean_speech_wav, ctx)

        language, probability = recognizer.detect_language(asset.mono_16k, ctx)

        assert language == "en", "VoxConverse dev excerpt is English"
        assert 0.0 <= probability <= 1.0

    def test_identification_is_less_certain_on_a_fragment(
        self, recognizer, clean_speech_wav, ctx
    ) -> None:
        """Why the pipeline identifies once per session rather than per crop.

        A few hundred milliseconds is not enough evidence about a language; this
        pins the observation that motivates ``language_detection.auto_once``
        without asserting a particular wrong answer, which would be flaky.
        """
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder

        asset = FfmpegAudioDecoder().decode(clean_speech_wav, ctx)
        whole = asset.mono_16k
        fragment = whole.crop_ms(TimeInterval(whole.start_ms + 1_000, whole.start_ms + 1_300))

        _whole_language, whole_probability = recognizer.detect_language(whole, ctx)
        _fragment_language, fragment_probability = recognizer.detect_language(fragment, ctx)

        assert whole_probability >= fragment_probability


# --------------------------------------------------------------------------- #
# Speaker embedding (spec 5.6)
# --------------------------------------------------------------------------- #


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        from sastt.adapters.speaker3d import CamPlusPlusEmbedder

        path = require("/models/campplus", "CAM++")
        try:
            return CamPlusPlusEmbedder(path, model_version=version_of("3d_speaker_campplus"))
        except ModelNotReadyError as exc:
            pytest.skip(str(exc))

    def _embed(self, embedder, samples: np.ndarray, ctx):
        from sastt.domain.audio import AudioBuffer

        buffer = AudioBuffer(
            samples=np.ascontiguousarray(samples[np.newaxis, :], dtype=np.float32),
            sample_rate=16_000,
            start_sample=0,
            channel_layout=("mono",),
            source_clock_hz=16_000,
        )
        return embedder.embed(buffer, ctx)

    def test_embeddings_are_normalised_and_versioned(
        self, embedder, two_speaker_session, ctx
    ) -> None:
        embedding = self._embed(embedder, two_speaker_session.speaker_a, ctx)
        assert embedding.vector.shape == (192,)
        assert float(np.linalg.norm(embedding.vector)) == pytest.approx(1.0, abs=1e-4)
        assert embedding.model_version == version_of("3d_speaker_campplus")
        assert 0.0 <= embedding.quality <= 1.0

    def test_same_speaker_scores_above_different_speakers(
        self, embedder, two_speaker_session, ctx
    ) -> None:
        """Structural claim only: the ordering must hold, no threshold is asserted."""
        half = two_speaker_session.speaker_a.size // 2
        a_first = self._embed(embedder, two_speaker_session.speaker_a[:half], ctx)
        a_second = self._embed(embedder, two_speaker_session.speaker_a[half:], ctx)
        b = self._embed(embedder, two_speaker_session.speaker_b, ctx)
        same = cosine_similarity(a_first, a_second)
        cross = cosine_similarity(a_first, b)
        assert same > cross, f"same-speaker {same:.3f} did not beat cross-speaker {cross:.3f}"

    def test_short_speech_is_refused(self, embedder, two_speaker_session, ctx) -> None:
        from sastt.domain.errors import InsufficientSpeechForEmbeddingError

        with pytest.raises(InsufficientSpeechForEmbeddingError):
            self._embed(embedder, two_speaker_session.speaker_a[:8_000], ctx)  # 0.5 s


# --------------------------------------------------------------------------- #
# Separation + linking (spec 5.4, 5.8)
# --------------------------------------------------------------------------- #


class TestSeparationAndLinking:
    @pytest.fixture(scope="class")
    def separator(self):
        from sastt.adapters.clearvoice import MossFormer2Separator

        path = require("/models/mossformer2-ss-16k", "MossFormer2")
        try:
            return MossFormer2Separator(path, separator_version=version_of("mossformer2_ss_16k"))
        except ModelNotReadyError as exc:
            pytest.skip(str(exc))

    @staticmethod
    def _buffer(samples: np.ndarray, start_ms: int = 0):
        from sastt.domain.audio import AudioBuffer, ms_to_samples

        return AudioBuffer(
            samples=np.ascontiguousarray(samples[np.newaxis, :], dtype=np.float32),
            sample_rate=16_000,
            start_sample=ms_to_samples(start_ms, 16_000),
            channel_layout=("mono",),
            source_clock_hz=16_000,
        )

    @staticmethod
    def _si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        scale = float(np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-9))
        target = scale * reference
        noise = estimate - target
        return float(10 * np.log10((np.dot(target, target) + 1e-9) / (np.dot(noise, noise) + 1e-9)))

    def test_separates_two_real_speakers(self, separator, two_speaker_session, ctx) -> None:
        a, b = two_speaker_session.speaker_a, two_speaker_session.speaker_b
        n = min(a.size, b.size)
        mixture = (a[:n] + b[:n]) / 2

        batch = separator.separate(self._buffer(mixture), ctx, requested_source_count=2)
        assert batch.source_count == 2
        assert batch.sample_rate == 16_000
        assert batch.separator_version
        # A fixed two-source model cannot count sources (spec 5.3).
        assert batch.estimated_source_count is None
        assert len(batch.source_quality) == 2
        for quality in batch.source_quality:
            assert quality.energy_ratio is not None
            assert quality.leakage_similarity is not None

        best = [
            max(self._si_sdr(batch.sources[i][:n], reference) for i in range(2))
            for reference in (a[:n], b[:n])
        ]
        baseline = [self._si_sdr(mixture, a[:n]), self._si_sdr(mixture, b[:n])]
        assert all(best[i] > baseline[i] for i in range(2)), (
            f"no SI-SDR improvement: {best} vs mixture {baseline}"
        )

    def test_linking_survives_a_source_order_swap(
        self, separator, two_speaker_session, ctx
    ) -> None:
        """S03 on real audio: the separator's source order carries no identity."""
        from sastt.adapters.speaker3d import CamPlusPlusEmbedder
        from sastt.application.source_linking import link_sources
        from sastt.config import SourceLinkingConfig

        embedder_path = require("/models/campplus", "CAM++")
        embedder = CamPlusPlusEmbedder(
            embedder_path, model_version=version_of("3d_speaker_campplus")
        )

        a, b = two_speaker_session.speaker_a, two_speaker_session.speaker_b
        n = min(a.size, b.size)
        prototypes = []
        for key, clean in (("spk_a", a), ("spk_b", b)):
            embedding = embedder.embed(self._buffer(clean), ctx)
            prototypes.append(SpeakerPrototype.from_embedding(key, embedding))

        batch = separator.separate(self._buffer((a[:n] + b[:n]) / 2), ctx, requested_source_count=2)
        embeddings = [
            embedder.embed(self._buffer(batch.sources[i]), ctx) for i in range(batch.source_count)
        ]
        thresholds = SourceLinkingConfig(accept_threshold=0.35, ambiguous_margin=0.05)

        forward = link_sources(embeddings, prototypes, thresholds).mapping()
        reversed_ = link_sources(list(reversed(embeddings)), prototypes, thresholds).mapping()
        assert forward, "real separated sources did not link to any speaker"
        # Whatever track a speaker lands on, the identity must follow the voice.
        assert set(forward.values()) == set(reversed_.values())
        assert len(set(forward.values())) == len(forward), "two sources took one identity"


# --------------------------------------------------------------------------- #
# Diarization / OSD (spec 5.2) — gated checkpoints
# --------------------------------------------------------------------------- #


class TestDiarization:
    def test_diarizer_finds_two_speakers(self, config, two_speaker_session, ctx) -> None:
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder
        from sastt.adapters.pyannote import PyannoteDiarizer

        path = require(config.diarization.model_path, "pyannote diarization")
        try:
            diarizer = PyannoteDiarizer(path, model_version=version_of("pyannote-community-1"))
        except ModelNotReadyError as exc:
            pytest.skip(str(exc))

        asset = FfmpegAudioDecoder().decode(two_speaker_session.wav, ctx)
        result = diarizer.diarize(asset.mono_16k, ctx, min_speakers=1, max_speakers=5)
        assert result.regular_tracks
        assert 1 <= result.estimated_session_speakers <= 5
        assert result.overlap_regions, "the overlapped middle was not detected"

    def test_overlap_detector_runs(self, config, two_speaker_session, ctx) -> None:
        from sastt.adapters.ffmpeg import FfmpegAudioDecoder
        from sastt.adapters.pyannote import PyannoteOverlapDetector

        path = require(config.overlap_detection.model_path, "pyannote segmentation-3.0")
        try:
            detector = PyannoteOverlapDetector(
                path, model_version=version_of("pyannote_segmentation_3.0")
            )
        except ModelNotReadyError as exc:
            pytest.skip(str(exc))

        asset = FfmpegAudioDecoder().decode(two_speaker_session.wav, ctx)
        regions = detector.detect(asset.mono_16k, ctx)
        assert all(region.interval.duration_ms >= 300 for region in regions)


# --------------------------------------------------------------------------- #
# End-to-end (spec 18 Milestone 1 DoD)
# --------------------------------------------------------------------------- #


class TestOfflineEndToEnd:
    def test_real_models_produce_two_concurrent_overlap_segments(
        self, config, two_speaker_session, ctx
    ) -> None:
        """M1 DoD: one real file end to end, non-overlap plus two concurrent segments."""
        from sastt.api.http import build_real_engine

        try:
            engine = build_real_engine(config)
        except ModelNotReadyError as exc:
            pytest.skip(str(exc))

        from sastt.application.offline_pipeline import OfflinePipeline

        calibrated = config.model_copy(
            update={
                "source_linking": config.source_linking.model_copy(
                    update={"accept_threshold": 0.35, "ambiguous_margin": 0.05}
                )
            }
        )
        result = OfflinePipeline(calibrated, engine.adapters_for(None)).run(
            two_speaker_session.wav, ctx
        )
        assert result.succeeded
        assert result.segments
        overlapping = [segment for segment in result.segments if segment.is_overlap]
        assert len(overlapping) >= 2, "the concurrent speakers were collapsed"
        assert overlapping[0].interval.intersects(overlapping[1].interval)
        assert len({segment.session_speaker_id for segment in overlapping}) == 2
        assert all(segment.confidences.status == "uncalibrated" for segment in result.segments)
