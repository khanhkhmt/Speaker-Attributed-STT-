"""Spec conformance on real weights — spec 16.1.4.

These tests guard *contract invariants* that the fake adapters cannot exercise,
because every one of them was found broken only once real checkpoints ran:

* the session speaker roster stays within ``max_session_speakers`` (spec 0.1.1),
  and an unidentifiable fragment does not become a person (spec 15);
* every declared container, sample rate and channel count of spec 1.1 survives
  the whole pipeline;
* the multichannel original is preserved beside the mono derivative (spec 5.1.4);
* concurrent speech stays two segments (spec 0.1.7).

They deliberately assert **no accuracy**: how many speakers pyannote actually
resolves is a benchmark question (spec 16.4/16.5, 21.1), and tuning code to make
such a number pass is forbidden (spec 18 rule 5).

Marked ``model``: excluded from the default run, so ordinary CI downloads no
weights and needs no Hub token (spec 16.3). Missing weights skip with a reason
rather than falling back to a fake adapter (spec 18 rule 6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import (
    MAX_SESSION_SPEAKERS,
    SAMPLE_RATE,
    _write_wav,
    overlap_at_start,
    sequential_session,
    transcode,
)
from sastt.adapters.ffmpeg import FfmpegAudioDecoder
from sastt.config import SasttConfig, load_config
from sastt.domain.errors import ModelNotReadyError
from sastt.domain.speakers import IdentityStatus
from sastt.observability import CallContext

pytestmark = pytest.mark.model

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config() -> SasttConfig:
    return load_config(
        REPO_ROOT / "configs" / "default.yaml", environment="development", manifest_dir=None
    )


@pytest.fixture(scope="module")
def linked_config(config: SasttConfig) -> SasttConfig:
    """Thresholds are ``null`` by design and fail closed (spec 12, 18 rule 7).

    Linking needs *some* threshold to run at all, so these are supplied here and
    nowhere else. They are arbitrary test values, **not** a calibration release:
    spec 21.3 leaves the real numbers to the benchmark.
    """
    return config.model_copy(
        update={
            "source_linking": config.source_linking.model_copy(
                update={"accept_threshold": 0.35, "ambiguous_margin": 0.05}
            )
        }
    )


@pytest.fixture(scope="module")
def adapters(config: SasttConfig):
    from sastt.api.http import build_real_engine

    try:
        return build_real_engine(config).adapters_for(None)
    except ModelNotReadyError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def ctx() -> CallContext:
    return CallContext(stage="spec_conformance", timeout_seconds=900.0)


def run_pipeline(linked_config: SasttConfig, adapters, audio: bytes, ctx: CallContext):
    from sastt.application.offline_pipeline import OfflinePipeline

    return OfflinePipeline(linked_config, adapters).run(audio, ctx)


# --------------------------------------------------------------------------- #
# Session speaker roster (FR-004, spec 0.1.1, 15)
# --------------------------------------------------------------------------- #


class TestSpeakerRoster:
    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_roster_never_exceeds_the_session_bound(
        self, linked_config, adapters, real_speakers, ctx, count
    ) -> None:
        """Spec 0.1.1 caps a session at five speakers, provisional ones included.

        Before this was enforced, short fragments at speaker transitions each
        minted their own temporary identity and a two-speaker file reported six.
        """
        if len(real_speakers) < count:
            pytest.skip(f"dataset yielded {len(real_speakers)} speakers, need {count}")
        audio = _write_wav(sequential_session(real_speakers[:count]))
        result = run_pipeline(linked_config, adapters, audio, ctx)

        assert result.succeeded
        identified = {
            segment.session_speaker_id
            for segment in result.segments
            if segment.identity_status is not IdentityStatus.UNKNOWN
        }
        assert len(identified) <= MAX_SESSION_SPEAKERS

    def test_unidentifiable_speech_shares_one_unknown_identity(
        self, linked_config, adapters, real_speakers, ctx
    ) -> None:
        """Spec 15: a source too short to embed is ``Unknown``, not a new person.

        All such fragments collapse into a single session-scoped sink, so they
        cannot inflate the roster however many of them a recording produces.
        """
        audio = _write_wav(sequential_session(real_speakers[:2]))
        result = run_pipeline(linked_config, adapters, audio, ctx)

        unknown_ids = {
            segment.session_speaker_id
            for segment in result.segments
            if segment.identity_status is IdentityStatus.UNKNOWN
        }
        assert len(unknown_ids) <= 1

    def test_unknown_segments_are_labelled_consistently(
        self, linked_config, adapters, real_speakers, ctx
    ) -> None:
        """An ``Unknown`` label must not carry a ``provisional`` status.

        The sink is created during fusion, after ``finalize_unresolved`` has run,
        so it has to enter its terminal state itself (spec 6).
        """
        audio = _write_wav(sequential_session(real_speakers[:2]))
        result = run_pipeline(linked_config, adapters, audio, ctx)

        for segment in result.segments:
            if segment.speaker_label == "Unknown":
                assert segment.identity_status is IdentityStatus.UNKNOWN


# --------------------------------------------------------------------------- #
# Input handling (spec 1.1, 5.1.4, FR-002)
# --------------------------------------------------------------------------- #


class TestInputMatrix:
    @pytest.mark.parametrize(
        ("suffix", "encoder"),
        [
            (".wav", None),
            (".flac", None),
            (".mp3", ["-b:a", "128k"]),
            (".m4a", ["-c:a", "aac", "-b:a", "128k"]),
            (".ogg", ["-c:a", "libopus", "-b:a", "96k"]),
        ],
    )
    def test_declared_containers_run_end_to_end(
        self, linked_config, adapters, two_speaker_session, ctx, suffix, encoder
    ) -> None:
        """Spec 1.1 lists WAV, FLAC, MP3, M4A/AAC and Ogg/Opus as accepted input."""
        audio = (
            two_speaker_session.wav
            if suffix == ".wav"
            else transcode(two_speaker_session.wav, suffix, encoder)
        )
        result = run_pipeline(linked_config, adapters, audio, ctx)

        assert result.succeeded
        assert result.segments

    @pytest.mark.parametrize("rate", [8_000, 16_000, 44_100, 48_000])
    def test_declared_sample_rates_run_end_to_end(
        self, linked_config, adapters, two_speaker_session, ctx, rate
    ) -> None:
        """Spec 1.1 accepts 8-48 kHz; the mono 16 kHz derivative is internal."""
        audio = transcode(two_speaker_session.wav, ".wav", ["-ar", str(rate)])
        result = run_pipeline(linked_config, adapters, audio, ctx)

        assert result.succeeded
        assert result.segments

    @pytest.mark.parametrize("channels", [1, 2, 4, 6, 8])
    def test_multichannel_original_is_preserved(self, real_speakers, ctx, channels) -> None:
        """Spec 5.1.4: the original layout survives; mono 16 kHz is a derivative.

        Downmixing in place would destroy the input the GSS branch needs.
        """
        stack = np.stack([real_speakers[i % len(real_speakers)] for i in range(channels)])
        asset = FfmpegAudioDecoder().decode(_write_wav(stack), ctx)

        assert asset.original.samples.shape[0] == channels
        assert asset.mono_16k.is_mono
        assert asset.mono_16k.sample_rate == SAMPLE_RATE


# --------------------------------------------------------------------------- #
# Concurrency (spec 0.1.7, S04)
# --------------------------------------------------------------------------- #


class TestConcurrency:
    def test_overlap_at_the_first_second_keeps_both_voices(
        self, linked_config, adapters, real_speakers, ctx
    ) -> None:
        """S04: overlap before any clean centroid exists still yields two tracks.

        Spec 0.1.7 forbids collapsing concurrent speech into one segment, so the
        assertion is on concurrency, not on who the speakers turn out to be.
        """
        audio = _write_wav(overlap_at_start(real_speakers[0], real_speakers[1]))
        result = run_pipeline(linked_config, adapters, audio, ctx)

        assert result.succeeded
        overlapping = [segment for segment in result.segments if segment.is_overlap]
        assert len(overlapping) >= 2
        assert overlapping[0].interval.intersects(overlapping[1].interval)
