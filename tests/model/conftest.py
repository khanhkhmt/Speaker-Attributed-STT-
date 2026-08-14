"""Fixtures for the model smoke tests — spec 16.1.4.

Real speech comes from VoxConverse dev (public, multi-speaker, with reference
speaker timestamps). Two single-speaker excerpts are cut from one recording and
arranged into a session with a clean part, a genuinely overlapped part and a
second clean part, so the pipeline can be exercised on real audio with a known
*structure*.

This is a smoke fixture, not a benchmark corpus: spec 16.4 requires 10–20
annotated hours before any accuracy claim, and nothing here measures accuracy.

Everything caches under ``/models/_testdata`` and nothing is committed to Git
(spec 18 rule 4). If the dataset or the weights are unavailable, tests skip with
a reason rather than falling back to the fake adapters (spec 18 rule 6).
"""

from __future__ import annotations

import io
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

TESTDATA_DIR = Path("/models/_testdata")
DATASET = "diarizers-community/voxconverse"
DATASET_FILE = "data/dev-00000-of-00005.parquet"
SAMPLE_RATE = 16_000
EXCERPT_SECONDS = 5.0
#: Session bound of spec 0.1.1 / config ``product.max_session_speakers``.
MAX_SESSION_SPEAKERS = 5


@dataclass(frozen=True)
class TwoSpeakerFixture:
    """A synthetic session built from two real speakers."""

    wav: bytes
    speaker_a: np.ndarray
    speaker_b: np.ndarray
    clean_a_ms: tuple[int, int]
    overlap_ms: tuple[int, int]
    clean_b_ms: tuple[int, int]
    duration_ms: int


def _write_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """PCM16 WAV. ``samples`` is ``[n]`` for mono or ``[channels, n]`` otherwise."""
    if samples.ndim == 1:
        channels = 1
        interleaved = samples
    else:
        channels = samples.shape[0]
        interleaved = samples.T.reshape(-1)
    pcm = (np.clip(interleaved, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    block = channels * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * block, block, 16)
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm


def transcode(wav: bytes, suffix: str, extra: list[str] | None = None) -> bytes:
    """Re-encode WAV bytes into another container with ffmpeg — spec 1.1.

    ffmpeg cannot write most containers to a pipe, so this goes through a
    temporary file. Skips rather than fails when the codec is unavailable: the
    build's ffmpeg, not the product, would be at fault.
    """
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / f"clip{suffix}"
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                *(extra or []),
                str(destination),
            ],
            input=wav,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            pytest.skip(f"ffmpeg cannot produce {suffix}: {proc.stderr.decode()[:200]}")
        return destination.read_bytes()


@pytest.fixture(scope="session")
def two_speaker_session() -> TwoSpeakerFixture:
    """Clean A · A+B overlap · clean B, cut from real VoxConverse speech."""
    try:
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional test dependency
        pytest.skip(f"missing test dependency: {exc}")

    try:
        parquet = hf_hub_download(
            DATASET, DATASET_FILE, repo_type="dataset", cache_dir=str(TESTDATA_DIR / "hf")
        )
    except Exception as exc:  # noqa: BLE001 - offline CI must skip, not fail
        pytest.skip(f"VoxConverse sample unavailable ({type(exc).__name__}); no network?")

    row = pq.read_table(parquet).to_pandas().iloc[0]
    audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
    if rate != SAMPLE_RATE:
        pytest.skip(f"unexpected sample rate {rate}")

    per_speaker: dict[str, list[tuple[float, float]]] = {}
    for start, end, speaker in zip(
        row["timestamps_start"], row["timestamps_end"], row["speakers"], strict=False
    ):
        if end - start >= EXCERPT_SECONDS:
            per_speaker.setdefault(str(speaker), []).append((float(start), float(end)))
    speakers = [name for name, spans in per_speaker.items() if spans][:2]
    if len(speakers) < 2:
        pytest.skip("recording has fewer than two speakers with long enough turns")

    span = int(EXCERPT_SECONDS * SAMPLE_RATE)
    excerpts = []
    for name in speakers:
        start = int(per_speaker[name][0][0] * SAMPLE_RATE)
        clip = audio[start : start + span].astype(np.float32)
        peak = float(np.abs(clip).max()) or 1.0
        excerpts.append(clip / peak * 0.7)
    speaker_a, speaker_b = excerpts

    # clean A | A+B overlap | clean B — the layout of scenario S02.
    half = span // 2
    timeline = np.zeros(span + half + span, dtype=np.float32)
    timeline[:span] += speaker_a
    timeline[half : half + span] += speaker_b
    timeline /= max(1.0, float(np.abs(timeline).max()))

    ms = lambda samples: int(samples * 1000 / SAMPLE_RATE)  # noqa: E731
    return TwoSpeakerFixture(
        wav=_write_wav(timeline),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        clean_a_ms=(0, ms(half)),
        overlap_ms=(ms(half), ms(span)),
        clean_b_ms=(ms(span), ms(span + half)),
        duration_ms=ms(timeline.size),
    )


@pytest.fixture(scope="session")
def clean_speech_wav(two_speaker_session: TwoSpeakerFixture) -> bytes:
    return _write_wav(two_speaker_session.speaker_a)


@pytest.fixture(scope="session")
def real_speakers() -> list[np.ndarray]:
    """Up to five distinct real speakers, one excerpt each — FR-004, spec 0.1.1.

    Walks several VoxConverse recordings because a single one rarely has five
    speakers with long enough turns.
    """
    try:
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional test dependency
        pytest.skip(f"missing test dependency: {exc}")

    try:
        parquet = hf_hub_download(
            DATASET, DATASET_FILE, repo_type="dataset", cache_dir=str(TESTDATA_DIR / "hf")
        )
    except Exception as exc:  # noqa: BLE001 - offline CI must skip, not fail
        pytest.skip(f"VoxConverse sample unavailable ({type(exc).__name__}); no network?")

    span = int(EXCERPT_SECONDS * SAMPLE_RATE)
    collected: list[np.ndarray] = []
    for _, row in pq.read_table(parquet).to_pandas().iterrows():
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if rate != SAMPLE_RATE:
            continue
        per_speaker: dict[str, list[tuple[float, float]]] = {}
        for start, end, speaker in zip(
            row["timestamps_start"], row["timestamps_end"], row["speakers"], strict=False
        ):
            if end - start >= EXCERPT_SECONDS:
                per_speaker.setdefault(str(speaker), []).append((float(start), float(end)))
        for spans in per_speaker.values():
            start = int(spans[0][0] * SAMPLE_RATE)
            clip = audio[start : start + span].astype(np.float32)
            if clip.size < span:
                continue
            peak = float(np.abs(clip).max()) or 1.0
            collected.append(clip / peak * 0.7)
            if len(collected) >= MAX_SESSION_SPEAKERS:
                return collected
    if len(collected) < 2:
        pytest.skip("could not cut two distinct speakers from the dataset")
    return collected


def sequential_session(speakers: list[np.ndarray], gap_seconds: float = 0.5) -> np.ndarray:
    """Speakers one after another with a silent gap — the S01 layout, no overlap."""
    gap = np.zeros(int(gap_seconds * SAMPLE_RATE), dtype=np.float32)
    parts: list[np.ndarray] = []
    for clip in speakers:
        parts.extend((clip, gap))
    timeline = np.concatenate(parts)
    return timeline / max(1.0, float(np.abs(timeline).max()))


def overlap_at_start(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Both speakers from the first sample, then one clean stretch each — S04."""
    span = first.size
    half = span // 2
    timeline = np.zeros(span + span, dtype=np.float32)
    timeline[:half] += first[:half] + second[:half]
    timeline[half:span] += first[half:]
    timeline[span : span + half] += second[half:]
    return timeline / max(1.0, float(np.abs(timeline).max()))
