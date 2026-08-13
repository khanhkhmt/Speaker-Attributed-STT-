"""FFmpeg audio decoder — spec 5.1, 1.1.

Decodes WAV/FLAC/MP3/M4A-AAC/Ogg-Opus once, keeps the original channel layout
and derives the mono 16 kHz buffer the models consume. The original is never
overwritten and is never downmixed before the spatial branch (spec 5.1.4).

Hardening of spec 14.8 lives here too: a size ceiling, a duration ceiling and a
decode timeout, so a malicious file cannot pin a worker.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from sastt.domain.audio import (
    CANONICAL_SAMPLE_RATE,
    MAX_CHANNELS,
    MAX_INPUT_SAMPLE_RATE,
    MIN_CHANNELS,
    MIN_INPUT_SAMPLE_RATE,
    AudioAsset,
    AudioBuffer,
    FloatArray,
    measure_quality,
    sha256_of_bytes,
)
from sastt.domain.errors import (
    AudioTooLongError,
    InvalidChannelLayoutError,
    UnsupportedAudioFormatError,
)
from sastt.observability import CallContext

DEFAULT_MAX_BYTES = 2 * 1024**3
DEFAULT_DECODE_TIMEOUT_SECONDS = 300.0


class FfmpegAudioDecoder:
    """``AudioDecoder`` port backed by the ``ffmpeg``/``ffprobe`` binaries."""

    def __init__(
        self,
        *,
        max_hours: float = 4.0,
        max_bytes: int = DEFAULT_MAX_BYTES,
        decode_timeout_seconds: float = DEFAULT_DECODE_TIMEOUT_SECONDS,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ) -> None:
        self.max_hours = max_hours
        self.max_bytes = max_bytes
        self.decode_timeout_seconds = decode_timeout_seconds
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    # -- port ---------------------------------------------------------------- #

    def decode(
        self,
        payload: bytes,
        ctx: CallContext,
        *,
        container_hint: str | None = None,
    ) -> AudioAsset:
        ctx.check()
        if not payload:
            raise UnsupportedAudioFormatError("empty payload")
        if len(payload) > self.max_bytes:
            raise UnsupportedAudioFormatError(
                f"input is {len(payload)} bytes, over the {self.max_bytes} byte limit"
            )

        with tempfile.NamedTemporaryFile(suffix=".input") as handle:
            handle.write(payload)
            handle.flush()
            path = Path(handle.name)
            probe = self._probe(path)
            samples = self._decode(path, probe["channels"], probe["sample_rate"])

        original = AudioBuffer(
            samples=samples,
            sample_rate=probe["sample_rate"],
            start_sample=0,
            channel_layout=probe["channel_layout"],
            source_clock_hz=probe["sample_rate"],
        )
        mono = _to_canonical_mono(original)
        return AudioAsset(
            original=original,
            mono_16k=mono,
            input_sha256=sha256_of_bytes(payload),
            container_format=container_hint or probe["format_name"],
            quality=measure_quality(mono),
            channel_map=probe["channel_layout"],
        )

    # -- internals ------------------------------------------------------------ #

    def _probe(self, path: Path) -> dict[str, object]:
        command = [
            self.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate,codec_name,channel_layout:format=format_name,duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=self.decode_timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise UnsupportedAudioFormatError("probing timed out") from exc
        if completed.returncode != 0:
            raise UnsupportedAudioFormatError(
                "ffprobe could not read the input",
                details={"stderr": completed.stderr.decode("utf-8", "replace")[:400]},
            )

        report = json.loads(completed.stdout or "{}")
        streams = report.get("streams") or []
        if not streams:
            raise UnsupportedAudioFormatError("no audio stream found")
        stream = streams[0]

        channels = int(stream.get("channels") or 0)
        if channels < MIN_CHANNELS or channels > MAX_CHANNELS:
            raise InvalidChannelLayoutError(
                f"channel count {channels} outside the supported range "
                f"{MIN_CHANNELS}-{MAX_CHANNELS}"
            )
        sample_rate = int(stream.get("sample_rate") or 0)
        if not MIN_INPUT_SAMPLE_RATE <= sample_rate <= MAX_INPUT_SAMPLE_RATE:
            raise UnsupportedAudioFormatError(
                f"sample rate {sample_rate} outside the supported range "
                f"{MIN_INPUT_SAMPLE_RATE}-{MAX_INPUT_SAMPLE_RATE} Hz"
            )

        duration_raw = (report.get("format") or {}).get("duration")
        duration = float(duration_raw) if duration_raw not in (None, "N/A") else 0.0
        if duration < 0:
            raise UnsupportedAudioFormatError("negative duration")
        if duration > self.max_hours * 3600:
            raise AudioTooLongError(
                f"{duration:.0f}s exceeds the {self.max_hours}h limit",
                details={"duration_seconds": duration},
            )

        layout = str(stream.get("channel_layout") or "")
        return {
            "channels": channels,
            "sample_rate": sample_rate,
            "codec_name": str(stream.get("codec_name") or "unknown"),
            "format_name": str((report.get("format") or {}).get("format_name") or "unknown"),
            "channel_layout": _channel_names(channels, layout),
        }

    def _decode(self, path: Path, channels: int, sample_rate: int) -> FloatArray:
        command = [
            self.ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=self.decode_timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise UnsupportedAudioFormatError("decoding timed out") from exc
        if completed.returncode != 0 or not completed.stdout:
            raise UnsupportedAudioFormatError(
                "ffmpeg could not decode the input",
                details={"stderr": completed.stderr.decode("utf-8", "replace")[:400]},
            )

        flat = np.frombuffer(completed.stdout, dtype="<f4")
        usable = (flat.size // channels) * channels
        interleaved = flat[:usable].reshape(-1, channels).T
        samples = np.ascontiguousarray(interleaved, dtype=np.float32)
        if not np.isfinite(samples).all():
            raise UnsupportedAudioFormatError("decoded audio contains NaN or Inf")
        return samples


def _channel_names(channels: int, layout: str) -> tuple[str, ...]:
    if channels == 1:
        return ("mono",)
    if layout and layout != "unknown":
        parts = tuple(part.strip() for part in layout.split("+"))
        if len(parts) == channels:
            return parts
    return tuple(f"ch{index}" for index in range(channels))


def _to_canonical_mono(buffer: AudioBuffer) -> AudioBuffer:
    """Mono 16 kHz derivative for the models (spec 5.1.3)."""
    mono = buffer.to_mono().samples[0]
    if buffer.sample_rate != CANONICAL_SAMPLE_RATE:
        from math import gcd

        divisor = gcd(CANONICAL_SAMPLE_RATE, buffer.sample_rate)
        mono = resample_poly(
            mono, CANONICAL_SAMPLE_RATE // divisor, buffer.sample_rate // divisor
        ).astype(np.float32)
    return AudioBuffer(
        samples=np.ascontiguousarray(mono[np.newaxis, :], dtype=np.float32),
        sample_rate=CANONICAL_SAMPLE_RATE,
        start_sample=0,
        channel_layout=("mono",),
        source_clock_hz=buffer.source_clock_hz,
    )


__all__ = ["FfmpegAudioDecoder"]
