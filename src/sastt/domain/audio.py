"""Audio domain types and interval math — spec 5.1.

Two hard rules from the spec are enforced here:

* internal time is a sample index or an integer millisecond count; floats are
  never accumulated (spec 0.3, 5.1.7);
* the original multichannel input and its channel map are preserved; the mono
  16 kHz buffer is only a derivative (spec 1.1, 5.1.4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sastt.domain.errors import InvalidChannelLayoutError, UnsupportedAudioFormatError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float32]
else:  # pragma: no cover - runtime alias
    FloatArray = np.ndarray

MS_PER_SECOND = 1000
MIN_CHANNELS = 1
MAX_CHANNELS = 8
MIN_INPUT_SAMPLE_RATE = 8_000
MAX_INPUT_SAMPLE_RATE = 48_000
CANONICAL_SAMPLE_RATE = 16_000


def samples_to_ms(sample_index: int, sample_rate: int) -> int:
    """Convert a sample index to integer milliseconds (round half up).

    Integer arithmetic only: no float accumulation, therefore no drift.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    return (sample_index * MS_PER_SECOND + sample_rate // 2) // sample_rate


def ms_to_samples(ms: int, sample_rate: int) -> int:
    """Convert integer milliseconds to a sample index (round half up)."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if ms < 0:
        raise ValueError("ms must be non-negative")
    return (ms * sample_rate + MS_PER_SECOND // 2) // MS_PER_SECOND


def seconds_to_ms(seconds: float) -> int:
    """Convert a config value expressed in seconds to integer milliseconds."""
    return int(round(seconds * MS_PER_SECOND))


@dataclass(frozen=True, order=True)
class TimeInterval:
    """Half-open interval ``[start_ms, end_ms)`` in absolute session time."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.start_ms, int) or not isinstance(self.end_ms, int):
            raise TypeError("TimeInterval bounds must be integer milliseconds")
        if self.start_ms < 0:
            raise ValueError(f"start_ms must be >= 0, got {self.start_ms}")
        if self.end_ms <= self.start_ms:
            raise ValueError(f"end_ms must be > start_ms, got {self.start_ms}..{self.end_ms}")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def intersects(self, other: TimeInterval) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms

    def intersection_ms(self, other: TimeInterval) -> int:
        """Overlap length in ms; ``0`` when the intervals are disjoint."""
        return max(0, min(self.end_ms, other.end_ms) - max(self.start_ms, other.start_ms))

    def contains(self, ms: int) -> bool:
        return self.start_ms <= ms < self.end_ms

    def gap_ms(self, other: TimeInterval) -> int:
        """Silence between two intervals; ``0`` when they touch or overlap."""
        if self.intersects(other):
            return 0
        return max(other.start_ms - self.end_ms, self.start_ms - other.end_ms)

    def pad(
        self, context_ms: int, *, lower_bound_ms: int = 0, upper_bound_ms: int | None = None
    ) -> TimeInterval:
        """Grow the interval by ``context_ms`` on both sides, clamped to bounds.

        Used for the 0.5 s separation context of spec 5.4.
        """
        start = max(lower_bound_ms, self.start_ms - context_ms)
        end = self.end_ms + context_ms
        if upper_bound_ms is not None:
            end = min(upper_bound_ms, end)
        return TimeInterval(start, max(end, start + 1))

    def clamp(self, bounds: TimeInterval) -> TimeInterval | None:
        start = max(self.start_ms, bounds.start_ms)
        end = min(self.end_ms, bounds.end_ms)
        if end <= start:
            return None
        return TimeInterval(start, end)


def merge_intervals(intervals: list[TimeInterval], merge_gap_ms: int = 0) -> list[TimeInterval]:
    """Sort and merge intervals separated by at most ``merge_gap_ms`` (spec 5.2)."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start_ms - last.end_ms <= merge_gap_ms:
            merged[-1] = TimeInterval(last.start_ms, max(last.end_ms, interval.end_ms))
        else:
            merged.append(interval)
    return merged


def total_duration_ms(intervals: list[TimeInterval]) -> int:
    """Total covered time, counting overlapped regions once."""
    return sum(interval.duration_ms for interval in merge_intervals(intervals))


@dataclass(frozen=True)
class AudioBuffer:
    """Canonical audio domain type — spec 5.1.

    ``samples`` has shape ``[channels, samples]``, dtype float32, range ``[-1, 1]``.
    ``start_sample`` anchors the buffer in absolute session time.
    """

    samples: FloatArray
    sample_rate: int
    start_sample: int
    channel_layout: tuple[str, ...]
    source_clock_hz: int

    def __post_init__(self) -> None:
        arr = self.samples
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise UnsupportedAudioFormatError("samples must be a 2-D [channels, samples] array")
        if arr.dtype != np.float32:
            raise UnsupportedAudioFormatError(f"samples dtype must be float32, got {arr.dtype}")
        channels = int(arr.shape[0])
        if channels < MIN_CHANNELS or channels > MAX_CHANNELS:
            raise InvalidChannelLayoutError(
                f"channel count {channels} outside supported range {MIN_CHANNELS}-{MAX_CHANNELS}"
            )
        if len(self.channel_layout) != channels:
            raise InvalidChannelLayoutError(
                f"channel_layout has {len(self.channel_layout)} entries for {channels} channels"
            )
        if arr.shape[1] == 0:
            raise UnsupportedAudioFormatError("audio buffer is empty")
        if self.sample_rate <= 0:
            raise UnsupportedAudioFormatError(f"invalid sample rate {self.sample_rate}")
        if self.start_sample < 0:
            raise UnsupportedAudioFormatError("start_sample must be non-negative")
        if not np.isfinite(arr).all():
            raise UnsupportedAudioFormatError("audio contains NaN or Inf samples")

    @property
    def num_channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[1])

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.num_samples

    @property
    def start_ms(self) -> int:
        return samples_to_ms(self.start_sample, self.sample_rate)

    @property
    def end_ms(self) -> int:
        return samples_to_ms(self.end_sample, self.sample_rate)

    @property
    def duration_ms(self) -> int:
        return samples_to_ms(self.num_samples, self.sample_rate)

    @property
    def is_mono(self) -> bool:
        return self.num_channels == 1

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.start_ms, max(self.end_ms, self.start_ms + 1))

    def channel(self, index: int) -> FloatArray:
        channel: FloatArray = self.samples[index]
        return channel

    def crop_ms(self, interval: TimeInterval) -> AudioBuffer:
        """Crop by absolute time; the result keeps its absolute ``start_sample``."""
        start = ms_to_samples(interval.start_ms, self.sample_rate)
        end = ms_to_samples(interval.end_ms, self.sample_rate)
        start = max(start, self.start_sample)
        end = min(end, self.end_sample)
        if end <= start:
            raise UnsupportedAudioFormatError("crop interval does not intersect the buffer")
        local_start = start - self.start_sample
        local_end = end - self.start_sample
        return AudioBuffer(
            samples=np.ascontiguousarray(self.samples[:, local_start:local_end]),
            sample_rate=self.sample_rate,
            start_sample=start,
            channel_layout=self.channel_layout,
            source_clock_hz=self.source_clock_hz,
        )

    def to_mono(self) -> AudioBuffer:
        """Downmix to one channel.

        Only legal when building the mono derivative; never for the spatial
        branch, which needs per-channel data (spec 5.1.4).
        """
        if self.is_mono:
            return self
        mono = self.samples.mean(axis=0, keepdims=True).astype(np.float32)
        return AudioBuffer(
            samples=mono,
            sample_rate=self.sample_rate,
            start_sample=self.start_sample,
            channel_layout=("mono",),
            source_clock_hz=self.source_clock_hz,
        )


@dataclass(frozen=True)
class AudioQuality:
    """Ingest quality metadata — spec 5.1.5."""

    clipping_ratio: float
    rms: float
    dc_offset: float
    speech_duration_ms: int
    estimated_snr_db: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "clipping_ratio": self.clipping_ratio,
            "rms": self.rms,
            "dc_offset": self.dc_offset,
            "speech_duration_ms": self.speech_duration_ms,
            "estimated_snr_db": self.estimated_snr_db,
        }


def measure_quality(
    buffer: AudioBuffer,
    *,
    speech_duration_ms: int = 0,
    clip_threshold: float = 0.99,
    estimated_snr_db: float | None = None,
) -> AudioQuality:
    """Compute the ingest metrics of spec 5.1.5 over the mono derivative."""
    mono = buffer.to_mono().samples[0]
    clipping_ratio = float(np.mean(np.abs(mono) >= clip_threshold))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    dc_offset = float(np.mean(mono, dtype=np.float64))
    return AudioQuality(
        clipping_ratio=clipping_ratio,
        rms=rms,
        dc_offset=dc_offset,
        speech_duration_ms=speech_duration_ms,
        estimated_snr_db=estimated_snr_db,
    )


@dataclass(frozen=True)
class AudioAsset:
    """One decoded input: the preserved original plus its mono derivative.

    Spec 5.1: decode once, checksum the input, derive mono 16 kHz for the
    models, never overwrite the multichannel original.
    """

    original: AudioBuffer
    mono_16k: AudioBuffer
    input_sha256: str
    container_format: str
    quality: AudioQuality
    channel_map: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.mono_16k.sample_rate != CANONICAL_SAMPLE_RATE:
            raise UnsupportedAudioFormatError(
                f"derivative must be {CANONICAL_SAMPLE_RATE} Hz, got {self.mono_16k.sample_rate}"
            )
        if not self.mono_16k.is_mono:
            raise InvalidChannelLayoutError("derivative must be mono")
        if not self.channel_map:
            object.__setattr__(self, "channel_map", self.original.channel_layout)

    @property
    def duration_ms(self) -> int:
        return self.original.duration_ms

    @property
    def is_multichannel(self) -> bool:
        return self.original.num_channels > 1


def sha256_of_bytes(payload: bytes) -> str:
    """SHA-256 checksum of the raw input (spec 5.1.2)."""
    return hashlib.sha256(payload).hexdigest()


def sha256_of_samples(buffer: AudioBuffer) -> str:
    """Stable checksum of decoded samples, for fixtures and reproducibility."""
    digest = hashlib.sha256()
    digest.update(str(buffer.sample_rate).encode())
    digest.update(str(buffer.channel_layout).encode())
    digest.update(np.ascontiguousarray(buffer.samples).tobytes())
    return digest.hexdigest()


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "FloatArray",
    "MAX_CHANNELS",
    "MAX_INPUT_SAMPLE_RATE",
    "MIN_CHANNELS",
    "MIN_INPUT_SAMPLE_RATE",
    "MS_PER_SECOND",
    "AudioAsset",
    "AudioBuffer",
    "AudioQuality",
    "TimeInterval",
    "measure_quality",
    "merge_intervals",
    "ms_to_samples",
    "samples_to_ms",
    "seconds_to_ms",
    "sha256_of_bytes",
    "sha256_of_samples",
    "total_duration_ms",
]
