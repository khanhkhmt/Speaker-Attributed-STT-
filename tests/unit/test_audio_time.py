"""Interval and time math — spec 0.3, 5.1."""

from __future__ import annotations

import numpy as np
import pytest

from sastt.domain.audio import (
    AudioBuffer,
    TimeInterval,
    measure_quality,
    merge_intervals,
    ms_to_samples,
    samples_to_ms,
    seconds_to_ms,
    total_duration_ms,
)
from sastt.domain.errors import InvalidChannelLayoutError, UnsupportedAudioFormatError

pytestmark = pytest.mark.unit


def _buffer(seconds: float = 1.0, channels: int = 1, rate: int = 16_000) -> AudioBuffer:
    samples = np.zeros((channels, int(seconds * rate)), dtype=np.float32)
    layout = tuple(f"ch{i}" for i in range(channels)) if channels > 1 else ("mono",)
    return AudioBuffer(samples, rate, 0, layout, rate)


class TestSampleTimeConversion:
    def test_round_trip_is_exact_on_frame_boundaries(self) -> None:
        for ms in (0, 1, 20, 40, 1_000, 3_599_999):
            assert samples_to_ms(ms_to_samples(ms, 16_000), 16_000) == ms

    def test_no_drift_accumulates_over_four_hours(self) -> None:
        """Integer arithmetic only — spec 0.3 forbids accumulating floats."""
        rate = 16_000
        hop = ms_to_samples(20, rate)
        total_ms = samples_to_ms(hop * 720_000, rate)  # 720k hops of 20 ms = 4 h
        assert total_ms == 14_400_000

    def test_rejects_negative_input(self) -> None:
        with pytest.raises(ValueError):
            samples_to_ms(-1, 16_000)
        with pytest.raises(ValueError):
            ms_to_samples(10, 0)

    def test_seconds_to_ms_rounds(self) -> None:
        assert seconds_to_ms(0.5) == 500
        assert seconds_to_ms(1.2) == 1200


class TestTimeInterval:
    def test_rejects_empty_or_negative(self) -> None:
        with pytest.raises(ValueError):
            TimeInterval(10, 10)
        with pytest.raises(ValueError):
            TimeInterval(-1, 5)

    def test_intersection_and_gap(self) -> None:
        a = TimeInterval(0, 1000)
        b = TimeInterval(800, 1500)
        c = TimeInterval(2000, 2500)
        assert a.intersects(b)
        assert a.intersection_ms(b) == 200
        assert a.intersection_ms(c) == 0
        assert a.gap_ms(c) == 1000
        assert a.gap_ms(b) == 0

    def test_pad_is_clamped(self) -> None:
        padded = TimeInterval(100, 200).pad(500, lower_bound_ms=0, upper_bound_ms=600)
        assert padded == TimeInterval(0, 600)

    def test_clamp_returns_none_when_disjoint(self) -> None:
        assert TimeInterval(0, 100).clamp(TimeInterval(200, 300)) is None
        assert TimeInterval(0, 300).clamp(TimeInterval(100, 200)) == TimeInterval(100, 200)

    def test_merge_respects_gap(self) -> None:
        intervals = [TimeInterval(0, 100), TimeInterval(150, 200), TimeInterval(400, 500)]
        assert merge_intervals(intervals, merge_gap_ms=50) == [
            TimeInterval(0, 200),
            TimeInterval(400, 500),
        ]
        assert merge_intervals(intervals) == intervals

    def test_total_duration_counts_overlap_once(self) -> None:
        assert total_duration_ms([TimeInterval(0, 1000), TimeInterval(500, 1500)]) == 1500


class TestAudioBuffer:
    def test_rejects_out_of_range_channel_count(self) -> None:
        with pytest.raises(InvalidChannelLayoutError):
            AudioBuffer(np.zeros((9, 10), dtype=np.float32), 16_000, 0, tuple("abcdefghi"), 16_000)

    def test_rejects_layout_mismatch(self) -> None:
        with pytest.raises(InvalidChannelLayoutError):
            AudioBuffer(np.zeros((2, 10), dtype=np.float32), 16_000, 0, ("mono",), 16_000)

    def test_rejects_nan(self) -> None:
        samples = np.zeros((1, 10), dtype=np.float32)
        samples[0, 3] = np.nan
        with pytest.raises(UnsupportedAudioFormatError):
            AudioBuffer(samples, 16_000, 0, ("mono",), 16_000)

    def test_crop_keeps_absolute_position(self) -> None:
        buffer = _buffer(2.0)
        crop = buffer.crop_ms(TimeInterval(500, 1500))
        assert crop.start_ms == 500
        assert crop.duration_ms == 1000
        assert crop.start_sample == ms_to_samples(500, 16_000)

    def test_to_mono_preserves_original(self) -> None:
        stereo = _buffer(0.5, channels=2)
        mono = stereo.to_mono()
        assert mono.num_channels == 1
        assert stereo.num_channels == 2  # spec 5.1.4: the original is never overwritten

    def test_quality_metrics(self) -> None:
        samples = np.full((1, 1600), 0.5, dtype=np.float32)
        quality = measure_quality(AudioBuffer(samples, 16_000, 0, ("mono",), 16_000))
        assert quality.clipping_ratio == 0.0
        assert quality.rms == pytest.approx(0.5, abs=1e-6)
        assert quality.dc_offset == pytest.approx(0.5, abs=1e-6)
