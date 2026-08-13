"""Deterministic synthetic scenarios for the fake adapters (spec 16.1.3).

Each speaker is a pure tone at a distinct frequency, so the fake models can
recover "who is in this waveform" from the signal itself instead of being handed
the answer out of band. Overlap is a real sum of tones, and the fake separator
splits it back into its components — including in a permuted source order, which
is what makes the source-swap scenario (S03) meaningful.

These are integration fixtures, never an accuracy benchmark: spec 19.1 is
explicit that the deterministic harness cannot evaluate model quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sastt.domain.audio import (
    CANONICAL_SAMPLE_RATE,
    AudioBuffer,
    FloatArray,
    TimeInterval,
    merge_intervals,
    ms_to_samples,
)

#: Base frequency per speaker index; 90 Hz apart so FFT peaks never collide.
BASE_FREQUENCY_HZ = 220.0
FREQUENCY_STEP_HZ = 90.0
SPEAKER_AMPLITUDE = 0.30
PEAK_RELATIVE_THRESHOLD = 0.35


def speaker_frequency(index: int) -> float:
    return BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * index


@dataclass(frozen=True)
class ScriptedTurn:
    speaker: str
    start_ms: int
    end_ms: int
    text: str

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.start_ms, self.end_ms)


@dataclass(frozen=True)
class SourceOrder:
    """Forced separator output order for one region (source-swap scenario S03)."""

    start_ms: int
    end_ms: int
    order: tuple[str, ...]

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.start_ms, self.end_ms)


@dataclass(frozen=True)
class Scenario:
    """A synthetic session: speakers, turns and optional separator permutations."""

    name: str
    speakers: tuple[str, ...]
    turns: tuple[ScriptedTurn, ...]
    duration_ms: int
    sample_rate: int = CANONICAL_SAMPLE_RATE
    channels: int = 1
    source_orders: tuple[SourceOrder, ...] = ()
    enrolled: dict[str, str] = field(default_factory=dict)

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Scenario:
        turns = tuple(
            ScriptedTurn(
                speaker=str(turn["speaker"]),
                start_ms=int(turn["start_ms"]),
                end_ms=int(turn["end_ms"]),
                text=str(turn["text"]),
            )
            for turn in raw.get("turns", [])
        )
        speakers = tuple(raw.get("speakers") or sorted({turn.speaker for turn in turns}))
        orders = tuple(
            SourceOrder(
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                order=tuple(str(s) for s in item["order"]),
            )
            for item in raw.get("source_orders", [])
        )
        duration = int(raw.get("duration_ms") or max((t.end_ms for t in turns), default=0))
        return cls(
            name=str(raw.get("name", "scenario")),
            speakers=speakers,
            turns=turns,
            duration_ms=duration,
            sample_rate=int(raw.get("sample_rate", CANONICAL_SAMPLE_RATE)),
            channels=int(raw.get("channels", 1)),
            source_orders=orders,
            enrolled=dict(raw.get("enrolled", {})),
        )

    @classmethod
    def load(cls, path: Path | str) -> Scenario:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- queries ------------------------------------------------------------- #

    def speaker_index(self, speaker: str) -> int:
        return self.speakers.index(speaker)

    def frequency(self, speaker: str) -> float:
        return speaker_frequency(self.speaker_index(speaker))

    def turns_in(self, interval: TimeInterval) -> list[ScriptedTurn]:
        return [turn for turn in self.turns if turn.interval.intersects(interval)]

    def speakers_in(self, interval: TimeInterval) -> list[str]:
        seen: list[str] = []
        for turn in sorted(self.turns_in(interval), key=lambda t: t.start_ms):
            if turn.speaker not in seen:
                seen.append(turn.speaker)
        return seen

    def speech_intervals(self) -> list[TimeInterval]:
        return merge_intervals([turn.interval for turn in self.turns])

    def overlap_intervals(
        self, *, min_duration_ms: int = 300, merge_gap_ms: int = 200
    ) -> list[TimeInterval]:
        """Regions where at least two scripted turns are active."""
        edges = sorted({ms for turn in self.turns for ms in (turn.start_ms, turn.end_ms)})
        regions: list[TimeInterval] = []
        for start, end in zip(edges, edges[1:], strict=False):
            if end <= start:
                continue
            slot = TimeInterval(start, end)
            active = sum(1 for turn in self.turns if turn.interval.intersection_ms(slot) > 0)
            if active >= 2:
                regions.append(slot)
        merged = merge_intervals(regions, merge_gap_ms=merge_gap_ms)
        return [region for region in merged if region.duration_ms >= min_duration_ms]

    def source_order_for(self, interval: TimeInterval) -> tuple[str, ...] | None:
        for order in self.source_orders:
            if order.interval.intersects(interval):
                return order.order
        return None

    # -- rendering ----------------------------------------------------------- #

    def render(self, *, sample_rate: int | None = None) -> AudioBuffer:
        """Render the whole session to one mono buffer."""
        rate = sample_rate or self.sample_rate
        total = ms_to_samples(self.duration_ms, rate)
        samples = np.zeros((1, total), dtype=np.float32)
        for turn in self.turns:
            start = ms_to_samples(turn.start_ms, rate)
            end = min(ms_to_samples(turn.end_ms, rate), total)
            if end <= start:
                continue
            samples[0, start:end] += _tone(self.frequency(turn.speaker), start, end, rate)
        return AudioBuffer(
            samples=samples,
            sample_rate=rate,
            start_sample=0,
            channel_layout=("mono",),
            source_clock_hz=rate,
        )

    def render_speaker(
        self, speaker: str, interval: TimeInterval, *, sample_rate: int | None = None
    ) -> FloatArray:
        """Render one speaker's contribution inside ``interval`` (fake separation)."""
        rate = sample_rate or self.sample_rate
        start = ms_to_samples(interval.start_ms, rate)
        end = ms_to_samples(interval.end_ms, rate)
        out = np.zeros(end - start, dtype=np.float32)
        for turn in self.turns:
            if turn.speaker != speaker:
                continue
            clipped = turn.interval.clamp(interval)
            if clipped is None:
                continue
            local_start = ms_to_samples(clipped.start_ms, rate) - start
            local_end = ms_to_samples(clipped.end_ms, rate) - start
            absolute_start = ms_to_samples(clipped.start_ms, rate)
            absolute_end = ms_to_samples(clipped.end_ms, rate)
            out[local_start:local_end] += _tone(
                self.frequency(speaker), absolute_start, absolute_end, rate
            )[: local_end - local_start]
        return out


def _tone(frequency: float, start_sample: int, end_sample: int, sample_rate: int) -> FloatArray:
    """Phase-continuous tone slice, so a crop matches the full-session render."""
    index = np.arange(start_sample, end_sample, dtype=np.float64)
    tone: FloatArray = (
        SPEAKER_AMPLITUDE * np.sin(2 * np.pi * frequency * index / sample_rate)
    ).astype(np.float32)
    return tone


def detect_speaker_energies(
    samples: FloatArray,
    sample_rate: int,
    speakers: tuple[str, ...],
) -> dict[str, float]:
    """Recover per-speaker energy from a waveform via its FFT peaks."""
    mono = samples if samples.ndim == 1 else samples.mean(axis=0)
    if mono.size < 32:
        return {}
    spectrum = np.abs(np.fft.rfft(mono.astype(np.float64)))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    peak = float(spectrum.max()) if spectrum.size else 0.0
    energies: dict[str, float] = {}
    if peak <= 0.0:
        return energies
    for index, speaker in enumerate(speakers):
        target = speaker_frequency(index)
        window = np.abs(freqs - target) <= 15.0
        if not window.any():
            continue
        magnitude = float(spectrum[window].max())
        if magnitude / peak >= PEAK_RELATIVE_THRESHOLD:
            energies[speaker] = magnitude / peak
    return energies


def dominant_speaker(
    samples: FloatArray,
    sample_rate: int,
    speakers: tuple[str, ...],
) -> str | None:
    energies = detect_speaker_energies(samples, sample_rate, speakers)
    if not energies:
        return None
    return max(energies, key=lambda key: energies[key])


__all__ = [
    "BASE_FREQUENCY_HZ",
    "FREQUENCY_STEP_HZ",
    "SPEAKER_AMPLITUDE",
    "Scenario",
    "ScriptedTurn",
    "SourceOrder",
    "detect_speaker_energies",
    "dominant_speaker",
    "speaker_frequency",
]
