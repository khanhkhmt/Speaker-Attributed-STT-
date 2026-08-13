"""Fake model adapters — Milestone 0 (spec 18).

They satisfy the ports of spec 9 and derive their answers from the *signal*
(FFT peaks of the synthetic tones), not from a hidden answer key: diarization,
overlap detection, separation and embedding all read the waveform they are
given, so windowing, cropping and permutation behave like the real thing.

Transcript text is the one thing a fake cannot invent, so it comes from the
scenario script. These adapters are integration fixtures only — spec 18 rule 6
forbids presenting an oracle as a model test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from sastt.adapters.fake.scenario import Scenario, detect_speaker_energies, speaker_frequency
from sastt.domain.audio import (
    CANONICAL_SAMPLE_RATE,
    AudioAsset,
    AudioBuffer,
    FloatArray,
    TimeInterval,
    measure_quality,
    merge_intervals,
    ms_to_samples,
    samples_to_ms,
    sha256_of_bytes,
    total_duration_ms,
)
from sastt.domain.errors import (
    InsufficientSpeechForEmbeddingError,
    InvalidChannelLayoutError,
    SeparationFailedError,
    TenantAccessDeniedError,
    UnsupportedAudioFormatError,
)
from sastt.domain.speakers import (
    DiarizationResult,
    EmbeddingOrigin,
    EnrollmentClipReport,
    EnrollmentReport,
    OverlapRegion,
    SeparatedBatch,
    SourceQuality,
    SpeakerEmbedding,
    SpeakerPrototype,
    SpeakerTurn,
    VoiceIdDecision,
    l2_normalize,
)
from sastt.domain.transcript import ASRResult, Word
from sastt.observability import CallContext

EMBEDDING_DIMENSION = 192  # CAM++ embedding size (spec 0.2)
FRAME_MS = 100
CLUSTER_PREFIX = "cluster_"


def _cluster_id(speaker: str) -> str:
    return f"{CLUSTER_PREFIX}{speaker}"


def _frames(buffer: AudioBuffer, frame_ms: int = FRAME_MS) -> list[tuple[TimeInterval, FloatArray]]:
    step = ms_to_samples(frame_ms, buffer.sample_rate)
    mono = buffer.to_mono().samples[0]
    out: list[tuple[TimeInterval, FloatArray]] = []
    for offset in range(0, mono.size, step):
        chunk = mono[offset : offset + step]
        if chunk.size < step // 2:
            break
        start_ms = samples_to_ms(buffer.start_sample + offset, buffer.sample_rate)
        end_ms = samples_to_ms(buffer.start_sample + offset + chunk.size, buffer.sample_rate)
        out.append((TimeInterval(start_ms, end_ms), chunk))
    return out


class FakeAudioDecoder:
    """Decodes raw interleaved PCM s16le — ``AudioDecoder`` port (spec 5.1)."""

    def __init__(self, sample_rate: int = CANONICAL_SAMPLE_RATE, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

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
        if len(payload) % (2 * self.channels) != 0:
            raise UnsupportedAudioFormatError("PCM payload is not frame aligned")
        if not 1 <= self.channels <= 8:
            raise InvalidChannelLayoutError(f"unsupported channel count {self.channels}")

        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        interleaved = pcm.reshape(-1, self.channels).T
        layout = tuple(f"ch{i}" for i in range(self.channels)) if self.channels > 1 else ("mono",)
        original = AudioBuffer(
            samples=np.ascontiguousarray(interleaved),
            sample_rate=self.sample_rate,
            start_sample=0,
            channel_layout=layout,
            source_clock_hz=self.sample_rate,
        )
        if self.sample_rate != CANONICAL_SAMPLE_RATE:
            raise UnsupportedAudioFormatError(
                "the fake decoder does not resample; feed 16 kHz PCM (real resampling: Milestone 1)"
            )
        mono = original.to_mono()
        return AudioAsset(
            original=original,
            mono_16k=mono,
            input_sha256=sha256_of_bytes(payload),
            container_format=container_hint or "pcm_s16le",
            quality=measure_quality(mono),
            channel_map=layout,
        )


class FakeVoiceActivityDetector:
    """Energy-based VAD — ``VoiceActivityDetector`` port."""

    def __init__(self, threshold: float = 0.02, merge_gap_ms: int = 120) -> None:
        self.threshold = threshold
        self.merge_gap_ms = merge_gap_ms

    def detect(self, buffer: AudioBuffer, ctx: CallContext) -> list[TimeInterval]:
        ctx.check()
        speech = [
            interval
            for interval, chunk in _frames(buffer)
            if float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) >= self.threshold
        ]
        return merge_intervals(speech, merge_gap_ms=self.merge_gap_ms)


class FakeDiarizer:
    """Diarization derived from the tone content of the buffer — spec 5.2.

    ``regular_tracks`` keeps concurrent speakers; ``exclusive_tracks`` assigns
    every frame to its dominant speaker and is only usable for non-overlap
    alignment.
    """

    def __init__(
        self, speakers: tuple[str, ...], *, model_version: str = "fake-diarizer@1"
    ) -> None:
        self.speakers = speakers
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    def diarize(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ) -> DiarizationResult:
        ctx.check()
        per_speaker: dict[str, list[TimeInterval]] = {}
        exclusive: dict[str, list[TimeInterval]] = {}
        overlaps: list[tuple[TimeInterval, float]] = []

        for interval, chunk in _frames(buffer):
            energies = detect_speaker_energies(chunk, buffer.sample_rate, self.speakers)
            if not energies:
                continue
            for speaker in energies:
                per_speaker.setdefault(speaker, []).append(interval)
            dominant = max(energies, key=lambda key: energies[key])
            exclusive.setdefault(dominant, []).append(interval)
            if len(energies) >= 2:
                overlaps.append((interval, min(1.0, sum(energies.values()) / len(energies))))

        regular = [
            SpeakerTurn(cluster_id=_cluster_id(speaker), interval=merged, kind="regular")
            for speaker, intervals in sorted(per_speaker.items())
            for merged in merge_intervals(intervals, merge_gap_ms=FRAME_MS)
        ]
        exclusive_tracks = [
            SpeakerTurn(cluster_id=_cluster_id(speaker), interval=merged, kind="exclusive")
            for speaker, intervals in sorted(exclusive.items())
            for merged in merge_intervals(intervals, merge_gap_ms=FRAME_MS)
        ]
        overlap_regions = [
            OverlapRegion(
                interval=merged,
                osd_activation=max(
                    (
                        activation
                        for interval, activation in overlaps
                        if interval.intersects(merged)
                    ),
                    default=None,
                ),
                model_version=self._model_version,
            )
            for merged in merge_intervals([i for i, _ in overlaps], merge_gap_ms=FRAME_MS * 2)
        ]
        estimated = min(max(len(per_speaker), min_speakers), max_speakers)
        return DiarizationResult(
            turns=sorted(regular, key=lambda t: (t.start_ms, t.cluster_id)),
            regular_tracks=sorted(regular, key=lambda t: (t.start_ms, t.cluster_id)),
            exclusive_tracks=sorted(exclusive_tracks, key=lambda t: (t.start_ms, t.cluster_id)),
            overlap_regions=overlap_regions,
            estimated_session_speakers=estimated,
            model_version=self._model_version,
        )


class FakeOverlapDetector:
    """OSD over tone content — ``OverlapDetector`` port (spec 5.2)."""

    def __init__(
        self,
        speakers: tuple[str, ...],
        *,
        min_duration_ms: int = 300,
        merge_gap_ms: int = 200,
        model_version: str = "fake-osd@1",
    ) -> None:
        self.speakers = speakers
        self.min_duration_ms = min_duration_ms
        self.merge_gap_ms = merge_gap_ms
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect(self, buffer: AudioBuffer, ctx: CallContext) -> list[OverlapRegion]:
        ctx.check()
        hits: list[tuple[TimeInterval, float]] = []
        for interval, chunk in _frames(buffer):
            energies = detect_speaker_energies(chunk, buffer.sample_rate, self.speakers)
            if len(energies) >= 2:
                hits.append((interval, min(1.0, sum(energies.values()) / len(energies))))
        regions = []
        for merged in merge_intervals([i for i, _ in hits], merge_gap_ms=self.merge_gap_ms):
            if merged.duration_ms < self.min_duration_ms:
                continue
            activation = max(
                (value for interval, value in hits if interval.intersects(merged)), default=None
            )
            regions.append(
                OverlapRegion(
                    interval=merged,
                    osd_activation=activation,
                    model_version=self._model_version,
                )
            )
        return regions


class FakeSpeechSeparator:
    """Band-pass separation of the synthetic tones — ``SpeechSeparator`` port.

    The output order is deliberately unstable (``alternate_order`` or a scenario
    permutation), because a real separator gives no cross-chunk identity
    guarantee (spec 5.4, S03).
    """

    def __init__(
        self,
        speakers: tuple[str, ...],
        *,
        backend: str = "fake_two_source",
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        supported_source_counts: tuple[int, ...] = (2,),
        scenario: Scenario | None = None,
        alternate_order: bool = False,
        fail_first_call: bool = False,
    ) -> None:
        self.speakers = speakers
        self._backend = backend
        self._sample_rate = sample_rate
        self._supported = supported_source_counts
        self.scenario = scenario
        self.alternate_order = alternate_order
        self.fail_first_call = fail_first_call
        self.calls = 0

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def separator_version(self) -> str:
        return f"{self._backend}@fake"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def supported_source_counts(self) -> tuple[int, ...]:
        return self._supported

    def separate(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        requested_source_count: int,
    ) -> SeparatedBatch:
        ctx.check()
        self.calls += 1
        if self.fail_first_call and self.calls == 1:
            raise SeparationFailedError("fake separator failure (first call)")
        if requested_source_count not in self._supported:
            raise SeparationFailedError(
                f"{self._backend} does not support K={requested_source_count}"
            )

        mono = buffer.to_mono().samples[0]
        energies = detect_speaker_energies(mono, buffer.sample_rate, self.speakers)
        present = sorted(energies, key=lambda key: energies[key], reverse=True)

        order = None
        if self.scenario is not None:
            order = self.scenario.source_order_for(buffer.interval)
        if order is not None:
            present = [s for s in order if s in present] + [s for s in present if s not in order]
        elif self.alternate_order and self.calls % 2 == 0:
            present = list(reversed(present))

        selected = present[:requested_source_count]
        sources = np.zeros((requested_source_count, mono.size), dtype=np.float32)
        quality: list[SourceQuality] = []
        for index in range(requested_source_count):
            if index < len(selected):
                speaker = selected[index]
                sources[index] = _bandpass(mono, buffer.sample_rate, self.speakers.index(speaker))
                energy = float(np.sqrt(np.mean(np.square(sources[index], dtype=np.float64))))
                quality.append(
                    SourceQuality(
                        speech_duration_ms=buffer.duration_ms,
                        energy_ratio=energy,
                        passed_gate=energy > 0.01,
                        reasons=() if energy > 0.01 else ("low_energy",),
                    )
                )
            else:
                quality.append(
                    SourceQuality(
                        speech_duration_ms=0,
                        energy_ratio=0.0,
                        passed_gate=False,
                        reasons=("no_source_found",),
                    )
                )
        return SeparatedBatch(
            sources=sources,
            sample_rate=buffer.sample_rate,
            requested_source_count=requested_source_count,
            estimated_source_count=len(present),
            source_quality=quality,
            separator_version=self.separator_version,
            start_sample=buffer.start_sample,
        )


def _bandpass(
    mono: FloatArray, sample_rate: int, speaker_index: int, width_hz: float = 25.0
) -> FloatArray:
    spectrum = np.fft.rfft(mono.astype(np.float64))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    target = speaker_frequency(speaker_index)
    mask = np.abs(freqs - target) <= width_hz
    filtered = np.zeros_like(spectrum)
    filtered[mask] = spectrum[mask]
    source: FloatArray = np.fft.irfft(filtered, n=mono.size).astype(np.float32)
    return source


class FakeSpeechRecognizer:
    """Scripted ASR — ``SpeechRecognizer`` port (spec 5.5).

    The speaker is recovered from the waveform; only the words come from the
    scenario script. Word timestamps are absolute session time.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        model_version: str = "fake-asr@1",
        language: str = "vi",
        word_probability: float = 0.91,
    ) -> None:
        self.scenario = scenario
        self._model_version = model_version
        self.language = language
        self.word_probability = word_probability

    @property
    def model_version(self) -> str:
        return self._model_version

    def transcribe(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        language: str | None = None,
        source_track: int | None = None,
    ) -> ASRResult:
        ctx.check()
        mono = buffer.to_mono().samples[0]
        energies = detect_speaker_energies(mono, buffer.sample_rate, self.scenario.speakers)
        words: list[Word] = []
        if energies:
            speaker = max(energies, key=lambda key: energies[key])
            words = self._words_for(speaker, buffer.interval, source_track)
        return ASRResult(
            words=words,
            detected_language=language or self.language,
            language_score=None,
            model_version=self._model_version,
            raw_scores={"asr_word_probability": self.word_probability} if words else {},
        )

    def _words_for(
        self, speaker: str, interval: TimeInterval, source_track: int | None
    ) -> list[Word]:
        words: list[Word] = []
        for turn in self.scenario.turns:
            if turn.speaker != speaker:
                continue
            clipped = turn.interval.clamp(interval)
            if clipped is None or clipped.duration_ms < 50:
                continue
            tokens = turn.text.split()
            if not tokens:
                continue
            span = turn.interval.duration_ms / len(tokens)
            for index, token in enumerate(tokens):
                start = turn.start_ms + int(index * span)
                end = turn.start_ms + int((index + 1) * span)
                word_interval = TimeInterval(start, max(end, start + 1))
                if word_interval.intersection_ms(clipped) <= 0:
                    continue
                words.append(
                    Word(
                        text=token,
                        start_ms=word_interval.start_ms,
                        end_ms=word_interval.end_ms,
                        raw_probability=self.word_probability,
                        source_track=source_track,
                    )
                )
        return sorted(words, key=lambda w: w.start_ms)


class FakeSpeakerEmbedder:
    """Deterministic speaker embeddings — ``SpeakerEmbedder`` port (spec 5.6).

    Each speaker owns a fixed random unit vector; a mixture yields an
    energy-weighted combination, so a leaky source lands between two prototypes
    exactly as it would in production.
    """

    def __init__(
        self,
        speakers: tuple[str, ...],
        *,
        model_version: str = "fake-campplus@1",
        minimum_speech_ms: int = 1500,
        target_speech_ms: int = 3000,
    ) -> None:
        self.speakers = speakers
        self._model_version = model_version
        self.minimum_speech_ms = minimum_speech_ms
        self.target_speech_ms = target_speech_ms
        self._basis = {speaker: _unit_vector(speaker) for speaker in speakers}

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def embed(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        speech_intervals: list[TimeInterval] | None = None,
        origin: EmbeddingOrigin = "clean",
        source_track: int | None = None,
    ) -> SpeakerEmbedding:
        ctx.check()
        speech_ms = total_duration_ms(speech_intervals) if speech_intervals else buffer.duration_ms
        if speech_ms < self.minimum_speech_ms:
            raise InsufficientSpeechForEmbeddingError(
                f"{speech_ms} ms of speech is below the {self.minimum_speech_ms} ms minimum",
                details={"speech_ms": speech_ms, "minimum_ms": self.minimum_speech_ms},
            )
        mono = buffer.to_mono().samples[0]
        energies = detect_speaker_energies(mono, buffer.sample_rate, self.speakers)
        if not energies:
            raise InsufficientSpeechForEmbeddingError("no speech content found in buffer")

        total = sum(energies.values())
        vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        for speaker, energy in energies.items():
            vector += self._basis[speaker] * float(energy / total)
        purity = max(energies.values()) / total
        quality = float(min(1.0, speech_ms / self.target_speech_ms) * purity)
        return SpeakerEmbedding(
            vector=l2_normalize(vector),
            model_version=self._model_version,
            quality=quality,
            speech_duration_ms=speech_ms,
            origin=origin,
            interval=buffer.interval,
            source_track=source_track,
        )


def _unit_vector(key: str) -> FloatArray:
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32))


@dataclass
class _RegistryEntry:
    identity_id: str
    display_name: str
    prototypes: list[SpeakerPrototype] = field(default_factory=list)
    consent_ref: str | None = None


class FakeVoiceRegistry:
    """Tenant-scoped in-memory registry — ``VoiceRegistry`` port (spec 5.10, 14).

    Fails closed while thresholds are null and never lets one tenant see
    another tenant's templates (S15).
    """

    def __init__(
        self,
        *,
        embedding_model_version: str,
        accept_threshold: float | None = None,
        ambiguous_margin: float | None = None,
        minimum_clips: int = 3,
        minimum_total_speech_ms: int = 15_000,
    ) -> None:
        self._model_version = embedding_model_version
        self.accept_threshold = accept_threshold
        self.ambiguous_margin = ambiguous_margin
        self.minimum_clips = minimum_clips
        self.minimum_total_speech_ms = minimum_total_speech_ms
        self._tenants: dict[str, dict[str, _RegistryEntry]] = {}

    @property
    def embedding_model_version(self) -> str:
        return self._model_version

    @property
    def is_calibrated(self) -> bool:
        return self.accept_threshold is not None and self.ambiguous_margin is not None

    def enroll(
        self,
        tenant_id: str,
        identity_id: str,
        embeddings: list[SpeakerEmbedding],
        ctx: CallContext,
        *,
        display_name: str | None = None,
        consent_ref: str | None = None,
    ) -> EnrollmentReport:
        ctx.check()
        clips: list[EnrollmentClipReport] = []
        accepted: list[SpeakerEmbedding] = []
        for index, embedding in enumerate(embeddings):
            reasons: list[str] = []
            if embedding.model_version != self._model_version:
                reasons.append("embedding_model_mismatch")
            if embedding.speech_duration_ms < 3_000:
                reasons.append("clip_shorter_than_3s")
            if embedding.speech_duration_ms > 15_000:
                reasons.append("clip_longer_than_15s")
            if embedding.origin == "separated":
                reasons.append("separated_source_not_allowed")
            ok = not reasons
            clips.append(
                EnrollmentClipReport(
                    clip_index=index,
                    accepted=ok,
                    speech_duration_ms=embedding.speech_duration_ms,
                    reasons=tuple(reasons),
                )
            )
            if ok:
                accepted.append(embedding)

        total_speech = sum(e.speech_duration_ms for e in accepted)
        policy_reasons: list[str] = []
        if len(accepted) < self.minimum_clips:
            policy_reasons.append("fewer_than_minimum_clips")
        if total_speech < self.minimum_total_speech_ms:
            policy_reasons.append("insufficient_total_speech")
        meets_policy = not policy_reasons

        entry = self._tenants.setdefault(tenant_id, {}).setdefault(
            identity_id,
            _RegistryEntry(identity_id=identity_id, display_name=display_name or identity_id),
        )
        entry.consent_ref = consent_ref or entry.consent_ref
        if display_name:
            entry.display_name = display_name
        # Spec 5.10: store several prototypes, not one averaged centroid.
        for embedding in accepted:
            entry.prototypes.append(SpeakerPrototype.from_embedding(identity_id, embedding))

        return EnrollmentReport(
            identity_id=identity_id,
            accepted_clips=len(accepted),
            rejected_clips=len(embeddings) - len(accepted),
            total_speech_ms=total_speech,
            prototype_count=len(entry.prototypes),
            embedding_model_version=self._model_version,
            meets_policy=meets_policy,
            clips=tuple(clips),
            reasons=tuple(policy_reasons),
        )

    def identify(
        self,
        tenant_id: str,
        embedding: SpeakerEmbedding,
        ctx: CallContext,
    ) -> VoiceIdDecision:
        ctx.check()
        if not self.is_calibrated:
            return VoiceIdDecision(status="uncalibrated", reason="thresholds_null")
        if embedding.model_version != self._model_version:
            return VoiceIdDecision(status="unknown", reason="embedding_model_mismatch")

        entries = self._tenants.get(tenant_id, {})
        scores: list[tuple[float, _RegistryEntry]] = []
        for entry in entries.values():
            if not entry.prototypes:
                continue
            scores.append((max(p.similarity(embedding) for p in entry.prototypes), entry))
        if not scores:
            return VoiceIdDecision(status="unknown", reason="empty_registry")

        scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_entry = scores[0]
        second = scores[1][0] if len(scores) > 1 else None
        margin = None if second is None else best_score - second

        assert self.accept_threshold is not None and self.ambiguous_margin is not None
        if best_score < self.accept_threshold:
            return VoiceIdDecision(
                status="unknown", best_score=best_score, margin=margin, reason="below_accept"
            )
        if margin is not None and margin < self.ambiguous_margin:
            return VoiceIdDecision(
                status="ambiguous", best_score=best_score, margin=margin, reason="low_margin"
            )
        return VoiceIdDecision(
            status="enrolled",
            registry_speaker_id=best_entry.identity_id,
            speaker_name=best_entry.display_name,
            best_score=best_score,
            margin=margin,
            reason="accept",
        )

    def delete_identity(self, tenant_id: str, identity_id: str, ctx: CallContext) -> bool:
        ctx.check()
        return self._tenants.get(tenant_id, {}).pop(identity_id, None) is not None

    def identity_exists(self, tenant_id: str, identity_id: str) -> bool:
        return identity_id in self._tenants.get(tenant_id, {})

    def prototypes_of(self, tenant_id: str, identity_id: str) -> list[SpeakerPrototype]:
        entries = self._tenants.get(tenant_id, {})
        if identity_id not in entries:
            raise TenantAccessDeniedError(
                f"identity {identity_id!r} is not visible to this tenant",
                details={"identity_id": identity_id},
            )
        return list(entries[identity_id].prototypes)


__all__ = [
    "CLUSTER_PREFIX",
    "EMBEDDING_DIMENSION",
    "FakeAudioDecoder",
    "FakeDiarizer",
    "FakeOverlapDetector",
    "FakeSpeakerEmbedder",
    "FakeSpeechRecognizer",
    "FakeSpeechSeparator",
    "FakeVoiceActivityDetector",
    "FakeVoiceRegistry",
]
