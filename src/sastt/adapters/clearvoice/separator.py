"""ClearerVoice MossFormer2 two-source separator — spec 5.4, 9.

``MossFormer2_SS_16K`` separates a mono mixture into K=2 waveforms at 16 kHz.
The separator returns **waveforms, never identities**: the source index is local
to this crop and MUST NOT be used as a cross-chunk identity (spec 5.4). Identity
comes later, from embeddings and Hungarian assignment (spec 5.8).

Separator confidence is not a probability the model provides, so it stays
``null``; what ships instead are the diagnostics of spec 5.4 — energy ratio,
leakage similarity against the other source, and residual speech level.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sastt.domain.audio import AudioBuffer, FloatArray, ms_to_samples
from sastt.domain.errors import ModelNotReadyError, SeparationFailedError
from sastt.domain.speakers import SeparatedBatch, SourceQuality
from sastt.observability import CallContext

BACKEND = "mossformer2_ss_16k"
NATIVE_SAMPLE_RATE = 16_000
SUPPORTED_SOURCE_COUNTS = (2,)
MIN_SOURCE_SPEECH_MS = 500
LOW_ENERGY_RATIO = 0.02


class MossFormer2Separator:
    """``SpeechSeparator`` port backed by ClearerVoice-Studio."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        separator_version: str,
        config_path: str | Path | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.model_path = Path(model_path)
        self._separator_version = separator_version
        if not (self.model_path / "last_best_checkpoint").exists():
            raise ModelNotReadyError(
                f"MossFormer2 weights are not staged at {self.model_path} (spec 11.2)",
                details={"model_path": str(self.model_path)},
            )
        try:
            import clearvoice
            from clearvoice.networks import CLS_MossFormer2_SS_16K
            from clearvoice.utils.decode import decode_one_audio_mossformer2_ss_16k
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ModelNotReadyError("clearvoice is not installed") from exc

        config_file = (
            Path(config_path)
            if config_path
            else (
                Path(clearvoice.__file__).parent
                / "config"
                / "inference"
                / "MossFormer2_SS_16K.yaml"
            )
        )
        if not config_file.exists():
            raise ModelNotReadyError(f"MossFormer2 inference config not found at {config_file}")

        settings: dict[str, Any] = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        settings["checkpoint_dir"] = str(self.model_path)
        self._args = types.SimpleNamespace(**settings)
        self._decode = decode_one_audio_mossformer2_ss_16k
        # clearvoice's SpeechModel picks "the GPU with the most free memory" and
        # calls torch.cuda.set_device() globally. On a multi-GPU host that both
        # strands the separator on a different device from the rest of the
        # pipeline and drags every adapter constructed afterwards onto that
        # device, which surfaces as "invalid device ordinal" or an out-of-memory
        # error once the two halves disagree. Spec 11.1 gives a worker one GPU, so
        # the choice is pinned here and the global device is put back.
        try:
            import torch

            previous = torch.cuda.current_device() if torch.cuda.is_available() else None
            try:
                self._wrapper = CLS_MossFormer2_SS_16K(self._args)
            finally:
                if previous is not None:
                    torch.cuda.set_device(previous)
            if torch.cuda.is_available():
                target = torch.device(device)
                self._wrapper.model.to(target)
                self._wrapper.device = target
        except Exception as exc:  # noqa: BLE001 - map to a domain error
            raise ModelNotReadyError(f"could not load MossFormer2: {exc}") from exc

    # -- port ---------------------------------------------------------------- #

    @property
    def backend(self) -> str:
        return BACKEND

    @property
    def separator_version(self) -> str:
        return self._separator_version

    @property
    def sample_rate(self) -> int:
        return NATIVE_SAMPLE_RATE

    @property
    def supported_source_counts(self) -> tuple[int, ...]:
        return SUPPORTED_SOURCE_COUNTS

    def separate(
        self,
        buffer: AudioBuffer,
        ctx: CallContext,
        *,
        requested_source_count: int,
    ) -> SeparatedBatch:
        ctx.check()
        if requested_source_count not in SUPPORTED_SOURCE_COUNTS:
            raise SeparationFailedError(
                f"{BACKEND} supports K={SUPPORTED_SOURCE_COUNTS}, got {requested_source_count}",
                details={"requested_source_count": requested_source_count},
            )
        if buffer.sample_rate != NATIVE_SAMPLE_RATE:
            raise SeparationFailedError(
                f"{BACKEND} expects {NATIVE_SAMPLE_RATE} Hz, got {buffer.sample_rate}"
            )

        mixture = buffer.to_mono().samples[0].astype(np.float32)
        if mixture.size < ms_to_samples(MIN_SOURCE_SPEECH_MS, buffer.sample_rate):
            raise SeparationFailedError("crop is too short to separate")

        try:
            outputs = self._decode(
                self._wrapper.model, self._wrapper.device, mixture[np.newaxis, :], self._args
            )
        except Exception as exc:  # noqa: BLE001 - OOM/model failure -> domain error
            raise SeparationFailedError(
                f"MossFormer2 failed: {exc}", details={"samples": int(mixture.size)}
            ) from exc

        sources = np.asarray(
            [np.asarray(source, dtype=np.float32).squeeze()[: mixture.size] for source in outputs],
            dtype=np.float32,
        )
        if sources.ndim != 2 or sources.shape[0] < requested_source_count:
            raise SeparationFailedError(f"separator returned an unexpected shape {sources.shape}")
        sources = sources[:requested_source_count]

        return SeparatedBatch(
            sources=np.ascontiguousarray(sources),
            sample_rate=buffer.sample_rate,
            requested_source_count=requested_source_count,
            # MossFormer2 is a fixed two-source model: it cannot count sources,
            # so the estimate stays null rather than echoing the request (spec 5.3).
            estimated_source_count=None,
            source_quality=_source_quality(mixture, sources, buffer.sample_rate),
            separator_version=self._separator_version,
            start_sample=buffer.start_sample,
        )


def _source_quality(
    mixture: FloatArray, sources: FloatArray, sample_rate: int
) -> list[SourceQuality]:
    """Diagnostics of spec 5.4 — no invented probability."""
    mixture_energy = float(np.sqrt(np.mean(np.square(mixture, dtype=np.float64)))) + 1e-9
    residual = mixture - sources.sum(axis=0)
    residual_ratio = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))) / mixture_energy)

    quality: list[SourceQuality] = []
    for index, source in enumerate(sources):
        energy = float(np.sqrt(np.mean(np.square(source, dtype=np.float64))))
        energy_ratio = energy / mixture_energy
        others = np.delete(sources, index, axis=0)
        leakage = 0.0
        if others.size:
            for other in others:
                denominator = (np.linalg.norm(source) * np.linalg.norm(other)) + 1e-9
                leakage = max(leakage, abs(float(np.dot(source, other) / denominator)))
        speech_ms = int(source.size * 1000 / sample_rate)
        reasons: list[str] = []
        if energy_ratio < LOW_ENERGY_RATIO:
            reasons.append("low_energy")
        if speech_ms < MIN_SOURCE_SPEECH_MS:
            reasons.append("shorter_than_minimum")
        quality.append(
            SourceQuality(
                speech_duration_ms=speech_ms,
                energy_ratio=energy_ratio,
                leakage_similarity=leakage,
                residual_speech_ratio=residual_ratio,
                snr_db=None,
                passed_gate=not reasons,
                reasons=tuple(reasons),
            )
        )
    return quality


__all__ = ["MossFormer2Separator"]
