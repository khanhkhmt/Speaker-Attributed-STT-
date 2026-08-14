"""SpeechBrain SepFormer Libri3Mix three-source adapter — M4 beta.

The checkpoint is 8 kHz, English read-speech oriented and intentionally never
downloaded at runtime.  It may only be wired when ``three_source_beta`` and a
pre-staged local model directory are both configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from sastt.domain.audio import AudioBuffer, FloatArray, ms_to_samples
from sastt.domain.errors import ModelNotReadyError, SeparationFailedError
from sastt.domain.speakers import SeparatedBatch, SourceQuality
from sastt.observability import CallContext

BACKEND = "sepformer_libri3mix"
NATIVE_SAMPLE_RATE = 8_000
SUPPORTED_SOURCE_COUNTS = (3,)
MIN_SOURCE_SPEECH_MS = 500


class SepFormerLibri3MixSeparator:
    """Three-source, 8 kHz SepFormer adapter used only by the M4 beta route."""

    def __init__(
        self, model_path: str | Path, *, separator_version: str, device: str = "cuda:0"
    ) -> None:
        self.model_path = Path(model_path)
        self._separator_version = separator_version
        if not self.model_path.is_dir():
            raise ModelNotReadyError(
                f"SepFormer weights are not staged at {self.model_path} (spec 11.2)",
                details={"model_path": str(self.model_path)},
            )
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:  # pragma: no cover - optional model dependency
            raise ModelNotReadyError("speechbrain is not installed for the SepFormer beta") from exc
        try:
            self._model: Any = SepformerSeparation.from_hparams(
                source=str(self.model_path),
                savedir=str(self.model_path),
                run_opts={"device": device},
            )
        except Exception as exc:  # noqa: BLE001 - map framework errors to domain error
            raise ModelNotReadyError(f"could not load SepFormer: {exc}") from exc

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
        self, buffer: AudioBuffer, ctx: CallContext, *, requested_source_count: int
    ) -> SeparatedBatch:
        ctx.check()
        if requested_source_count not in SUPPORTED_SOURCE_COUNTS:
            raise SeparationFailedError(f"{BACKEND} supports only K=3")
        mixture_16k = buffer.to_mono().samples[0].astype(np.float32)
        if mixture_16k.size < ms_to_samples(MIN_SOURCE_SPEECH_MS, buffer.sample_rate):
            raise SeparationFailedError("crop is too short to separate")
        mixture_8k = _resample(mixture_16k, buffer.sample_rate, NATIVE_SAMPLE_RATE)
        try:
            # SpeechBrain accepts a mono batch tensor/array and returns
            # [batch, time, speakers] for current SepFormer releases.
            output = self._model.separate_batch(mixture_8k[np.newaxis, :])
            sources_8k = _as_sources(output, requested_source_count)
        except Exception as exc:  # noqa: BLE001
            raise SeparationFailedError(f"SepFormer failed: {exc}") from exc
        sources = np.asarray(
            [_resample(source, NATIVE_SAMPLE_RATE, buffer.sample_rate) for source in sources_8k],
            dtype=np.float32,
        )
        sources = np.ascontiguousarray(sources[:, : mixture_16k.size])
        if sources.shape[1] < mixture_16k.size:
            sources = np.pad(sources, ((0, 0), (0, mixture_16k.size - sources.shape[1])))
        return SeparatedBatch(
            sources=sources,
            sample_rate=buffer.sample_rate,
            requested_source_count=requested_source_count,
            estimated_source_count=None,
            source_quality=_source_quality(mixture_16k, sources, buffer.sample_rate),
            separator_version=self._separator_version,
            start_sample=buffer.start_sample,
        )


def _resample(samples: FloatArray, source_rate: int, target_rate: int) -> FloatArray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    result: FloatArray = np.asarray(
        resample_poly(samples, target_rate, source_rate), dtype=np.float32
    )
    return result


def _as_sources(output: Any, count: int) -> FloatArray:
    value = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
    value = np.squeeze(value)
    if value.ndim != 2:
        raise SeparationFailedError(f"SepFormer returned unexpected shape {value.shape}")
    # Some releases return [time, speakers], others [speakers, time].
    sources = value.T if value.shape[-1] >= count and value.shape[0] > value.shape[-1] else value
    array = np.asarray(sources, dtype=np.float32)
    if array.shape[0] < count:
        raise SeparationFailedError(f"SepFormer returned only {array.shape[0]} source(s)")
    return np.ascontiguousarray(array[:count], dtype=np.float32)


def _source_quality(
    mixture: FloatArray, sources: FloatArray, sample_rate: int
) -> list[SourceQuality]:
    mixture_energy = float(np.sqrt(np.mean(np.square(mixture, dtype=np.float64)))) + 1e-9
    residual = mixture - sources.sum(axis=0)
    residual_ratio = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))) / mixture_energy)
    reports: list[SourceQuality] = []
    for index, source in enumerate(sources):
        energy = float(np.sqrt(np.mean(np.square(source, dtype=np.float64))))
        ratio = energy / mixture_energy
        others = np.delete(sources, index, axis=0)
        leakage = max(
            (
                abs(
                    float(
                        np.dot(source, other)
                        / ((np.linalg.norm(source) * np.linalg.norm(other)) + 1e-9)
                    )
                )
                for other in others
            ),
            default=0.0,
        )
        reports.append(
            SourceQuality(
                speech_duration_ms=int(source.size * 1000 / sample_rate),
                energy_ratio=ratio,
                leakage_similarity=leakage,
                residual_speech_ratio=residual_ratio,
                passed_gate=ratio >= 0.02,
                reasons=() if ratio >= 0.02 else ("low_energy",),
            )
        )
    return reports


__all__ = ["BACKEND", "NATIVE_SAMPLE_RATE", "SepFormerLibri3MixSeparator"]
