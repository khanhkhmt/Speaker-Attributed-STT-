"""3D-Speaker CAM++ embedding adapter — spec 5.6, 9.

CAM++ 16 kHz is the default embedder (spec 0.2). Every embedding is taken from
VAD-ed speech, needs the configured minimum of clean speech, is L2-normalised
and carries its model version — embeddings of different versions are never
compared (spec 5.6).

The quality score ``q`` feeds the quality-weighted centroid of spec 5.6; it is a
diagnostic derived from duration, level and clipping, not a probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sastt.domain.audio import AudioBuffer, FloatArray, TimeInterval, total_duration_ms
from sastt.domain.errors import (
    InsufficientSpeechForEmbeddingError,
    ModelNotReadyError,
    SasttError,
)
from sastt.domain.speakers import EmbeddingOrigin, SpeakerEmbedding, l2_normalize
from sastt.observability import CallContext

FEATURE_DIM = 80
EMBEDDING_DIM = 192
CLIP_THRESHOLD = 0.99


class CamPlusPlusEmbedder:
    """``SpeakerEmbedder`` port backed by the 3D-Speaker CAM++ checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_version: str,
        device: str = "cuda:0",
        minimum_speech_ms: int = 1500,
        target_speech_ms: int = 3000,
        checkpoint_name: str = "campplus_cn_en_common.pt",
    ) -> None:
        self.model_path = Path(model_path)
        self._model_version = model_version
        self.minimum_speech_ms = minimum_speech_ms
        self.target_speech_ms = target_speech_ms

        checkpoint = self.model_path / checkpoint_name
        if not checkpoint.exists():
            candidates = sorted(self.model_path.glob("*.pt")) + sorted(
                self.model_path.glob("*.bin")
            )
            if not candidates:
                raise ModelNotReadyError(
                    f"CAM++ weights are not staged at {self.model_path} (spec 11.2)",
                    details={"model_path": str(self.model_path)},
                )
            checkpoint = candidates[0]

        try:
            import torch
            import torchaudio  # noqa: F401  (imported for the fbank frontend)
            from modelscope.models.audio.sv.DTDNN import CAMPPlus
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ModelNotReadyError(
                "the 3D-Speaker CAM++ backend needs torch, torchaudio and modelscope"
            ) from exc

        self._torch = torch
        self.device = (
            device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        )
        try:
            model = CAMPPlus(feat_dim=FEATURE_DIM, embedding_size=EMBEDDING_DIM)
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
            model.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001 - map to a domain error
            raise ModelNotReadyError(
                f"could not load CAM++ from {checkpoint}: {exc}",
                details={"checkpoint": str(checkpoint)},
            ) from exc
        self._model = model

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

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
                f"{speech_ms} ms of clean speech is below the {self.minimum_speech_ms} ms minimum",
                details={"speech_ms": speech_ms, "minimum_ms": self.minimum_speech_ms},
            )

        waveform = _speech_only(buffer, speech_intervals)
        if waveform.size < buffer.sample_rate // 10:
            raise InsufficientSpeechForEmbeddingError("less than 100 ms of samples to embed")

        try:
            import torchaudio

            torch = self._torch
            tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
            features = torchaudio.compliance.kaldi.fbank(
                tensor,
                num_mel_bins=FEATURE_DIM,
                sample_frequency=buffer.sample_rate,
                dither=0.0,
            )
            features = features - features.mean(dim=0, keepdim=True)  # cepstral mean norm
            with torch.no_grad():
                vector = self._model(features.unsqueeze(0).to(self.device))
            embedding = vector.squeeze(0).detach().cpu().numpy().astype(np.float32)
        except SasttError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak a backend exception
            raise SasttError(f"embedding failed: {exc}") from exc

        return SpeakerEmbedding(
            vector=l2_normalize(embedding),
            model_version=self._model_version,
            quality=self._quality(waveform, speech_ms),
            speech_duration_ms=speech_ms,
            origin=origin,
            interval=buffer.interval,
            source_track=source_track,
        )

    def _quality(self, waveform: FloatArray, speech_ms: int) -> float:
        """Duration, level and clipping — the ``q`` of the spec 5.6 centroid."""
        duration_score = min(1.0, speech_ms / max(1, self.target_speech_ms))
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        level_score = float(np.clip(rms / 0.05, 0.0, 1.0))
        clipping = float(np.mean(np.abs(waveform) >= CLIP_THRESHOLD))
        clipping_score = float(np.clip(1.0 - clipping * 20.0, 0.0, 1.0))
        return float(np.clip(duration_score * level_score * clipping_score, 0.0, 1.0))


def _speech_only(buffer: AudioBuffer, intervals: list[TimeInterval] | None) -> FloatArray:
    """Concatenate the VAD-ed speech; embeddings never see silence (spec 5.6)."""
    mono: FloatArray = buffer.to_mono().samples[0]
    if not intervals:
        return mono
    pieces: list[FloatArray] = []
    for interval in intervals:
        clipped = interval.clamp(buffer.interval)
        if clipped is None:
            continue
        start = int((clipped.start_ms - buffer.start_ms) * buffer.sample_rate / 1000)
        end = int((clipped.end_ms - buffer.start_ms) * buffer.sample_rate / 1000)
        piece = mono[max(0, start) : max(0, end)]
        if piece.size:
            pieces.append(piece)
    joined: FloatArray = np.concatenate(pieces) if pieces else mono
    return joined


__all__ = ["CamPlusPlusEmbedder"]
