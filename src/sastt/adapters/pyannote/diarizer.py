"""pyannote diarization and overlap detection — spec 5.2, 9, 20.

``speaker-diarization-community-1`` produces the global turns and the 1–5
speaker count. Its **regular** tracks keep concurrent speakers and are the
source of truth; the **exclusive** tracks assign each frame to one speaker and
may only be used to align words outside overlap — never to delete a second
track inside overlap (spec 5.2).

``segmentation-3.0`` backs overlap detection. It models at most two active
speakers per frame, so a positive region means ``K > 1`` and nothing more; it
MUST NOT be used to count three to five sources (spec 5.2, 20).

Both checkpoints are gated: the terms must be accepted on the model page and the
attribution recorded in the manifest before use (spec 20).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sastt.domain.audio import AudioBuffer, TimeInterval, merge_intervals
from sastt.domain.errors import ModelNotReadyError, SasttError
from sastt.domain.speakers import DiarizationResult, OverlapRegion, SpeakerTurn
from sastt.observability import CallContext


def _load_pipeline(model_path: Path, what: str) -> Any:
    if not model_path.exists():
        raise ModelNotReadyError(
            f"{what} weights are not staged at {model_path}. The pyannote checkpoints are "
            "gated: accept the conditions on the model page, then run "
            "deploy/prestage_models.py (spec 11.2, 20).",
            details={"model_path": str(model_path)},
        )
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ModelNotReadyError("pyannote.audio is not installed") from exc

    config = model_path / "config.yaml"
    try:
        pipeline = Pipeline.from_pretrained(str(config if config.exists() else model_path))
    except Exception as exc:  # noqa: BLE001 - map to a domain error
        raise ModelNotReadyError(f"could not load {what}: {exc}") from exc
    if pipeline is None:
        raise ModelNotReadyError(f"{what} pipeline could not be constructed from {model_path}")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def _as_torch_input(buffer: AudioBuffer) -> dict[str, Any]:
    import torch

    mono = buffer.to_mono().samples
    return {
        "waveform": torch.from_numpy(np.ascontiguousarray(mono)),
        "sample_rate": buffer.sample_rate,
    }


class PyannoteDiarizer:
    """``Diarizer`` port — spec 5.2."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_version: str,
        exclusive_tracks: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self._model_version = model_version
        self.exclusive_tracks = exclusive_tracks
        self._pipeline = _load_pipeline(self.model_path, "diarization")

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
        try:
            output = self._pipeline(
                _as_torch_input(buffer), min_speakers=min_speakers, max_speakers=max_speakers
            )
        except Exception as exc:  # noqa: BLE001 - never leak a backend exception
            raise SasttError(f"diarization failed: {exc}") from exc

        offset = buffer.start_ms
        regular = _turns_from_annotation(_speaker_diarization(output), offset, "regular")
        exclusive: list[SpeakerTurn] | None = None
        if self.exclusive_tracks:
            exclusive_annotation = _exclusive_diarization(output)
            if exclusive_annotation is not None:
                exclusive = _turns_from_annotation(exclusive_annotation, offset, "exclusive")

        overlaps = _overlap_regions_from_turns(regular, self._model_version)
        speakers = {turn.cluster_id for turn in regular}
        return DiarizationResult(
            turns=regular,
            regular_tracks=regular,
            exclusive_tracks=exclusive,
            overlap_regions=overlaps,
            estimated_session_speakers=min(max(len(speakers), min_speakers), max_speakers),
            model_version=self._model_version,
        )


class PyannoteOverlapDetector:
    """``OverlapDetector`` port — spec 5.2.

    A positive region only means ``K > 1``; segmentation-3.0 models at most two
    active speakers per frame and is never a 3–5 source counter.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_version: str,
        onset: float = 0.60,
        offset: float = 0.50,
        min_duration_ms: int = 300,
        merge_gap_ms: int = 200,
    ) -> None:
        self.model_path = Path(model_path)
        self._model_version = model_version
        self.onset = onset
        self.offset = offset
        self.min_duration_ms = min_duration_ms
        self.merge_gap_ms = merge_gap_ms

        if not self.model_path.exists():
            raise ModelNotReadyError(
                f"OSD weights are not staged at {self.model_path}; the segmentation-3.0 "
                "checkpoint is gated (spec 11.2, 20).",
                details={"model_path": str(self.model_path)},
            )
        try:
            import torch
            from pyannote.audio import Inference, Model
            from pyannote.audio.utils.powerset import Powerset
            from pyannote.audio.utils.signal import Binarize
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ModelNotReadyError("pyannote.audio is not installed") from exc

        try:
            model = Model.from_pretrained(str(self.model_path))
            if model is None:
                raise ModelNotReadyError(
                    f"segmentation model could not be constructed from {self.model_path}"
                )
            if torch.cuda.is_available():
                model.to(torch.device("cuda"))
            self._inference = Inference(model, skip_aggregation=False)
            specifications: Any = model.specifications
            self._powerset = Powerset(
                int(len(specifications.classes)), int(specifications.powerset_max_classes)
            )
            # Hysteresis of spec 5.2: onset 0.60 / offset 0.50, minimum duration
            # 0.30 s, merge gap 0.20 s. Seed values, not calibrated thresholds.
            self._binarize = Binarize(
                onset=onset,
                offset=offset,
                min_duration_on=min_duration_ms / 1000,
                min_duration_off=merge_gap_ms / 1000,
            )
        except ModelNotReadyError:
            raise
        except Exception as exc:  # noqa: BLE001 - map to a domain error
            raise ModelNotReadyError(f"could not load the OSD model: {exc}") from exc

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect(self, buffer: AudioBuffer, ctx: CallContext) -> list[OverlapRegion]:
        """Frames where more than one speaker is active.

        pyannote.audio 4 has no ``OverlappedSpeechDetection`` pipeline, so the
        segmentation model is run directly: its powerset scores are converted to
        per-speaker activations, the number of simultaneously active speakers is
        summed, and the ``>= 2`` curve is binarised with the spec 5.2 hysteresis.
        The result only means ``K > 1`` — segmentation-3.0 caps at two active
        speakers per frame and can never count three to five.
        """
        ctx.check()
        try:
            import torch
            from pyannote.core import SlidingWindowFeature

            scores: Any = self._inference(_as_torch_input(buffer))
            powerset = torch.from_numpy(scores.data).unsqueeze(0)
            multilabel = self._powerset.to_multilabel(powerset, soft=True)
            overlap = multilabel.sum(dim=-1).squeeze(0).clamp(0.0, 2.0) - 1.0
            curve = SlidingWindowFeature(
                overlap.clamp(0.0, 1.0).cpu().numpy().reshape(-1, 1), scores.sliding_window
            )
            timeline = self._binarize(curve).get_timeline().support()
        except Exception as exc:  # noqa: BLE001 - never leak a backend exception
            raise SasttError(f"overlap detection failed: {exc}") from exc

        origin = buffer.start_ms
        intervals = [
            TimeInterval(
                origin + int(segment.start * 1000), origin + max(1, int(segment.end * 1000))
            )
            for segment in timeline
        ]
        return [
            OverlapRegion(
                interval=interval,
                # A raw activation would need a calibration release before it could
                # be published as a score, so it stays null (spec 0.3, 5.2).
                osd_activation=None,
                model_version=self._model_version,
            )
            for interval in merge_intervals(intervals, merge_gap_ms=self.merge_gap_ms)
            if interval.duration_ms >= self.min_duration_ms
        ]


# --------------------------------------------------------------------------- #
# pyannote output helpers
# --------------------------------------------------------------------------- #


def _speaker_diarization(output: Any) -> Any:
    """The overlap-preserving annotation (community-1 returns a rich object)."""
    for attribute in ("speaker_diarization", "regular"):
        candidate = getattr(output, attribute, None)
        if candidate is not None:
            return candidate
    return output


def _exclusive_diarization(output: Any) -> Any | None:
    return getattr(output, "exclusive_speaker_diarization", None) or getattr(
        output, "exclusive", None
    )


def _turns_from_annotation(annotation: Any, offset_ms: int, kind: str) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        start = offset_ms + int(segment.start * 1000)
        end = offset_ms + int(segment.end * 1000)
        if end <= start:
            continue
        turns.append(
            SpeakerTurn(
                cluster_id=f"cluster_{label}",
                interval=TimeInterval(start, end),
                kind="regular" if kind == "regular" else "exclusive",
            )
        )
    return sorted(turns, key=lambda turn: (turn.start_ms, turn.cluster_id))


def _overlap_regions_from_turns(
    turns: list[SpeakerTurn], model_version: str
) -> list[OverlapRegion]:
    """Regions where two regular tracks are simultaneously active (spec 5.2)."""
    edges = sorted({ms for turn in turns for ms in (turn.start_ms, turn.end_ms)})
    hits: list[TimeInterval] = []
    for start, end in zip(edges, edges[1:], strict=False):
        if end <= start:
            continue
        slot = TimeInterval(start, end)
        active = sum(1 for turn in turns if turn.interval.intersection_ms(slot) > 0)
        if active >= 2:
            hits.append(slot)
    return [
        OverlapRegion(interval=interval, osd_activation=None, model_version=model_version)
        for interval in merge_intervals(hits)
    ]


__all__ = ["PyannoteDiarizer", "PyannoteOverlapDetector"]
