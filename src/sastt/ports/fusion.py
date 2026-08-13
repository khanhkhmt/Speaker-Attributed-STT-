"""Confidence calibration and fusion ports — spec 5.11, 9."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sastt.domain.speakers import DiarizationResult
from sastt.domain.transcript import Confidences, TranscriptSegment
from sastt.observability import CallContext


@runtime_checkable
class ConfidenceCalibrator(Protocol):
    """Maps raw component scores onto calibrated confidences.

    Until a calibrator release exists the implementation returns
    ``Confidences(status="uncalibrated")`` with every field ``None``; the
    pipeline MUST NOT invent probability-looking numbers (spec 0.3).
    """

    @property
    def calibration_version(self) -> str | None: ...

    def calibrate(self, raw_scores: dict[str, float]) -> Confidences: ...


@runtime_checkable
class FusionEngine(Protocol):
    """Word/turn fusion into the public output contract (spec 5.11, 7)."""

    def fuse(
        self,
        diarization: DiarizationResult,
        ctx: CallContext,
    ) -> list[TranscriptSegment]: ...


__all__ = ["ConfidenceCalibrator", "FusionEngine"]
