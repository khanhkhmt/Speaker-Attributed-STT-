"""Permutation linking of separated sources — spec 5.8.

Hungarian assignment over a score matrix extended with dummy ``Unknown``
columns. Exhaustive permutation enumeration is deliberately absent: it is cubic
vs factorial, and spec 5.8/16.3 forbid a factorial path once ``K`` can exceed 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from sastt.config import SourceLinkingConfig
from sastt.domain.speakers import LinkingDecision, SpeakerEmbedding, SpeakerPrototype
from sastt.observability import (
    METRIC_SOURCE_LINK_UNKNOWN,
    CallContext,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

    ScoreMatrix = NDArray[np.float64]
else:  # pragma: no cover - runtime alias
    ScoreMatrix = np.ndarray

REASON_NO_EMBEDDING = "no_embedding"
REASON_UNCALIBRATED = "uncalibrated_threshold"
REASON_BELOW_ACCEPT = "below_accept_threshold"
REASON_LOW_MARGIN = "low_margin"
REASON_ASSIGNED_DUMMY = "assigned_to_unknown"
REASON_NO_PROTOTYPE = "no_prototype"
REASON_CONTESTED_IDENTITY = "contested_identity"
REASON_LINKED = "linked"


@dataclass(frozen=True)
class LinkingResult:
    """Decisions plus the diagnostics needed to explain them."""

    decisions: list[LinkingDecision]
    score_matrix: ScoreMatrix
    prototype_keys: list[str]
    calibrated: bool

    def mapping(self) -> dict[int, str]:
        return {
            d.source_track: d.session_speaker_id
            for d in self.decisions
            if d.status == "linked" and d.session_speaker_id is not None
        }


def build_score_matrix(
    embeddings: list[SpeakerEmbedding | None],
    prototypes: list[SpeakerPrototype],
) -> ScoreMatrix:
    """``S[i, j] = cosine(e_i, centroid_j)`` — spec 5.8 step 2.

    Rows without a usable embedding are filled with ``-inf`` so they can only be
    assigned to a dummy column.
    """
    matrix = np.full((len(embeddings), len(prototypes)), -np.inf, dtype=np.float64)
    for i, embedding in enumerate(embeddings):
        if embedding is None:
            continue
        for j, prototype in enumerate(prototypes):
            matrix[i, j] = prototype.similarity(embedding)
    return matrix


def _apply_continuity_bonus(
    matrix: ScoreMatrix,
    prototype_keys: list[str],
    previous_mapping: dict[int, str] | None,
    bonus: float,
) -> ScoreMatrix:
    """Spec 5.8 step 3: a small bonus when the previous mapping is not contradicted.

    The bonus is only a tie-breaker; the source index itself carries no identity
    (spec 5.8 last paragraph).
    """
    if not previous_mapping or bonus <= 0.0:
        return matrix
    boosted = matrix.copy()
    key_index = {key: idx for idx, key in enumerate(prototype_keys)}
    for source_track, speaker_key in previous_mapping.items():
        j = key_index.get(speaker_key)
        if j is None or source_track >= boosted.shape[0]:
            continue
        if np.isneginf(boosted[source_track, j]):
            continue
        boosted[source_track, j] += bonus
    return boosted


def link_sources(
    embeddings: list[SpeakerEmbedding | None],
    prototypes: list[SpeakerPrototype],
    config: SourceLinkingConfig,
    *,
    previous_mapping: dict[int, str] | None = None,
    ctx: CallContext | None = None,
) -> LinkingResult:
    """Assign each separated source to at most one session speaker — spec 5.8."""
    prototype_keys = [prototype.speaker_key for prototype in prototypes]
    base = build_score_matrix(embeddings, prototypes)
    calibrated = config.is_calibrated

    if not prototypes:
        rejected = [
            LinkingDecision(
                source_track=i,
                session_speaker_id=None,
                status="unknown",
                reason=REASON_NO_PROTOTYPE if embedding is not None else REASON_NO_EMBEDDING,
            )
            for i, embedding in enumerate(embeddings)
        ]
        _count_unknowns(rejected, ctx)
        return LinkingResult(rejected, base, prototype_keys, calibrated)

    scored = _apply_continuity_bonus(
        base, prototype_keys, previous_mapping, config.continuity_bonus
    )

    # Spec 5.8 step 4: one dummy Unknown column per source, so a source may reject.
    dummy_value = config.accept_threshold if config.accept_threshold is not None else 0.0
    dummy = np.full((scored.shape[0], scored.shape[0]), -np.inf, dtype=np.float64)
    np.fill_diagonal(dummy, dummy_value)
    extended = np.concatenate([scored, dummy], axis=1)
    # linear_sum_assignment cannot handle -inf; use a finite floor below every real score.
    finite = np.nanmin(extended[np.isfinite(extended)], initial=0.0) - 1.0
    solvable = np.where(np.isneginf(extended), finite, extended)

    rows, cols = linear_sum_assignment(solvable, maximize=True)
    assignment = dict(zip(rows.tolist(), cols.tolist(), strict=True))

    num_prototypes = len(prototypes)
    argmax_owner: dict[int, list[int]] = {}
    for i, embedding in enumerate(embeddings):
        if embedding is None:
            continue
        best_j = int(np.argmax(base[i]))
        if np.isfinite(base[i, best_j]):
            argmax_owner.setdefault(best_j, []).append(i)

    decisions: list[LinkingDecision] = []
    for i, embedding in enumerate(embeddings):
        if embedding is None:
            decisions.append(
                LinkingDecision(
                    source_track=i,
                    session_speaker_id=None,
                    status="unknown",
                    reason=REASON_NO_EMBEDDING,
                )
            )
            continue

        j = assignment.get(i, num_prototypes)
        row = base[i]
        top1, top2 = _top_two(row)
        margin = None if top2 is None else float(top1 - top2)

        if not calibrated:
            # Spec 12/18.7: thresholds are null until calibration -> fail closed.
            decisions.append(
                LinkingDecision(
                    source_track=i,
                    session_speaker_id=None,
                    status="uncalibrated",
                    score=float(top1) if np.isfinite(top1) else None,
                    margin=margin,
                    reason=REASON_UNCALIBRATED,
                )
            )
            continue

        if j >= num_prototypes:
            decisions.append(
                LinkingDecision(
                    source_track=i,
                    session_speaker_id=None,
                    status="unknown",
                    score=float(top1) if np.isfinite(top1) else None,
                    margin=margin,
                    reason=REASON_ASSIGNED_DUMMY,
                )
            )
            continue

        score = float(base[i, j])
        assert config.accept_threshold is not None and config.ambiguous_margin is not None
        if score < config.accept_threshold:
            decisions.append(
                LinkingDecision(
                    source_track=i,
                    session_speaker_id=None,
                    status="unknown",
                    score=score,
                    margin=margin,
                    reason=REASON_BELOW_ACCEPT,
                )
            )
            continue

        if margin is not None and margin < config.ambiguous_margin:
            decisions.append(
                LinkingDecision(
                    source_track=i,
                    session_speaker_id=None,
                    status="ambiguous",
                    score=score,
                    margin=margin,
                    reason=REASON_LOW_MARGIN,
                )
            )
            continue

        # Spec 5.8 step 7: when two sources are drawn to the same identity, the one
        # the assignment did not award it to is reported ambiguous rather than guessed.
        preferred = int(np.argmax(row))
        if len(argmax_owner.get(preferred, [])) > 1 and j != preferred:
            decisions.append(
                LinkingDecision(
                    source_track=i,
                    session_speaker_id=None,
                    status="ambiguous",
                    score=score,
                    margin=margin,
                    reason=REASON_CONTESTED_IDENTITY,
                )
            )
            continue

        decisions.append(
            LinkingDecision(
                source_track=i,
                session_speaker_id=prototype_keys[j],
                status="linked",
                score=score,
                margin=margin,
                reason=REASON_LINKED,
            )
        )

    _count_unknowns(decisions, ctx)
    return LinkingResult(decisions, base, prototype_keys, calibrated)


def _top_two(row: ScoreMatrix) -> tuple[float, float | None]:
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        return (-np.inf, None)
    ordered = np.sort(finite)[::-1]
    if ordered.size == 1:
        return (float(ordered[0]), None)
    return (float(ordered[0]), float(ordered[1]))


def _count_unknowns(decisions: list[LinkingDecision], ctx: CallContext | None) -> None:
    if ctx is None:
        return
    for decision in decisions:
        if decision.status != "linked":
            ctx.metrics.increment(
                METRIC_SOURCE_LINK_UNKNOWN, reason=decision.reason or decision.status
            )


class HungarianSourceLinker:
    """``SourceLinker`` port implementation (spec 5.8, 9)."""

    def __init__(self, config: SourceLinkingConfig) -> None:
        self.config = config

    def link(
        self,
        embeddings: list[SpeakerEmbedding | None],
        prototypes: list[SpeakerPrototype],
        ctx: CallContext,
        *,
        previous_mapping: dict[int, str] | None = None,
    ) -> list[LinkingDecision]:
        ctx.check()
        return link_sources(
            embeddings,
            prototypes,
            self.config,
            previous_mapping=previous_mapping,
            ctx=ctx,
        ).decisions


__all__ = [
    "REASON_ASSIGNED_DUMMY",
    "REASON_BELOW_ACCEPT",
    "REASON_CONTESTED_IDENTITY",
    "REASON_LINKED",
    "REASON_LOW_MARGIN",
    "REASON_NO_EMBEDDING",
    "REASON_NO_PROTOTYPE",
    "REASON_UNCALIBRATED",
    "HungarianSourceLinker",
    "LinkingResult",
    "build_score_matrix",
    "link_sources",
]
