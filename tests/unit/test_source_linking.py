"""Hungarian permutation linking — spec 5.8, 16.3."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import sastt.application.source_linking as linking_module
from sastt.application.source_linking import (
    REASON_ASSIGNED_DUMMY,
    REASON_CONTESTED_IDENTITY,
    REASON_LOW_MARGIN,
    REASON_NO_EMBEDDING,
    REASON_UNCALIBRATED,
    build_score_matrix,
    link_sources,
)
from sastt.config import SourceLinkingConfig
from sastt.domain.speakers import SpeakerEmbedding, SpeakerPrototype, l2_normalize

pytestmark = pytest.mark.unit

MODEL = "fake-embedder@1"
CALIBRATED = SourceLinkingConfig(
    accept_threshold=0.55, ambiguous_margin=0.10, continuity_bonus=0.02
)
UNCALIBRATED = SourceLinkingConfig()


def basis(index: int, dimension: int = 16) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def embedding(vector: np.ndarray, *, quality: float = 1.0) -> SpeakerEmbedding:
    return SpeakerEmbedding(
        vector=l2_normalize(vector.astype(np.float32)),
        model_version=MODEL,
        quality=quality,
        speech_duration_ms=3000,
        origin="separated",
    )


def prototype(key: str, vector: np.ndarray) -> SpeakerPrototype:
    return SpeakerPrototype(
        speaker_key=key,
        centroid=l2_normalize(vector.astype(np.float32)),
        weight_sum=1.0,
        model_version=MODEL,
    )


class TestAssignment:
    def test_one_to_one_assignment(self) -> None:
        prototypes = [prototype("spk_1", basis(0)), prototype("spk_2", basis(1))]
        sources = [embedding(basis(1)), embedding(basis(0))]
        result = link_sources(sources, prototypes, CALIBRATED)
        assert [d.session_speaker_id for d in result.decisions] == ["spk_2", "spk_1"]
        assert all(d.status == "linked" for d in result.decisions)

    def test_source_order_swap_does_not_swap_speakers(self) -> None:
        """S03: the separator may return the speakers in any order."""
        prototypes = [prototype("spk_1", basis(0)), prototype("spk_2", basis(1))]
        first = link_sources([embedding(basis(0)), embedding(basis(1))], prototypes, CALIBRATED)
        second = link_sources([embedding(basis(1)), embedding(basis(0))], prototypes, CALIBRATED)
        assert first.mapping() == {0: "spk_1", 1: "spk_2"}
        assert second.mapping() == {0: "spk_2", 1: "spk_1"}

    def test_two_sources_cannot_share_one_identity(self) -> None:
        prototypes = [prototype("spk_1", basis(0)), prototype("spk_2", basis(1))]
        near = np.array([1.0, 0.05] + [0.0] * 14, dtype=np.float32)
        result = link_sources([embedding(near), embedding(near)], prototypes, CALIBRATED)
        assigned = [d.session_speaker_id for d in result.decisions if d.session_speaker_id]
        assert len(assigned) == len(set(assigned))

    def test_score_matrix_shape_and_values(self) -> None:
        prototypes = [prototype("spk_1", basis(0)), prototype("spk_2", basis(1))]
        matrix = build_score_matrix([embedding(basis(0)), None], prototypes)
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == pytest.approx(1.0)
        assert np.isneginf(matrix[1]).all()


class TestOpenSetDecisions:
    def test_uncalibrated_thresholds_fail_closed(self) -> None:
        prototypes = [prototype("spk_1", basis(0))]
        result = link_sources([embedding(basis(0))], prototypes, UNCALIBRATED)
        decision = result.decisions[0]
        assert decision.status == "uncalibrated"
        assert decision.session_speaker_id is None
        assert decision.reason == REASON_UNCALIBRATED
        assert result.calibrated is False

    def test_unknown_speaker_is_rejected_to_the_dummy_column(self) -> None:
        prototypes = [prototype("spk_1", basis(0)), prototype("spk_2", basis(1))]
        result = link_sources([embedding(basis(5))], prototypes, CALIBRATED)
        decision = result.decisions[0]
        assert decision.session_speaker_id is None
        assert decision.status == "unknown"
        assert decision.reason == REASON_ASSIGNED_DUMMY

    def test_low_margin_is_ambiguous_not_a_guess(self) -> None:
        prototypes = [
            prototype("spk_1", np.array([1.0, 0.0] + [0.0] * 14, dtype=np.float32)),
            prototype("spk_2", np.array([0.99, 0.14] + [0.0] * 14, dtype=np.float32)),
        ]
        result = link_sources([embedding(basis(0))], prototypes, CALIBRATED)
        decision = result.decisions[0]
        assert decision.status == "ambiguous"
        assert decision.reason in (REASON_LOW_MARGIN, REASON_CONTESTED_IDENTITY)
        assert decision.session_speaker_id is None

    def test_missing_embedding_never_links(self) -> None:
        prototypes = [prototype("spk_1", basis(0))]
        result = link_sources([None], prototypes, CALIBRATED)
        assert result.decisions[0].reason == REASON_NO_EMBEDDING
        assert result.decisions[0].session_speaker_id is None

    def test_no_prototypes_yields_unknown(self) -> None:
        result = link_sources([embedding(basis(0))], [], CALIBRATED)
        assert result.decisions[0].session_speaker_id is None


class TestContinuityAndScale:
    def test_continuity_bonus_only_breaks_ties(self) -> None:
        tie = np.array([1.0, 1.0] + [0.0] * 14, dtype=np.float32)
        prototypes = [prototype("spk_1", basis(0)), prototype("spk_2", basis(1))]
        result = link_sources(
            [embedding(tie), embedding(basis(1))],
            prototypes,
            SourceLinkingConfig(accept_threshold=0.5, ambiguous_margin=0.0, continuity_bonus=0.03),
            previous_mapping={0: "spk_1"},
        )
        assert result.mapping()[0] == "spk_1"

    def test_five_sources_do_not_enumerate_permutations(self) -> None:
        """Spec 5.8/16.3: no factorial path once K can exceed 3."""
        prototypes = [prototype(f"spk_{i}", basis(i)) for i in range(5)]
        sources = [embedding(basis(i)) for i in (4, 2, 0, 3, 1)]
        result = link_sources(sources, prototypes, CALIBRATED)
        assert result.mapping() == {0: "spk_4", 1: "spk_2", 2: "spk_0", 3: "spk_3", 4: "spk_1"}

    def test_module_contains_no_permutation_enumeration(self) -> None:
        source = inspect.getsource(linking_module)
        assert "itertools.permutations" not in source
        assert "permutations(" not in source
