"""Overlap attribution uses what the diarizer already knows — spec 5.2, 5.8.

The separator returns waveforms, never identities, so something has to decide
which source is whom. Scoring every source against every session speaker asks
the embedding a question the diarization has already answered, and on a short
overlap crop the embedding answers it badly. These tests pin the narrowing, the
separate floor for comparing versus building a centroid, and the fact that the
riskier attribution path stays off unless it is configured on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sastt.application.offline_pipeline import OfflinePipeline, PipelineAdapters
from sastt.application.session_state import SessionSpeakerState
from sastt.application.source_linking import (
    REASON_LINKED_CONSTRAINED,
    link_sources,
    restrict_to_candidates,
)
from sastt.config import ConfigurationError, SasttConfig, load_config, load_linking_overlay
from sastt.domain.audio import TimeInterval
from sastt.domain.speakers import (
    DiarizationResult,
    SpeakerEmbedding,
    SpeakerPrototype,
    SpeakerTurn,
    l2_normalize,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEMO_THRESHOLDS = REPO_ROOT / "configs" / "linking-thresholds.demo.yaml"


def linking_config(**source_linking: Any) -> SasttConfig:
    settings: dict[str, Any] = {"accept_threshold": 0.55, "ambiguous_margin": 0.10}
    settings.update(source_linking)
    return load_config(
        CONFIG_PATH,
        environment="development",
        manifest_dir=None,
        overrides={"source_linking": settings},
    )


def embedding_of(vector: list[float], *, speech_ms: int = 2000) -> SpeakerEmbedding:
    return SpeakerEmbedding(
        vector=l2_normalize(np.array(vector, dtype=np.float32)),
        model_version="test@1",
        speech_duration_ms=speech_ms,
        quality=1.0,
        origin="separated",
    )


def prototype_of(key: str, vector: list[float]) -> SpeakerPrototype:
    return SpeakerPrototype.from_embedding(key, embedding_of(vector))


class TestRestrictToCandidates:
    def test_non_candidate_columns_are_unreachable(self) -> None:
        matrix = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)

        restricted = restrict_to_candidates(matrix, ["spk_a", "spk_b"], {"spk_b"})

        assert np.isneginf(restricted[:, 0]).all(), "a silent speaker must not be assignable"
        assert restricted[0, 1] == pytest.approx(0.1)

    def test_no_candidate_set_leaves_the_matrix_alone(self) -> None:
        matrix = np.array([[0.9, 0.1]], dtype=np.float64)

        assert np.array_equal(restrict_to_candidates(matrix, ["spk_a", "spk_b"], None), matrix)

    def test_an_unknown_candidate_set_is_treated_as_no_opinion(self) -> None:
        """Masking everything would turn silence-of-evidence into evidence-of-silence."""
        matrix = np.array([[0.9, 0.1]], dtype=np.float64)

        restricted = restrict_to_candidates(matrix, ["spk_a", "spk_b"], {"spk_absent"})

        assert np.array_equal(restricted, matrix)

    def test_the_input_matrix_is_not_mutated(self) -> None:
        matrix = np.array([[0.9, 0.1]], dtype=np.float64)

        restrict_to_candidates(matrix, ["spk_a", "spk_b"], {"spk_b"})

        assert matrix[0, 0] == pytest.approx(0.9)


class TestLinkingHonoursTheCandidateSet:
    def test_a_source_is_never_linked_to_a_silent_speaker(self) -> None:
        prototypes = [prototype_of("spk_a", [1.0, 0.0]), prototype_of("spk_b", [0.0, 1.0])]
        # The embedding looks exactly like spk_a, but the diarizer says only
        # spk_b is talking here.
        result = link_sources(
            [embedding_of([1.0, 0.0])],
            prototypes,
            linking_config().source_linking,
            candidate_keys={"spk_b"},
        )

        assert result.decisions[0].session_speaker_id != "spk_a"

    def test_without_the_restriction_the_same_input_links_to_spk_a(self) -> None:
        prototypes = [prototype_of("spk_a", [1.0, 0.0]), prototype_of("spk_b", [0.0, 1.0])]

        result = link_sources(
            [embedding_of([1.0, 0.0])], prototypes, linking_config().source_linking
        )

        assert result.decisions[0].session_speaker_id == "spk_a"


class TestConstrainedPermutation:
    """Two sources and exactly two active speakers is a permutation, not an open choice."""

    def _weak_pair(self) -> tuple[list[SpeakerEmbedding], list[SpeakerPrototype]]:
        prototypes = [
            prototype_of("spk_a", [1.0, 0.0, 0.0]),
            prototype_of("spk_b", [0.0, 1.0, 0.0]),
        ]
        # Most of the energy points away from both centroids, so every score sits
        # well below accept_threshold=0.55 while still ordering the two sources
        # differently — exactly the shape of a short separated overlap crop.
        weak = [embedding_of([0.30, 0.10, 1.0]), embedding_of([0.10, 0.30, 1.0])]
        return weak, prototypes

    def test_off_by_default_a_weak_score_stays_unknown(self) -> None:
        weak, prototypes = self._weak_pair()

        result = link_sources(
            weak, prototypes, linking_config().source_linking, candidate_keys={"spk_a", "spk_b"}
        )

        assert all(decision.session_speaker_id is None for decision in result.decisions)

    def test_when_enabled_the_permutation_is_accepted(self) -> None:
        weak, prototypes = self._weak_pair()

        result = link_sources(
            weak,
            prototypes,
            linking_config(short_source_policy="diarization_constrained").source_linking,
            candidate_keys={"spk_a", "spk_b"},
            constrained_permutation=True,
        )

        assigned = [decision.session_speaker_id for decision in result.decisions]
        assert set(assigned) == {"spk_a", "spk_b"}, "one source each, no sharing"
        assert all(d.reason == REASON_LINKED_CONSTRAINED for d in result.decisions)

    def test_a_source_without_an_embedding_is_still_unknown(self) -> None:
        _weak, prototypes = self._weak_pair()

        result = link_sources(
            [embedding_of([0.30, 0.10, 1.0]), None],
            prototypes,
            linking_config(short_source_policy="diarization_constrained").source_linking,
            candidate_keys={"spk_a", "spk_b"},
            constrained_permutation=True,
        )

        assert result.decisions[1].session_speaker_id is None


class TestLinkingMinimumIsSeparateFromPrototypeMinimum:
    def _pipeline(self, config: SasttConfig) -> OfflinePipeline:
        return OfflinePipeline(config, PipelineAdapters.__new__(PipelineAdapters))

    def test_it_defaults_to_the_prototype_floor(self) -> None:
        pipeline = self._pipeline(linking_config())

        assert pipeline.linking_minimum_speech_ms == 1500

    def test_an_explicit_value_wins(self) -> None:
        pipeline = self._pipeline(linking_config(min_embedding_ms=400))

        assert pipeline.linking_minimum_speech_ms == 400

    def test_building_a_centroid_keeps_its_own_floor(self) -> None:
        """Lowering the comparison floor must not lower the bar for new centroids."""
        config = linking_config(min_embedding_ms=400)

        assert config.speaker_embedding.minimum_clean_speech_seconds == 1.5


class TestActiveSpeakers:
    def _state(self, config: SasttConfig) -> SessionSpeakerState:
        return SessionSpeakerState(
            session_id="ses_test", config=config, embedding_model_version="test@1"
        )

    def _diarization(self) -> DiarizationResult:
        turns = [
            SpeakerTurn(cluster_id="cluster_a", interval=TimeInterval(0, 5_000)),
            SpeakerTurn(cluster_id="cluster_b", interval=TimeInterval(4_000, 9_000)),
            SpeakerTurn(cluster_id="cluster_c", interval=TimeInterval(20_000, 25_000)),
        ]
        return DiarizationResult(
            turns=turns,
            regular_tracks=turns,
            exclusive_tracks=None,
            overlap_regions=[],
            estimated_session_speakers=3,
            model_version="test@1",
        )

    def test_only_speakers_active_in_the_window_are_candidates(self) -> None:
        config = linking_config()
        state = self._state(config)
        speakers = {
            c: state.ensure_cluster_speaker(c) for c in ("cluster_a", "cluster_b", "cluster_c")
        }
        pipeline = OfflinePipeline(config, PipelineAdapters.__new__(PipelineAdapters))

        keys, cluster_backed = pipeline._active_speakers(
            self._diarization(), state, TimeInterval(4_200, 4_800)
        )

        assert keys == {
            speakers["cluster_a"].session_speaker_id,
            speakers["cluster_b"].session_speaker_id,
        }
        assert cluster_backed == 2
        assert speakers["cluster_c"].session_speaker_id not in keys

    def test_temporary_speakers_stay_candidates_but_are_not_counted(self) -> None:
        """Diarization knows nothing about temporary identities, so it cannot rule
        them out; counting them would break the permutation arithmetic."""
        config = linking_config()
        state = self._state(config)
        state.ensure_cluster_speaker("cluster_a")
        state.ensure_cluster_speaker("cluster_b")
        temporary = state.create_temporary_speaker()
        pipeline = OfflinePipeline(config, PipelineAdapters.__new__(PipelineAdapters))

        keys, cluster_backed = pipeline._active_speakers(
            self._diarization(), state, TimeInterval(4_200, 4_800)
        )

        assert keys is not None
        assert temporary.session_speaker_id in keys
        assert cluster_backed == 2

    def test_the_restriction_can_be_switched_off(self) -> None:
        config = linking_config(restrict_to_active_clusters=False)
        state = self._state(config)
        state.ensure_cluster_speaker("cluster_a")
        pipeline = OfflinePipeline(config, PipelineAdapters.__new__(PipelineAdapters))

        keys, cluster_backed = pipeline._active_speakers(
            self._diarization(), state, TimeInterval(4_200, 4_800)
        )

        assert keys is None
        assert cluster_backed == 0

    def test_no_diarization_means_no_opinion(self) -> None:
        config = linking_config()
        pipeline = OfflinePipeline(config, PipelineAdapters.__new__(PipelineAdapters))

        keys, cluster_backed = pipeline._active_speakers(
            None, self._state(config), TimeInterval(0, 1_000)
        )

        assert keys is None
        assert cluster_backed == 0


class TestLinkingThresholdOverlay:
    def test_the_demo_overlay_supplies_both_thresholds(self) -> None:
        overlay = load_linking_overlay(DEMO_THRESHOLDS)

        assert overlay == {"source_linking": {"accept_threshold": 0.55, "ambiguous_margin": 0.10}}

    def test_nothing_configured_leaves_the_pipeline_failing_closed(self) -> None:
        assert load_linking_overlay(None) == {}

    def test_a_missing_file_is_refused_rather_than_ignored(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_linking_overlay(REPO_ROOT / "configs" / "does-not-exist.yaml")

    def test_a_file_without_thresholds_is_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("release_id: nothing\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="no source_linking section"):
            load_linking_overlay(empty)

    def test_the_shipped_config_still_fails_closed(self) -> None:
        config = load_config(CONFIG_PATH, environment="development", manifest_dir=None)

        assert config.source_linking.accept_threshold is None
        assert not config.source_linking.is_calibrated

    def test_the_risky_path_is_off_in_the_shipped_config(self) -> None:
        config = load_config(CONFIG_PATH, environment="development", manifest_dir=None)

        assert config.source_linking.short_source_policy == "unknown"
        assert config.source_linking.min_embedding_ms is None
