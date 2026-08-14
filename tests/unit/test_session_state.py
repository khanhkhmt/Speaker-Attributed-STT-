"""Session speaker state, prototypes and merges — spec 5.6, 5.7, 5.9."""

from __future__ import annotations

import numpy as np
import pytest

from sastt.application.session_state import (
    REASON_LOW_LINK_MARGIN,
    REASON_SEPARATED_DISABLED,
    REASON_TOO_SHORT,
    SessionSpeakerState,
    overlapping_clusters,
)
from sastt.config import SasttConfig
from sastt.domain.audio import TimeInterval
from sastt.domain.errors import InvalidStateTransitionError
from sastt.domain.speakers import (
    EmbeddingOrigin,
    IdentityState,
    IdentityStatus,
    SpeakerEmbedding,
    VoiceIdDecision,
    l2_normalize,
)

pytestmark = pytest.mark.unit

MODEL = "fake-embedder@1"


def make_embedding(
    index: int,
    *,
    quality: float = 0.9,
    speech_ms: int = 3000,
    origin: EmbeddingOrigin = "clean",
    dimension: int = 16,
) -> SpeakerEmbedding:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return SpeakerEmbedding(
        vector=l2_normalize(vector),
        model_version=MODEL,
        quality=quality,
        speech_duration_ms=speech_ms,
        origin=origin,
    )


@pytest.fixture
def state(base_config: SasttConfig) -> SessionSpeakerState:
    return SessionSpeakerState(
        session_id="ses_test", config=base_config, embedding_model_version=MODEL
    )


class TestCreation:
    def test_cluster_speakers_are_reused(self, state: SessionSpeakerState) -> None:
        first = state.ensure_cluster_speaker("cluster_a")
        again = state.ensure_cluster_speaker("cluster_a")
        assert first.session_speaker_id == again.session_speaker_id
        assert first.display_label == "Speaker 1"

    def test_session_speaker_cap_is_five(self, state: SessionSpeakerState) -> None:
        for index in range(5):
            state.ensure_cluster_speaker(f"cluster_{index}")
        with pytest.raises(InvalidStateTransitionError):
            state.create_session_speaker(cluster_id="cluster_6")

    def test_temporary_speakers_start_provisional(self, state: SessionSpeakerState) -> None:
        temporary = state.create_temporary_speaker()
        assert temporary.state is IdentityState.PROVISIONAL
        assert temporary.display_label == "Temporary Speaker 1"
        assert temporary.identity_status is IdentityStatus.PROVISIONAL


class TestPrototypeGates:
    def test_short_speech_never_updates_a_centroid(self, state: SessionSpeakerState) -> None:
        speaker = state.ensure_cluster_speaker("cluster_a")
        update = state.update_prototype(
            speaker.session_speaker_id, make_embedding(0, speech_ms=1200)
        )
        assert update.accepted is False
        assert update.reason == REASON_TOO_SHORT
        assert speaker.prototype is None

    def test_separated_sources_are_disabled_by_default(self, state: SessionSpeakerState) -> None:
        speaker = state.ensure_cluster_speaker("cluster_a")
        state.update_prototype(speaker.session_speaker_id, make_embedding(0))
        update = state.update_prototype(
            speaker.session_speaker_id, make_embedding(0, origin="separated"), link_margin=0.5
        )
        assert update.reason == REASON_SEPARATED_DISABLED

    def test_separated_update_requires_margin_when_enabled(
        self, calibrated_config: SasttConfig
    ) -> None:
        config = calibrated_config.model_copy(
            update={
                "speaker_embedding": calibrated_config.speaker_embedding.model_copy(
                    update={"update_from_separated_sources": True}
                )
            }
        )
        state = SessionSpeakerState(session_id="ses", config=config, embedding_model_version=MODEL)
        speaker = state.ensure_cluster_speaker("cluster_a")
        state.update_prototype(speaker.session_speaker_id, make_embedding(0))
        low = state.update_prototype(
            speaker.session_speaker_id,
            make_embedding(0, origin="separated"),
            link_margin=0.01,
        )
        assert low.reason == REASON_LOW_LINK_MARGIN
        good = state.update_prototype(
            speaker.session_speaker_id,
            make_embedding(0, origin="separated"),
            link_margin=0.4,
        )
        assert good.accepted is True

    def test_model_version_mismatch_is_refused(self, state: SessionSpeakerState) -> None:
        speaker = state.ensure_cluster_speaker("cluster_a")
        other = SpeakerEmbedding(
            vector=l2_normalize(np.ones(16, dtype=np.float32)),
            model_version="other-model@2",
            quality=1.0,
            speech_duration_ms=4000,
            origin="clean",
        )
        assert state.update_prototype(speaker.session_speaker_id, other).accepted is False

    def test_updates_are_versioned_and_reversible(self, state: SessionSpeakerState) -> None:
        speaker = state.ensure_cluster_speaker("cluster_a")
        state.update_prototype(speaker.session_speaker_id, make_embedding(0))
        first = speaker.prototype
        state.update_prototype(speaker.session_speaker_id, make_embedding(0, quality=0.5))
        assert speaker.prototype is not None and speaker.prototype.version == 2
        speaker.rollback_prototype()
        assert speaker.prototype is first

    def test_quality_weighting_pulls_towards_the_better_embedding(
        self, state: SessionSpeakerState
    ) -> None:
        speaker = state.ensure_cluster_speaker("cluster_a")
        state.update_prototype(speaker.session_speaker_id, make_embedding(0, quality=0.2))
        state.update_prototype(speaker.session_speaker_id, make_embedding(1, quality=0.9))
        assert speaker.prototype is not None
        assert speaker.prototype.centroid[1] > speaker.prototype.centroid[0]

    def test_provisional_prototypes_are_not_linking_targets(
        self, state: SessionSpeakerState
    ) -> None:
        temporary = state.create_temporary_speaker()
        state.buffer_provisional_embedding(temporary.session_speaker_id, make_embedding(3))
        assert state.prototypes() == []
        assert len(state.provisional_prototypes()) == 1


class TestConstraintsAndMerges:
    def test_cannot_link_blocks_a_merge(self, state: SessionSpeakerState) -> None:
        first = state.ensure_cluster_speaker("cluster_a")
        second = state.ensure_cluster_speaker("cluster_b")
        state.add_cannot_link(first.session_speaker_id, second.session_speaker_id)
        assert state.can_merge(first.session_speaker_id, second.session_speaker_id) is False
        with pytest.raises(InvalidStateTransitionError):
            state.merge(first.session_speaker_id, second.session_speaker_id, "test")

    def test_merge_redirects_lookups_and_records_a_label_change(
        self, state: SessionSpeakerState
    ) -> None:
        target = state.ensure_cluster_speaker("cluster_a")
        temporary = state.create_temporary_speaker()
        change = state.merge(
            temporary.session_speaker_id, target.session_speaker_id, "reconciliation"
        )
        assert change is not None
        assert change.previous_label == "Temporary Speaker 1"
        assert change.new_label == target.display_label
        assert (
            state.get(temporary.session_speaker_id).session_speaker_id == target.session_speaker_id
        )
        assert temporary.state is IdentityState.MERGED
        assert state.active_speaker_count == 1

    def test_overlapping_clusters_are_detected(self) -> None:
        pairs = overlapping_clusters(
            {
                "a": [TimeInterval(0, 1000)],
                "b": [TimeInterval(900, 2000)],
                "c": [TimeInterval(5000, 6000)],
            }
        )
        assert pairs == {("a", "b")}


class TestIdentityResolution:
    def test_promotion_relabels_a_temporary_speaker(self, state: SessionSpeakerState) -> None:
        temporary = state.create_temporary_speaker()
        change = state.promote_provisional(temporary.session_speaker_id, "linked")
        assert change is not None and change.new_label.startswith("Speaker ")
        assert temporary.state is IdentityState.SESSION_ANONYMOUS

    def test_voice_id_accept_sets_registry_identity(self, state: SessionSpeakerState) -> None:
        speaker = state.ensure_cluster_speaker("cluster_a")
        state.apply_voice_id(
            speaker.session_speaker_id,
            VoiceIdDecision(
                status="enrolled",
                registry_speaker_id="EMP-042",
                speaker_name="Nguyễn Văn B",
                best_score=0.93,
                margin=0.2,
            ),
        )
        assert speaker.identity_status is IdentityStatus.ENROLLED
        assert speaker.public_speaker_id == "EMP-042"

    def test_uncalibrated_voice_id_changes_nothing(self, state: SessionSpeakerState) -> None:
        speaker = state.ensure_cluster_speaker("cluster_a")
        state.apply_voice_id(
            speaker.session_speaker_id,
            VoiceIdDecision(status="uncalibrated", reason="thresholds_null"),
        )
        assert speaker.identity_status is IdentityStatus.ANONYMOUS
        assert speaker.registry_speaker_id is None

    def test_provisional_with_separated_evidence_is_promoted_at_finalization(
        self, calibrated_config: SasttConfig
    ) -> None:
        state = SessionSpeakerState(
            session_id="ses", config=calibrated_config, embedding_model_version=MODEL
        )
        temporary = state.create_temporary_speaker()
        state.buffer_provisional_embedding(temporary.session_speaker_id, make_embedding(3))
        assert state.finalize_unresolved() == []
        assert temporary.state is IdentityState.SESSION_ANONYMOUS
        assert temporary.display_label == "Speaker 1"

    def test_unresolved_provisional_becomes_unknown_at_finalization(
        self, state: SessionSpeakerState
    ) -> None:
        temporary = state.create_temporary_speaker()
        assert state.finalize_unresolved() == [temporary.session_speaker_id]
        assert temporary.identity_status is IdentityStatus.UNKNOWN
