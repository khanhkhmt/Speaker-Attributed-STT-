"""Fusion rules — spec 5.11, 7."""

from __future__ import annotations

import pytest

from sastt.application.fusion import (
    FusionEngine,
    NullConfidenceCalibrator,
    WordGroup,
    dedup_words,
)
from sastt.application.session_state import SessionSpeakerState
from sastt.config import SasttConfig
from sastt.domain.audio import TimeInterval
from sastt.domain.errors import SchemaInvariantError
from sastt.domain.speakers import SpeakerTurn
from sastt.domain.transcript import ModelVersions, Word

pytestmark = pytest.mark.unit


@pytest.fixture
def state(base_config: SasttConfig) -> SessionSpeakerState:
    state = SessionSpeakerState(
        session_id="ses_1", config=base_config, embedding_model_version="fake@1"
    )
    state.ensure_cluster_speaker("cluster_a")
    state.ensure_cluster_speaker("cluster_b")
    return state


@pytest.fixture
def engine(base_config: SasttConfig, state: SessionSpeakerState) -> FusionEngine:
    return FusionEngine(
        session_id="ses_1",
        state=state,
        config=base_config,
        model_versions=ModelVersions(diarization="fake@1", asr="fake@1"),
    )


def words(*specs: tuple[str, int, int]) -> tuple[Word, ...]:
    return tuple(
        Word(text=text, start_ms=start, end_ms=end, raw_probability=0.9)
        for text, start, end in specs
    )


class TestAttribution:
    def test_word_goes_to_the_turn_with_the_largest_intersection(
        self, engine: FusionEngine, state: SessionSpeakerState
    ) -> None:
        turns = [
            SpeakerTurn("cluster_a", TimeInterval(0, 1000)),
            SpeakerTurn("cluster_b", TimeInterval(1000, 2000)),
        ]
        word = Word(text="chào", start_ms=900, end_ms=1800)
        speaker_id = engine.attribute_word(word, turns)
        assert speaker_id == state.by_cluster("cluster_b").session_speaker_id  # type: ignore[union-attr]

    def test_overlap_group_keeps_its_linked_speaker(
        self, engine: FusionEngine, state: SessionSpeakerState
    ) -> None:
        speaker = state.by_cluster("cluster_b")
        assert speaker is not None
        group = WordGroup(
            words=words(("một", 1000, 1500)),
            interval=TimeInterval(1000, 1500),
            is_overlap=True,
            source_track=0,
            separation_backend="mossformer2_ss_16k",
            session_speaker_id=speaker.session_speaker_id,
        )
        segments = engine.fuse([group], [SpeakerTurn("cluster_a", TimeInterval(0, 5000))])
        assert segments[0].session_speaker_id == speaker.session_speaker_id
        assert segments[0].is_overlap is True
        assert segments[0].source_track == 0


class TestCoalescing:
    def test_words_of_one_speaker_join_into_one_utterance(self, engine: FusionEngine) -> None:
        group = WordGroup(
            words=words(("xin", 0, 400), ("chào", 400, 800)),
            interval=TimeInterval(0, 800),
        )
        segments = engine.fuse([group], [SpeakerTurn("cluster_a", TimeInterval(0, 1000))])
        assert len(segments) == 1
        assert segments[0].text == "xin chào"

    def test_a_long_pause_splits_the_utterance(self, engine: FusionEngine) -> None:
        group = WordGroup(
            words=words(("xin", 0, 400), ("chào", 4000, 4400)),
            interval=TimeInterval(0, 4400),
        )
        segments = engine.fuse([group], [SpeakerTurn("cluster_a", TimeInterval(0, 5000))])
        assert len(segments) == 2

    def test_sentence_final_punctuation_splits(self, engine: FusionEngine) -> None:
        group = WordGroup(
            words=words(("chào.", 0, 400), ("tiếp", 500, 900)),
            interval=TimeInterval(0, 900),
        )
        segments = engine.fuse([group], [SpeakerTurn("cluster_a", TimeInterval(0, 1000))])
        assert len(segments) == 2

    def test_two_speakers_with_identical_text_are_never_merged(
        self, engine: FusionEngine, state: SessionSpeakerState
    ) -> None:
        first = state.by_cluster("cluster_a")
        second = state.by_cluster("cluster_b")
        assert first and second
        groups = [
            WordGroup(
                words=words(("vâng", 1000, 1500)),
                interval=TimeInterval(1000, 1500),
                is_overlap=True,
                source_track=0,
                session_speaker_id=first.session_speaker_id,
            ),
            WordGroup(
                words=words(("vâng", 1000, 1500)),
                interval=TimeInterval(1000, 1500),
                is_overlap=True,
                source_track=1,
                session_speaker_id=second.session_speaker_id,
            ),
        ]
        segments = engine.fuse(groups, [])
        assert len(segments) == 2
        assert segments[0].interval.intersects(segments[1].interval)


class TestGuards:
    def test_fusion_never_loses_a_word(self, engine: FusionEngine) -> None:
        group = WordGroup(words=words(("một", 0, 400)), interval=TimeInterval(0, 400))
        segments = engine.fuse([group], [SpeakerTurn("cluster_a", TimeInterval(0, 1000))])
        assert sum(len(segment.words) for segment in segments) == 1

    def test_word_loss_is_reported_not_swallowed(self, engine: FusionEngine) -> None:
        from sastt.application import fusion as fusion_module

        with pytest.raises(SchemaInvariantError):
            fusion_module._assert_concurrency_preserved(
                [
                    WordGroup(
                        words=words(("a", 0, 100), ("b", 100, 200)), interval=TimeInterval(0, 200)
                    )
                ],
                [],
            )

    def test_confidences_stay_null_without_a_calibrator(self, engine: FusionEngine) -> None:
        group = WordGroup(words=words(("một", 0, 400)), interval=TimeInterval(0, 400))
        segment = engine.fuse([group], [SpeakerTurn("cluster_a", TimeInterval(0, 1000))])[0]
        assert segment.confidences.status == "uncalibrated"
        assert segment.confidences.overall is None
        assert segment.raw_scores["asr_word_probability"] == pytest.approx(0.9)

    def test_null_calibrator_has_no_version(self) -> None:
        assert NullConfidenceCalibrator().calibration_version is None


class TestDedup:
    def test_words_are_deduplicated_by_stable_id(self) -> None:
        existing = [Word(text="xin", start_ms=0, end_ms=400).with_stable_id("ses")]
        incoming = [Word(text="xin", start_ms=0, end_ms=400).with_stable_id("ses")]
        assert dedup_words(existing, incoming) == []

    def test_same_text_within_tolerance_is_a_duplicate(self) -> None:
        existing = [Word(text="xin", start_ms=1000, end_ms=1400)]
        incoming = [Word(text="xin", start_ms=1100, end_ms=1500)]
        assert dedup_words(existing, incoming, tolerance_ms=200) == []

    def test_same_text_far_apart_is_a_new_word(self) -> None:
        existing = [Word(text="xin", start_ms=1000, end_ms=1400)]
        incoming = [Word(text="xin", start_ms=9000, end_ms=9400)]
        assert len(dedup_words(existing, incoming, tolerance_ms=200)) == 1
