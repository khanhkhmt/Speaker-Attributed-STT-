"""The overlap scorer — the instrument every later change is judged by.

Spec 18 rule 5 forbids tuning code to make a number look good. That only bites
if the number is honest in the first place, so the scorer is tested before it is
trusted. Two ways it could lie, both pinned here: rewarding a swap between two
known speakers, and reading a trade of ``Unknown`` for confusion as progress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# pyproject puts the repository root on the path, so `deploy` imports directly.
from deploy.overlap_eval import (  # noqa: E402
    MIXED_SOURCES,
    SINGLE_SPEAKER,
    UNDECIDABLE,
    align_to_reference,
    compare,
    load_labels,
    score,
)


def segment(
    start: int,
    end: int,
    track: int | None,
    label: str | None,
    *,
    overlap: bool = True,
    concurrent: int = 2,
) -> dict:
    return {
        "start_ms": start,
        "end_ms": end,
        "source_track": track,
        "is_overlap": overlap,
        "estimated_concurrent_speakers": concurrent if overlap else None,
        "speaker_label": "Unknown" if label is None else label,
        "identity_status": "unknown" if label is None else "anonymous",
    }


class TestIdentityComparison:
    def test_swapping_two_known_speakers_is_confusion_not_accuracy(self) -> None:
        """The regression that matters: a consistent swap must never score as correct.

        The labeller answers in the run's own vocabulary — "Speaker 2" is a person
        heard in the clean parts. Remapping the roster to maximise agreement would
        turn a swapped overlap into a perfect score.
        """
        segments = [
            segment(1000, 3000, 0, "Speaker 1"),
            segment(1000, 3000, 1, "Speaker 2"),
        ]
        labels = {(1000, 3000, 0): "Speaker 2", (1000, 3000, 1): "Speaker 1"}

        result = score(segments, labels)

        assert result["overall"]["confusion"] == 2
        assert result["overall"]["accuracy"] == 0

    def test_matching_labels_score_as_accuracy(self) -> None:
        segments = [segment(1000, 3000, 0, "Speaker 1")]
        result = score(segments, {(1000, 3000, 0): "Speaker 1"})
        assert result["overall"]["accuracy"] == 1


class TestRosterAlignment:
    def test_a_second_run_is_aligned_by_non_overlap_speaking_time(self) -> None:
        """Run B numbers its speakers differently; the anchor is clean speech, not the labels."""
        reference = [
            segment(0, 5000, None, "Speaker 1", overlap=False),
            segment(5000, 10000, None, "Speaker 2", overlap=False),
        ]
        candidate = [
            segment(0, 5000, None, "Speaker 2", overlap=False),
            segment(5000, 10000, None, "Speaker 1", overlap=False),
        ]

        alignment = align_to_reference(reference, candidate)

        assert alignment == {"Speaker 2": "Speaker 1", "Speaker 1": "Speaker 2"}

    def test_alignment_cannot_launder_an_overlap_mistake(self) -> None:
        """Both runs agree on clean speech, so a swapped overlap stays a swap."""
        clean = [
            segment(0, 5000, None, "Speaker 1", overlap=False),
            segment(5000, 10000, None, "Speaker 2", overlap=False),
        ]
        reference = [*clean, segment(11000, 13000, 0, "Speaker 1")]
        candidate = [*clean, segment(11000, 13000, 0, "Speaker 2")]
        labels = {(11000, 13000, 0): "Speaker 1"}

        alignment = align_to_reference(reference, candidate)
        result = score(candidate, labels, alignment=alignment)

        assert result["overall"]["confusion"] == 1


class TestTriple:
    def test_the_three_outcomes_partition_the_decidable_set(self) -> None:
        segments = [
            segment(0, 900, 0, "Speaker 1"),
            segment(0, 900, 1, "Speaker 2"),
            segment(2000, 2400, 0, None),
        ]
        labels = {(0, 900, 0): "Speaker 1", (0, 900, 1): "Speaker 2", (2000, 2400, 0): "Speaker 1"}
        overall = score(segments, labels)["overall"]
        assert overall["n"] == 3
        assert overall["accuracy"] + overall["confusion"] + overall["unknown"] == 3
        assert overall["accuracy"] == 2
        assert overall["unknown"] == 1

    def test_undecidable_rows_are_excluded_not_counted_as_failure(self) -> None:
        """What the labeller cannot hear measures the task's ceiling, not the model."""
        segments = [segment(0, 400, 0, None), segment(0, 900, 0, "Speaker 1")]
        labels = {(0, 400, 0): UNDECIDABLE, (0, 900, 0): "Speaker 1"}
        result = score(segments, labels)
        assert result["undecidable_by_labeller"] == 1
        assert result["overall"]["n"] == 1
        assert result["overall"]["accuracy"] == 1

    def test_a_region_that_is_not_really_overlap_is_counted_apart(self) -> None:
        """OSD fired on one speaker: the phantom source has no right answer.

        Scoring it as a linking failure would blame the wrong stage, and folding
        it into "cannot tell" would hide an overlap-detection error entirely.
        """
        segments = [segment(0, 900, 0, "Speaker 1"), segment(0, 900, 1, None)]
        labels = {(0, 900, 0): "Speaker 1", (0, 900, 1): SINGLE_SPEAKER}

        result = score(segments, labels)

        assert result["not_really_overlap"] == 1
        assert result["overall"]["n"] == 1
        assert result["overall"]["accuracy"] == 1

    def test_a_leaked_stream_is_not_blamed_on_linking(self) -> None:
        """One separated stream carrying two voices has no right answer.

        Scoring it would make the linker look wrong for the separator's mistake.
        """
        segments = [segment(0, 900, 0, "Speaker 1"), segment(0, 900, 1, "Speaker 2")]
        labels = {(0, 900, 0): "Speaker 1", (0, 900, 1): MIXED_SOURCES}

        result = score(segments, labels)

        assert result["leaked_separation"] == 1
        assert result["overall"]["n"] == 1

    def test_results_are_split_by_region_length(self) -> None:
        segments = [segment(0, 400, 0, "Speaker 1"), segment(5000, 7000, 0, "Speaker 1")]
        labels = {(0, 400, 0): "Speaker 1", (5000, 7000, 0): "Speaker 1"}
        buckets = score(segments, labels)["by_region_length_ms"]
        assert buckets["0-500"]["n"] == 1
        assert buckets["1500+"]["n"] == 1


class TestComparisonRule:
    def _result(self, accuracy: int, confusion: int, unknown: int) -> dict:
        segments, labels = [], {}
        start = 0
        for outcome, count in (("acc", accuracy), ("con", confusion), ("unk", unknown)):
            for _ in range(count):
                start += 1000
                end = start + 900
                label = {"acc": "Speaker 1", "con": "Speaker 2", "unk": None}[outcome]
                segments.append(segment(start, end, 0, label))
                labels[(start, end, 0)] = "Speaker 1"
        return score(segments, labels)

    def test_trading_unknown_for_confusion_is_not_an_improvement(self) -> None:
        """The count-of-names trap: fewer Unknown, more wrong names, rejected."""
        before = self._result(accuracy=4, confusion=0, unknown=6)
        after = self._result(accuracy=4, confusion=6, unknown=0)
        assert compare(before, after) != 0

    def test_real_gain_passes(self) -> None:
        before = self._result(accuracy=4, confusion=1, unknown=5)
        after = self._result(accuracy=8, confusion=1, unknown=1)
        assert compare(before, after) == 0


def test_label_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "job_id": "job_x",
                "labels": [
                    {"start_ms": 10, "end_ms": 20, "source_track": 0, "speaker": "Speaker 1"}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_labels(path) == {(10, 20, 0): "Speaker 1"}
