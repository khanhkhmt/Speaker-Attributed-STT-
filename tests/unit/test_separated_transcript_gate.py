"""Guards against hallucinated text from a speechless separated source."""

from __future__ import annotations

import pytest

from sastt.application.offline_pipeline import is_plausible_transcript
from sastt.domain.audio import TimeInterval
from sastt.domain.transcript import Word

pytestmark = pytest.mark.unit


def _words(*texts: str) -> tuple[Word, ...]:
    return tuple(
        Word(text=text, start_ms=index * 20, end_ms=(index + 1) * 20)
        for index, text in enumerate(texts)
    )


def test_rejects_a_long_transcript_from_a_320ms_separated_source() -> None:
    words = _words(
        "Hãy",
        "subscribe",
        "cho",
        "kênh",
        "La",
        "La",
        "School",
        "Để",
        "không",
        "bỏ",
        "lỡ",
        "video",
        "hấp",
        "dẫn",
    )

    assert is_plausible_transcript(words, [TimeInterval(0, 320)]) is False
    assert is_plausible_transcript(words, [TimeInterval(0, 2_000)]) is False


def test_keeps_a_short_genuine_acknowledgement() -> None:
    assert is_plausible_transcript(_words("Vâng"), [TimeInterval(0, 180)]) is True


def test_rejects_text_when_source_vad_found_no_speech() -> None:
    assert is_plausible_transcript(_words("xin", "chào"), []) is False
