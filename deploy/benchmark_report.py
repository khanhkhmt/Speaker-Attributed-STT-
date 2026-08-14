"""Create a reproducible benchmark report from scored JSONL evidence (spec 21)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Token WER using a deterministic Levenshtein distance."""
    expected, actual = reference.split(), hypothesis.split()
    if not expected:
        return 0.0 if not actual else 1.0
    row = list(range(len(actual) + 1))
    for index, token in enumerate(expected, start=1):
        next_row = [index]
        for actual_index, actual_token in enumerate(actual, start=1):
            next_row.append(
                min(
                    row[actual_index] + 1,
                    next_row[actual_index - 1] + 1,
                    row[actual_index - 1] + (token != actual_token),
                )
            )
        row = next_row
    return row[-1] / len(expected)


def build_report(records: Iterable[dict[str, Any]], *, release_id: str) -> dict[str, Any]:
    """Aggregate evidence without treating absent targets as a passing result."""
    scored = list(records)
    wers = [
        word_error_rate(str(item["reference_text"]), str(item["hypothesis_text"]))
        for item in scored
        if isinstance(item.get("reference_text"), str)
        and isinstance(item.get("hypothesis_text"), str)
    ]
    speaker = [
        bool(item["speaker_attributed_correct"])
        for item in scored
        if "speaker_attributed_correct" in item
    ]
    return {
        "schema_version": "1.0",
        "release_id": release_id,
        "records": len(scored),
        "wer": sum(wers) / len(wers) if wers else None,
        "speaker_attribution_accuracy": sum(speaker) / len(speaker) if speaker else None,
        "evidence_status": "measured" if wers or speaker else "not_evaluated",
        "quality_gate": "pending_review",
        "note": "A report is evidence, not an automatic production approval.",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"line {number} is not an object")
        rows.append(item)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL benchmark evidence")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(
        json.dumps(build_report(_read_jsonl(args.input), release_id=args.release_id), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
