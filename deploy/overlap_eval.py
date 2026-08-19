#!/usr/bin/env python3
"""Score speaker attribution inside overlap regions against hand labels.

The pipeline can only be improved in the overlap branch if there is a way to
tell a *right* name from a *name*. Counting how many segments got a label is not
that: turning honest ``Unknown`` into confident misattribution improves the count
and makes the product worse. So this tool never reports one number. It reports a
triple that always sums to the labelled, decidable set:

    accuracy   — named, and the name matches the label
    confusion  — named, and the name contradicts the label
    unknown    — not named at all

A change is an improvement only when accuracy rises **and** confusion does not.

Labels are written in the vocabulary of the run's own roster — the labeller
hears "Speaker 2" in the clean parts of the recording and says an overlap source
is that same person. So a label is **not** an arbitrary cluster name and the
comparison is by identity, never by a permutation that maximises agreement. A
run that swaps two known speakers inside an overlap must score as confusion; a
free mapping would silently rename its way to a perfect score.

Comparing a *second* run needs care: session speaker numbering is per run, so
"Speaker 2" in run B need not be "Speaker 2" in run A. Run B is therefore aligned
to the labelled run first, using how much **non-overlap** speaking time the two
rosters share. That anchor is independent of the overlap decisions being scored.

Usage:
    python deploy/overlap_eval.py result.json --labels labels.json
    python deploy/overlap_eval.py before.json after.json --labels labels.json
    python deploy/overlap_eval.py result.json --labels labels.json --json out.json

Label file format (what the console's labelling mode exports)::

    {"job_id": "...", "labels": [
      {"start_ms": 2393, "end_ms": 5713, "source_track": 0, "speaker": "A"},
      {"start_ms": 2393, "end_ms": 4453, "source_track": 1, "speaker": "undecidable"}
    ]}

``speaker: "undecidable"`` means the labeller could not tell who was speaking.
Those rows are excluded from the triple and reported separately: they measure the
ceiling of the task, not the performance of the model (spec 19.1).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The same solver spec 5.8 uses to assign sources to speakers, applied here to
# assign predicted speaker IDs to label identities.
from scipy.optimize import linear_sum_assignment  # noqa: E402

#: A labeller who cannot tell who spoke says so; it is not a speaker name.
UNDECIDABLE = "undecidable"

#: The region was detected as overlap but only one person is really speaking, so
#: the second source is a separator artefact with no right answer. Counting that
#: against linking would blame the wrong stage — it is an overlap-detection miss.
SINGLE_SPEAKER = "single_speaker"

#: The separated stream itself carries words from two different people, so no
#: single speaker is the right answer for that row. That is a separation failure,
#: measured apart from linking: a leaked stream would make the linker look wrong
#: for a mistake it did not make.
MIXED_SOURCES = "mixed_sources"

#: Region-length buckets, in ms. The overlap failure is length-driven, so a
#: single average would hide exactly the effect worth watching.
BUCKETS: tuple[tuple[int, int], ...] = ((0, 500), (500, 1000), (1000, 1500), (1500, 1 << 30))


def _key(row: dict[str, Any]) -> tuple[int, int, int | None]:
    track = row.get("source_track")
    return int(row["start_ms"]), int(row["end_ms"]), None if track is None else int(track)


def load_labels(path: Path) -> dict[tuple[int, int, int | None], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["labels"] if isinstance(payload, dict) else payload
    labels: dict[tuple[int, int, int | None], str] = {}
    for row in rows:
        labels[_key(row)] = str(row["speaker"])
    return labels


def load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["segments"] if isinstance(payload, dict) else payload)


def overlap_only(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [segment for segment in segments if segment.get("is_overlap")]


def predicted_speaker(segment: dict[str, Any]) -> str | None:
    """The identity the pipeline committed to, or ``None`` for an honest Unknown.

    The *displayed label* is the answer, because that is what the labeller was
    shown and answered in. A session speaker ID would need translating; a label
    is already the vocabulary of the question.
    """
    if segment.get("identity_status") == "unknown":
        return None
    label = segment.get("speaker_label")
    if label is None or str(label).strip().lower() == "unknown":
        return None
    return str(label)


def _speaking_time(segments: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    """Non-overlap spans per displayed label — the anchor for aligning two runs."""
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for segment in segments:
        if segment.get("is_overlap"):
            continue
        label = predicted_speaker(segment)
        if label is not None:
            spans[label].append((int(segment["start_ms"]), int(segment["end_ms"])))
    return spans


def align_to_reference(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, str]:
    """Map the candidate run's labels onto the labelled run's labels.

    Two runs of the same audio number their speakers independently. They are
    matched by how much non-overlap time each pair shares — audio the labels do
    not depend on, so the alignment cannot launder an overlap mistake.
    """
    left, right = _speaking_time(reference), _speaking_time(candidate)
    if not left or not right:
        return {}
    ref_labels, cand_labels = sorted(left), sorted(right)
    matrix = [
        [
            float(
                sum(
                    max(0, min(a_end, b_end) - max(a_start, b_start))
                    for a_start, a_end in left[ref]
                    for b_start, b_end in right[cand]
                )
            )
            for ref in ref_labels
        ]
        for cand in cand_labels
    ]
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return {
        cand_labels[row]: ref_labels[column]
        for row, column in zip(rows, columns, strict=True)
        if matrix[row][column] > 0
    }


def bucket_of(duration_ms: int) -> str:
    for low, high in BUCKETS:
        if low <= duration_ms < high:
            return f"{low}-{high}" if high < (1 << 30) else f"{low}+"
    return "?"


def score(
    segments: list[dict[str, Any]],
    labels: dict[tuple[int, int, int | None], str],
    *,
    alignment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score one run. ``alignment`` translates this run's labels into the labelled run's."""
    mapping = alignment or {}
    overlaps = overlap_only(segments)
    matched = [(segment, labels[_key(segment)]) for segment in overlaps if _key(segment) in labels]
    undecidable = [item for item in matched if item[1] == UNDECIDABLE]
    single = [item for item in matched if item[1] == SINGLE_SPEAKER]
    mixed = [item for item in matched if item[1] == MIXED_SOURCES]
    decidable = [
        item for item in matched if item[1] not in (UNDECIDABLE, SINGLE_SPEAKER, MIXED_SOURCES)
    ]

    rows: list[dict[str, Any]] = []
    for segment, truth in decidable:
        raw = predicted_speaker(segment)
        prediction = None if raw is None else mapping.get(raw, raw)
        duration = int(segment["end_ms"]) - int(segment["start_ms"])
        if prediction is None:
            outcome = "unknown"
        elif prediction == truth:
            outcome = "accuracy"
        else:
            outcome = "confusion"
        rows.append(
            {
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "source_track": segment.get("source_track"),
                "duration_ms": duration,
                "bucket": bucket_of(duration),
                "concurrent": segment.get("estimated_concurrent_speakers"),
                "truth": truth,
                "predicted": prediction,
                "outcome": outcome,
            }
        )

    def triple(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        counts = {
            name: sum(1 for r in subset if r["outcome"] == name)
            for name in ("accuracy", "confusion", "unknown")
        }
        return {
            "n": total,
            **counts,
            "rates": {
                name: (None if total == 0 else round(counts[name] / total, 4)) for name in counts
            },
        }

    by_bucket = {}
    for low, high in BUCKETS:
        name = f"{low}-{high}" if high < (1 << 30) else f"{low}+"
        by_bucket[name] = triple([r for r in rows if r["bucket"] == name])
    by_concurrent = {}
    for value in sorted({r["concurrent"] for r in rows}, key=lambda v: (v is None, v)):
        by_concurrent[str(value)] = triple([r for r in rows if r["concurrent"] == value])

    return {
        "labelled_overlap_segments": len(matched),
        "undecidable_by_labeller": len(undecidable),
        "not_really_overlap": len(single),
        "leaked_separation": len(mixed),
        "unlabelled_overlap_segments": len(overlaps) - len(matched),
        "overall": triple(rows),
        "by_region_length_ms": by_bucket,
        "by_estimated_concurrent_speakers": by_concurrent,
        "roster_alignment": mapping,
        "rows": rows,
    }


def _fmt_triple(name: str, block: dict[str, Any]) -> str:
    rates = block["rates"]

    def pct(key: str) -> str:
        value = rates[key]
        return "    —" if value is None else f"{value * 100:5.1f}%"

    return (
        f"  {name:<14} n={block['n']:<4} "
        f"đúng {pct('accuracy')}  nhầm {pct('confusion')}  Unknown {pct('unknown')}"
    )


def report(label: str, result: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(
        f"  segment overlap có nhãn: {result['labelled_overlap_segments']}"
        f"  (không nghe ra: {result['undecidable_by_labeller']},"
        f" thực ra 1 người: {result['not_really_overlap']},"
        f" luồng lẫn giọng: {result['leaked_separation']},"
        f" chưa gán nhãn: {result['unlabelled_overlap_segments']})"
    )
    if result["overall"]["n"] == 0:
        print("  không có segment nào quyết định được — chưa kết luận gì")
        return
    print(_fmt_triple("TỔNG", result["overall"]))
    print("  theo độ dài vùng:")
    for name, block in result["by_region_length_ms"].items():
        if block["n"]:
            print(_fmt_triple(f"  {name} ms", block))
    print("  theo số người đồng thời:")
    for name, block in result["by_estimated_concurrent_speakers"].items():
        if block["n"]:
            print(_fmt_triple(f"  K={name}", block))


def compare(before: dict[str, Any], after: dict[str, Any]) -> int:
    """Apply the rule the plan commits to: accuracy up AND confusion not up."""
    b, a = before["overall"]["rates"], after["overall"]["rates"]
    if before["overall"]["n"] == 0 or after["overall"]["n"] == 0:
        print("\nKhông đủ nhãn để so sánh.")
        return 1
    d_acc = (a["accuracy"] or 0) - (b["accuracy"] or 0)
    d_con = (a["confusion"] or 0) - (b["confusion"] or 0)
    d_unk = (a["unknown"] or 0) - (b["unknown"] or 0)
    print("\n=== SO SÁNH ===")
    print(f"  đúng     {d_acc * 100:+6.1f} điểm phần trăm")
    print(f"  nhầm     {d_con * 100:+6.1f}")
    print(f"  Unknown  {d_unk * 100:+6.1f}")
    improved = d_acc > 0 and d_con <= 0
    print("\n  => CẢI THIỆN" if improved else "\n  => KHÔNG ĐẠT: cần đúng tăng VÀ nhầm không tăng")
    if d_unk < 0 and d_con > 0:
        print("     (Unknown giảm nhưng nhầm tăng — đây chính là cái bẫy đếm số lượng)")
    return 0 if improved else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "results", type=Path, nargs="+", help="job result JSON (1 = đo, 2 = so sánh)"
    )
    parser.add_argument("--labels", type=Path, required=True, help="label JSON from the console")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args(argv)

    labels = load_labels(args.labels)
    if not labels:
        parser.error("label file is empty; nothing can be measured")

    if len(args.results) > 2:
        parser.error("pass one result to measure, or two to compare")
    runs = [(path.name, load_segments(path)) for path in args.results]
    reference = runs[0][1]
    scored = []
    for index, (name, segments) in enumerate(runs):
        alignment = None if index == 0 else align_to_reference(reference, segments)
        if alignment:
            renames = {c: r for c, r in alignment.items() if c != r}
            if renames:
                print(f"\nGióng roster của {name} về lượt đã gán nhãn: {renames}")
        scored.append((name, score(segments, labels, alignment=alignment)))
    for name, result in scored:
        report(name, result)

    exit_code = 0
    if len(scored) == 2:
        exit_code = compare(scored[0][1], scored[1][1])

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(dict(scored), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nĐã ghi {args.json}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
