"""Evaluate load/soak measurements against declared SLOs (spec 3, 19.3)."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SLO_LIMITS: dict[str, float] = {
    "rtf": 0.50,
    "provisional_latency_s": 2.5,
    "attributed_latency_s": 5.0,
    "gpu_utilization": 0.80,
}


def percentile(values: list[float], p: float = 0.95) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    index = max(0, math.ceil(p * len(clean)) - 1)
    return clean[index]


def build_capacity_report(measurements: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(measurements)
    p95: dict[str, float | None] = {}
    gates: dict[str, str] = {}
    for metric, limit in SLO_LIMITS.items():
        values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
        value = percentile(values)
        p95[metric] = value
        gates[metric] = "not_evaluated" if value is None else ("pass" if value <= limit else "fail")
    return {
        "schema_version": "1.0",
        "samples": len(rows),
        "p95": p95,
        "limits": SLO_LIMITS,
        "gates": gates,
        "overall": "pass"
        if gates and all(state == "pass" for state in gates.values())
        else "pending",
        "note": "Run this only on the declared baseline hardware; missing measurements never pass.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON array of load measurements")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw: Any = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("input must be a JSON array of objects")
    args.output.write_text(
        json.dumps(build_capacity_report(raw), indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
