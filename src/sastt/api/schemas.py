"""JSON Schema for the public output contract v2 — spec 7, 16.3.

The schema encodes the invariants of spec 7 that a document validator can
express: bounds on ``start_ms``/``end_ms``, ``revision >= 1``, the
``identity_status`` enum, ``[0,1]`` confidences that may also be ``null``, and
the rule that ``registry_speaker_id``/``speaker_name`` are only required when
the segment is ``enrolled``.

Backward compatibility (spec 16.3): :func:`downgrade_segment_to_v1` projects a
v2 segment onto the v1 field set. The repository carries no legacy transcript
producer, so v1 is pinned here as the minimal pre-v2 shape — timestamps, text
and a single ``speaker_id`` — and the compatibility test asserts that a v2
segment always yields a valid v1 document.
"""

from __future__ import annotations

from typing import Any

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

_CONFIDENCE = {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0}

TRANSCRIPT_SEGMENT_V2_SCHEMA: dict[str, Any] = {
    "$schema": SCHEMA_DIALECT,
    "$id": "https://sastt.local/schemas/transcript-segment-v2.json",
    "title": "Speaker-attributed transcript segment v2",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "session_id",
        "event_id",
        "revision",
        "start_ms",
        "end_ms",
        "text",
        "speaker_id",
        "session_speaker_id",
        "identity_status",
        "is_overlap",
        "confidence_status",
        "is_final",
        "model_versions",
    ],
    "properties": {
        "schema_version": {"const": "2.0"},
        "session_id": {"type": "string", "minLength": 1},
        "event_id": {"type": "string", "minLength": 1},
        "revision": {"type": "integer", "minimum": 1},
        "supersedes_event_id": {"type": ["string", "null"]},
        "start_ms": {"type": "integer", "minimum": 0},
        "end_ms": {"type": "integer", "minimum": 1},
        "text": {"type": "string"},
        "speaker_id": {"type": "string", "minLength": 1},
        "session_speaker_id": {"type": "string", "minLength": 1},
        "registry_speaker_id": {"type": ["string", "null"]},
        "speaker_label": {"type": "string"},
        "speaker_name": {"type": ["string", "null"]},
        "identity_status": {
            "enum": ["provisional", "enrolled", "anonymous", "unknown", "ambiguous"]
        },
        "is_overlap": {"type": "boolean"},
        "estimated_concurrent_speakers": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 5,
        },
        "count_confidence": _CONFIDENCE,
        "source_track": {"type": ["integer", "null"], "minimum": 0},
        "separation_backend": {"type": ["string", "null"]},
        "asr_confidence": _CONFIDENCE,
        "diarization_confidence": _CONFIDENCE,
        "linking_confidence": _CONFIDENCE,
        "voice_id_confidence": _CONFIDENCE,
        "overlap_confidence": _CONFIDENCE,
        "overall_confidence": _CONFIDENCE,
        "confidence_status": {"enum": ["uncalibrated", "calibrated"]},
        "raw_scores": {"type": "object", "additionalProperties": {"type": "number"}},
        "quality_flags": {"type": "array", "items": {"type": "string"}},
        "degraded_mode": {"type": "boolean"},
        "is_final": {"type": "boolean"},
        "model_versions": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "diarization": {"type": ["string", "null"]},
                "embedding": {"type": ["string", "null"]},
                "separation": {"type": ["string", "null"]},
                "asr": {"type": ["string", "null"]},
                "calibration": {"type": ["string", "null"]},
            },
        },
    },
    "allOf": [
        {
            "if": {"properties": {"identity_status": {"const": "enrolled"}}},
            "then": {
                "required": ["registry_speaker_id", "speaker_name"],
                "properties": {
                    "registry_speaker_id": {"type": "string", "minLength": 1},
                    "speaker_name": {"type": "string", "minLength": 1},
                },
            },
        },
        {
            "if": {
                "properties": {"separation_backend": {"type": "string"}},
                "required": ["separation_backend"],
            },
            "then": {
                "required": ["source_track"],
                "properties": {"source_track": {"type": "integer", "minimum": 0}},
            },
        },
        {
            "if": {"properties": {"confidence_status": {"const": "uncalibrated"}}},
            "then": {"properties": {"overall_confidence": {"const": None}}},
        },
    ],
}

SERVER_EVENT_V2_SCHEMA: dict[str, Any] = {
    "$schema": SCHEMA_DIALECT,
    "$id": "https://sastt.local/schemas/server-event-v2.json",
    "title": "Realtime server event",
    "type": "object",
    "required": [
        "event_id",
        "session_id",
        "sequence_number",
        "type",
        "server_time_ms",
        "revision",
        "is_final",
    ],
    "properties": {
        "event_id": {"type": "string", "minLength": 1},
        "session_id": {"type": "string", "minLength": 1},
        "sequence_number": {"type": "integer", "minimum": 1},
        "type": {
            "enum": [
                "session.started",
                "transcript.provisional",
                "transcript.revision",
                "transcript.final",
                "pipeline.warning",
                "session.finalized",
                "session.failed",
            ]
        },
        "server_time_ms": {"type": "integer", "minimum": 0},
        "revision": {"type": "integer", "minimum": 1},
        "supersedes_event_id": {"type": ["string", "null"]},
        "is_final": {"type": "boolean"},
        "payload": {"type": "object"},
        "model_versions": {"type": "object"},
        "config_version": {"type": ["string", "null"]},
    },
}

#: Legacy v1 projection kept for backward compatibility (spec 16.3).
SEGMENT_V1_SCHEMA: dict[str, Any] = {
    "$schema": SCHEMA_DIALECT,
    "$id": "https://sastt.local/schemas/transcript-segment-v1.json",
    "title": "Speaker-attributed transcript segment v1",
    "type": "object",
    "required": ["schema_version", "start_ms", "end_ms", "text", "speaker_id", "speaker_label"],
    "properties": {
        "schema_version": {"const": "1.0"},
        "session_id": {"type": "string"},
        "start_ms": {"type": "integer", "minimum": 0},
        "end_ms": {"type": "integer", "minimum": 1},
        "text": {"type": "string"},
        "speaker_id": {"type": "string", "minLength": 1},
        "speaker_label": {"type": "string"},
        "is_overlap": {"type": "boolean"},
        "is_final": {"type": "boolean"},
    },
}

V1_FIELDS = tuple(SEGMENT_V1_SCHEMA["properties"])


def downgrade_segment_to_v1(segment: dict[str, Any]) -> dict[str, Any]:
    """Project a v2 segment onto the v1 field set (spec 16.3).

    Concurrency survives the projection: two overlapping v2 segments become two
    overlapping v1 segments, they are never merged.
    """
    return {
        "schema_version": "1.0",
        "session_id": segment["session_id"],
        "start_ms": segment["start_ms"],
        "end_ms": segment["end_ms"],
        "text": segment["text"],
        "speaker_id": segment["speaker_id"],
        "speaker_label": segment.get("speaker_label") or segment["session_speaker_id"],
        "is_overlap": bool(segment.get("is_overlap", False)),
        "is_final": bool(segment.get("is_final", False)),
    }


def _validate(document: dict[str, Any], schema: dict[str, Any]) -> None:
    import jsonschema  # imported lazily: only tests and tooling need it

    jsonschema.validate(instance=document, schema=schema)


def validate_segment_v2(document: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` when the document violates the contract."""
    _validate(document, TRANSCRIPT_SEGMENT_V2_SCHEMA)


def validate_segment_v1(document: dict[str, Any]) -> None:
    _validate(document, SEGMENT_V1_SCHEMA)


def validate_server_event(document: dict[str, Any]) -> None:
    _validate(document, SERVER_EVENT_V2_SCHEMA)


__all__ = [
    "SCHEMA_DIALECT",
    "SEGMENT_V1_SCHEMA",
    "SERVER_EVENT_V2_SCHEMA",
    "TRANSCRIPT_SEGMENT_V2_SCHEMA",
    "V1_FIELDS",
    "downgrade_segment_to_v1",
    "validate_segment_v1",
    "validate_segment_v2",
    "validate_server_event",
]
