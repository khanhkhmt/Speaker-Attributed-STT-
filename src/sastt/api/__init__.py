"""Public API layer — output contract and transport (spec 7, 8).

Milestone 0 ships the schema/contract module; the FastAPI HTTP surface
(``http.py``) and the WebSocket ingest (``websocket.py``) follow in Milestones 1
and 3 (spec 18).
"""

from sastt.api.schemas import (
    SEGMENT_V1_SCHEMA,
    SERVER_EVENT_V2_SCHEMA,
    TRANSCRIPT_SEGMENT_V2_SCHEMA,
    downgrade_segment_to_v1,
    validate_segment_v2,
    validate_server_event,
)

__all__ = [
    "SEGMENT_V1_SCHEMA",
    "SERVER_EVENT_V2_SCHEMA",
    "TRANSCRIPT_SEGMENT_V2_SCHEMA",
    "downgrade_segment_to_v1",
    "validate_segment_v2",
    "validate_server_event",
]
