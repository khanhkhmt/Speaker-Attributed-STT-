"""WebSocket ingest — spec 8.2.

The client sends binary PCM s16le frames and receives the server events of
spec 8.2 as JSON, each with ``event_id``, a monotonic ``sequence_number``,
``revision``, ``server_time`` and the model/config versions.

A reconnect sends ``{"type": "resume", "last_sequence_number": N}`` and the
server replays from the event log rather than re-emitting finals (spec 8.2, 15).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from sastt.domain.errors import SasttError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sastt.api.http import AppState
    from sastt.application.streaming_pipeline import StreamingSession


def register_websocket_routes(app: FastAPI) -> None:
    @app.websocket("/v1/sessions/{session_id}/audio")
    async def audio_socket(websocket: WebSocket, session_id: str) -> None:
        state: AppState = websocket.app.state.sastt
        session = state.sessions.get(session_id)
        await websocket.accept()
        if session is None:
            await websocket.send_json(
                {"type": "session.failed", "payload": {"error_code": "UNKNOWN_SESSION"}}
            )
            await websocket.close()
            return

        try:
            if len(session.log) == 0:
                await _send(websocket, [session.start()])

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if (payload := message.get("bytes")) is not None:
                    await _send(websocket, session.push_pcm(payload))
                    continue

                text = message.get("text")
                if text is None:
                    continue
                await _handle_control(websocket, session, text)
                if _is_finalize(text):
                    break
        except WebSocketDisconnect:  # pragma: no cover - client hung up
            return
        except SasttError as exc:
            await websocket.send_json({"type": "session.failed", "payload": exc.to_dict()})
        finally:
            with contextlib.suppress(RuntimeError):  # already closed
                await websocket.close()


def _is_finalize(text: str) -> bool:
    import json

    try:
        return str(json.loads(text).get("type")) == "finalize"
    except (ValueError, AttributeError):
        return False


async def _handle_control(websocket: WebSocket, session: StreamingSession, text: str) -> None:
    import json

    try:
        message: dict[str, Any] = json.loads(text)
    except ValueError:
        return

    kind = message.get("type")
    if kind == "finalize":
        await _send(websocket, session.finalize())
    elif kind == "resume":
        # Reconnect replay — spec 8.2: no duplicated finals.
        last = int(message.get("last_sequence_number") or 0)
        await _send(websocket, session.replay(last))


async def _send(websocket: WebSocket, events: list[Any]) -> None:
    for event in events:
        await websocket.send_json(event.to_dict())


__all__ = ["register_websocket_routes"]
