"""Per-request JSONL tracing for the agent loop."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

_trace_context: ContextVar[tuple[str, Path] | None] = ContextVar(
    "agent_trace_context",
    default=None,
)
_write_lock = Lock()


@contextmanager
def request_trace(request_id: str, path: str | Path) -> Iterator[None]:
    """Attach a request id and output path to the current async context."""

    token = _trace_context.set((request_id, Path(path)))
    try:
        yield
    finally:
        _trace_context.reset(token)


def write_trace_event(event: str, **details: Any) -> None:
    """Append one structured event when request tracing is active."""

    context = _trace_context.get()
    if context is None:
        return
    request_id, path = context
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "event": event,
        **details,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
        with _write_lock, path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError:
        # Trace persistence must not make a chat request fail.
        return
