"""Async MCP client adapter used by the agent act node."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from music_agent.config import Settings, get_settings


class McpToolClient:
    """Call MCP tools over streamable HTTP using the MCP protocol."""

    def __init__(
        self,
        server_url: str | None = None,
        settings: Settings | None = None,
        timeout: float = 30.0,
        streamable_client: Callable[..., Any] = streamablehttp_client,
        session_cls: type = ClientSession,
    ) -> None:
        self.settings = settings or get_settings()
        self.server_url = server_url or self.settings.mcp_server_url
        self.timeout = timeout
        self.streamable_client = streamable_client
        self.session_cls = session_cls

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool and return a stable wrapper."""

        start = time.perf_counter()
        try:
            async with self.streamable_client(self.server_url, timeout=self.timeout) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with self.session_cls(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, tool_input)
        except Exception as exc:  # noqa: BLE001 - transport boundary returns structured errors.
            return self._wrap(
                ok=False,
                tool_name=tool_name,
                start=start,
                result=None,
                error={
                    "error_code": "mcp_transport_error",
                    "error": str(exc),
                },
            )

        payload = extract_tool_payload(result)
        if getattr(result, "isError", False):
            return self._wrap(
                ok=False,
                tool_name=tool_name,
                start=start,
                result=None,
                error={
                    "error_code": "mcp_tool_error",
                    "error": payload,
                },
            )

        return self._wrap(
            ok=True,
            tool_name=tool_name,
            start=start,
            result=payload,
            error=None,
        )

    def _wrap(
        self,
        *,
        ok: bool,
        tool_name: str,
        start: float,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "ok": ok,
            "tool_name": tool_name,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "result": result,
            "error": error,
        }


def extract_tool_payload(result: Any) -> dict[str, Any]:
    """Extract a dict payload from an MCP CallToolResult."""

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    content = getattr(result, "content", None) or []
    if not content:
        return {}

    first = content[0]
    text = getattr(first, "text", None)
    if not isinstance(text, str):
        return {"content": str(first)}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    if isinstance(parsed, dict):
        return parsed
    return {"content": parsed}
