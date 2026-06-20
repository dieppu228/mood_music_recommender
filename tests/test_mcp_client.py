import json
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, TextContent

from music_agent.mcp_server import rag_tool
from music_agent.agent.trace import request_trace
from music_agent.mcp_server.server import mcp
from music_agent.models import MusicRagSearchResult, SongPayload
from music_agent.tools.mcp_client import McpToolClient


class FakeStreamableClient:
    def __init__(self, url: str, **kwargs) -> None:
        self.url = url
        self.kwargs = kwargs

    async def __aenter__(self):
        return "read-stream", "write-stream", lambda: "session-id"

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FailingStreamableClient:
    def __init__(self, url: str, **kwargs) -> None:
        self.url = url
        self.kwargs = kwargs

    async def __aenter__(self):
        raise ConnectionError("connect refused")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeSession:
    calls: list[tuple[str, dict]] = []
    result: CallToolResult = CallToolResult(
        content=[TextContent(type="text", text="{}")],
        structuredContent={},
        isError=False,
    )

    def __init__(self, read_stream, write_stream) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.initialized = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def initialize(self) -> None:
        self.initialized = True

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        assert self.initialized is True
        self.calls.append((name, arguments))
        return self.result


def make_session(result: CallToolResult):
    class Session(FakeSession):
        calls: list[tuple[str, dict]] = []

    Session.result = result

    return Session


def text_result(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=None,
        isError=is_error,
    )


def structured_result(payload: dict, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text="{}")],
        structuredContent=payload,
        isError=is_error,
    )


@pytest.mark.asyncio
async def test_mcp_client_calls_tool_and_returns_payload() -> None:
    session_cls = make_session(structured_result({"ok": True, "query": "sad healing", "limit": 3}))
    client = McpToolClient(
        server_url="http://localhost:8001/mcp",
        streamable_client=FakeStreamableClient,
        session_cls=session_cls,
    )

    result = await client.call_tool("echo_tool", {"query": "sad healing", "limit": 3})

    assert session_cls.calls == [("echo_tool", {"query": "sad healing", "limit": 3})]
    assert result["ok"] is True
    assert result["tool_name"] == "echo_tool"
    assert result["duration_ms"] >= 0
    assert result["error"] is None
    assert result["result"] == {"ok": True, "query": "sad healing", "limit": 3}


@pytest.mark.asyncio
async def test_mcp_client_records_call_input_and_result(tmp_path) -> None:
    client = McpToolClient(
        server_url="http://localhost:8001/mcp",
        streamable_client=FakeStreamableClient,
        session_cls=make_session(structured_result({"ok": True, "result_count": 1})),
    )
    log_path = tmp_path / "agent_loop.jsonl"

    with request_trace("request-1", log_path):
        await client.call_tool("music_rag_search", {"query": "calm healing"})

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["mcp_call_started", "mcp_call_completed"]
    assert events[0]["tool_input"] == {"query": "calm healing"}
    assert events[1]["result"] == {"ok": True, "result_count": 1}


@pytest.mark.asyncio
async def test_mcp_client_parses_json_text_fallback() -> None:
    client = McpToolClient(
        streamable_client=FakeStreamableClient,
        session_cls=make_session(text_result('{"ok": true, "value": 1}')),
    )

    result = await client.call_tool("json_text_tool", {})

    assert result["ok"] is True
    assert result["result"] == {"ok": True, "value": 1}


@pytest.mark.asyncio
async def test_mcp_client_returns_transport_error_for_unreachable_server() -> None:
    client = McpToolClient(
        server_url="http://localhost:1/mcp",
        timeout=0.2,
        streamable_client=FailingStreamableClient,
    )

    result = await client.call_tool("echo_tool", {"query": "sad"})

    assert result["ok"] is False
    assert result["result"] is None
    assert result["duration_ms"] >= 0
    assert result["error"]["error_code"] == "mcp_transport_error"
    assert "connect refused" in result["error"]["error"]


@pytest.mark.asyncio
async def test_mcp_client_maps_tool_error() -> None:
    client = McpToolClient(
        streamable_client=FakeStreamableClient,
        session_cls=make_session(text_result("tool exploded", is_error=True)),
    )

    result = await client.call_tool("error_tool", {})

    assert result["ok"] is False
    assert result["result"] is None
    assert result["duration_ms"] >= 0
    assert result["error"]["error_code"] == "mcp_tool_error"
    assert result["error"]["error"] == {"text": "tool exploded"}


# --- Real-protocol integration: route McpToolClient through the actual MCP server ---


class InMemoryFakeStore:
    """Deterministic store so the real MCP flow never touches Gemini."""

    def ensure_index(self) -> None:
        return None

    def search(self, search_input) -> MusicRagSearchResult:
        payload = SongPayload(
            chunk_id="spotify_track:s1",
            song_id="s1",
            title="After Rain",
            artist="Local Echo",
            metadata_summary="sad healing recovery",
            mood=["sad", "healing"],
            genres=["indie pop"],
            tags=["rain"],
        )
        return MusicRagSearchResult(
            ok=True,
            results=[payload],
            result_count=1,
            diagnostics={
                "record_count": 1,
                "score_details": {
                    "s1": {"score": 0.9, "semantic_score": 0.8, "metadata_boost": 0.1}
                },
            },
        )


@asynccontextmanager
async def in_memory_streamable(url, timeout=None, **kwargs):
    """Stand-in for streamablehttp_client that wires McpToolClient to the real server.

    It runs the real ``mcp._mcp_server`` on one end of an in-memory stream pair and hands
    the client end to ``McpToolClient``, so the client speaks the real MCP protocol
    (initialize handshake + tools/call) against the real tool implementations.
    """

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                lambda: mcp._mcp_server.run(
                    server_read,
                    server_write,
                    mcp._mcp_server.create_initialization_options(),
                    raise_exceptions=False,
                )
            )
            try:
                yield client_read, client_write, lambda: None
            finally:
                task_group.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_mcp_client_end_to_end_over_real_protocol() -> None:
    rag_tool.set_song_store_for_testing(InMemoryFakeStore())
    try:
        client = McpToolClient(
            server_url="memory://test",
            streamable_client=in_memory_streamable,
            session_cls=ClientSession,
        )
        result = await client.call_tool(
            "music_rag_search",
            {"query": "sad healing", "mood_terms": ["sad"], "limit": 3},
        )
    finally:
        rag_tool.set_song_store_for_testing(None)

    assert result["ok"] is True
    assert result["tool_name"] == "music_rag_search"
    assert result["error"] is None
    assert result["duration_ms"] >= 0

    payload = result["result"]
    assert payload["ok"] is True
    assert payload["result_count"] == 1
    assert payload["results"][0]["song_id"] == "s1"
    assert payload["results"][0]["title"] == "After Rain"
    assert payload["results"][0]["score"] == 0.9
