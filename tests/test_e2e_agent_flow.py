import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from music_agent.agent.graph import build_agent_graph
from music_agent.api.main import app, get_agent_graph
from music_agent.config import Settings
from music_agent.mcp_server.rag_tool import music_rag_search
from music_agent.mcp_server.web_tool import web_search
from music_agent.models import AgentIntent, AgentStatus, ToolName, WebSearchIntent
from music_agent.retrieval.fixture_store import FixtureSongStore


class KeywordEmbeddingClient:
    dimensions = {
        "sad": 0,
        "healing": 1,
        "rare": 2,
        "artist": 3,
        "local echo": 4,
    }

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        vector = [0.0] * len(self.dimensions)
        for keyword, index in self.dimensions.items():
            if keyword in lowered:
                vector[index] = 1.0
        return vector


class FakeLlmClient:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.calls: list[str] = []

    async def complete_json(self, system_prompt, user_prompt, response_model, temperature=0.0):
        self.calls.append(response_model.__name__)
        output = self.outputs.pop(0)
        return response_model.model_validate(output)


class E2EMcpClient:
    def __init__(
        self,
        store: FixtureSongStore | None = None,
        web_client: "FakeTavilyClient | None" = None,
        fail_tools: set[str] | None = None,
    ) -> None:
        self.store = store
        self.web_client = web_client or FakeTavilyClient()
        self.fail_tools = fail_tools or set()
        self.calls: list[dict[str, Any]] = []

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"tool_name": tool_name, "tool_input": tool_input})
        if tool_name in self.fail_tools:
            return tool_wrapper(
                ok=False,
                tool_name=tool_name,
                result=None,
                error={"error_code": "e2e_tool_failure", "error": "forced failure"},
            )

        if tool_name == ToolName.MUSIC_RAG_SEARCH:
            result = music_rag_search(tool_input, store=self.store)
        elif tool_name == ToolName.WEB_SEARCH:
            result = web_search(
                tool_input,
                client=self.web_client,
                settings=Settings(tavily_api_key="test-key"),
            )
        else:
            result = {
                "ok": False,
                "diagnostics": {"error_code": "unknown_tool", "error": tool_name},
            }

        return tool_wrapper(
            ok=bool(result.get("ok")),
            tool_name=tool_name,
            result=result,
            error=None if result.get("ok") else result.get("diagnostics"),
        )


class FakeTavilyClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {
            "results": [
                {
                    "title": "Artist background",
                    "url": "https://example.com/artist",
                    "content": "Public career context and related music notes.",
                    "score": 0.91,
                }
            ]
        }


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_e2e_greeting_has_no_tool_calls() -> None:
    llm = FakeLlmClient(
        [
            think_respond(AgentIntent.SMALLTALK, "Chào bạn."),
            final_draft("Chào bạn, bạn muốn nghe mood nào hôm nay?"),
        ]
    )
    mcp = E2EMcpClient()
    override_graph(build_agent_graph(llm_client=llm, mcp_client=mcp))

    response = await post_chat({"message": "hello", "debug": True})

    assert response["status"] == "ok"
    assert response["tool_calls"] == []
    assert response["trace"]["iteration_count"] == 0
    assert mcp.calls == []


@pytest.mark.asyncio
async def test_e2e_sad_healing_returns_fixture_rag_recommendations(tmp_path: Path) -> None:
    store = build_store(
        tmp_path,
        [
            song("sad-001", "After Rain", "Local Echo", "sad healing recovery"),
            song("other-001", "Rare Static", "Noise Lab", "rare unknown signal"),
        ],
    )
    llm = FakeLlmClient(
        [
            think_call_rag("sad healing songs", mood_terms=["sad", "healing"]),
            think_respond(AgentIntent.MUSIC_RECOMMENDATION, "Use RAG recommendations."),
            final_draft(
                "Gợi ý hợp mood sad/healing: After Rain.",
                recommendations=[
                    {
                        "song_id": "sad-001",
                        "title": "After Rain",
                        "artist": "Local Echo",
                        "reason": "Hợp mood sad/healing và có cảm giác hồi phục.",
                    }
                ],
            ),
        ]
    )
    mcp = E2EMcpClient(store=store)
    override_graph(build_agent_graph(llm_client=llm, mcp_client=mcp))

    response = await post_chat({"message": "goi y nhac buon healing", "debug": True})

    assert response["status"] == "ok"
    assert response["recommendations"][0]["title"] == "After Rain"
    assert response["recommendations"][0]["reason"] == "Hợp mood sad/healing và có cảm giác hồi phục."
    assert [call["tool_name"] for call in response["tool_calls"]] == ["music_rag_search"]
    assert response["trace"]["scratchpad"]["enough_context"] is True


@pytest.mark.asyncio
async def test_e2e_unknown_recommendation_falls_back_from_rag_to_web(tmp_path: Path) -> None:
    store = build_store(tmp_path, [])
    llm = FakeLlmClient(
        [
            think_call_rag("rare unknown songs", mood_terms=["rare"]),
            think_call_web(WebSearchIntent.FALLBACK_RECOMMENDATION),
            think_respond(AgentIntent.MUSIC_RECOMMENDATION, "Use web fallback context."),
            final_draft("Không có RAG đủ tốt, nên câu trả lời dựa trên web context."),
        ]
    )
    mcp = E2EMcpClient(store=store)
    override_graph(build_agent_graph(llm_client=llm, mcp_client=mcp))

    response = await post_chat({"message": "goi y nhac rare unknown", "debug": True})

    assert response["status"] == "ok"
    assert [call["tool_name"] for call in response["tool_calls"]] == [
        "music_rag_search",
        "web_search",
    ]
    assert response["trace"]["scratchpad"]["last_tool"] == "web_search"
    assert "web context" in response["answer"]


@pytest.mark.asyncio
async def test_e2e_artist_deep_dive_uses_web_first() -> None:
    llm = FakeLlmClient(
        [
            think_call_web(WebSearchIntent.ARTIST_DEEP_DIVE),
            think_respond(AgentIntent.ARTIST_DEEP_DIVE, "Use web context."),
            final_draft("Dựa trên web context: Local Echo là artist trong nguồn test."),
        ]
    )
    mcp = E2EMcpClient()
    override_graph(build_agent_graph(llm_client=llm, mcp_client=mcp))

    response = await post_chat({"message": "noi sau ve Local Echo", "debug": True})

    assert response["status"] == "ok"
    assert [call["tool_name"] for call in response["tool_calls"]] == ["web_search"]
    assert response["trace"]["scratchpad"]["last_tool"] == "web_search"
    assert "web context" in response["answer"]


@pytest.mark.asyncio
async def test_e2e_tool_failure_returns_non_fabricated_failure_answer(tmp_path: Path) -> None:
    store = build_store(tmp_path, [song("sad-001", "After Rain", "Local Echo", "sad healing")])
    llm = FakeLlmClient(
        [
            think_call_rag("sad healing songs", mood_terms=["sad"]),
            think_respond(AgentIntent.MUSIC_RECOMMENDATION, "Tool failed; answer safely."),
            final_draft(
                "Mình chưa có đủ context đáng tin cậy để gợi ý bài hát.",
                status=AgentStatus.FAILED,
            ),
        ]
    )
    mcp = E2EMcpClient(store=store, fail_tools={str(ToolName.MUSIC_RAG_SEARCH)})
    override_graph(build_agent_graph(llm_client=llm, mcp_client=mcp))

    response = await post_chat({"message": "goi y nhac buon", "debug": True})

    assert response["status"] == "failed"
    assert response["recommendations"] == []
    assert "After Rain" not in response["answer"]
    assert response["trace"]["errors"]


async def post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    return response.json()


def override_graph(graph) -> None:
    async def override():
        return graph

    app.dependency_overrides[get_agent_graph] = override


def build_store(tmp_path: Path, rows: list[dict[str, Any]]) -> FixtureSongStore:
    path = tmp_path / "songs.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())


def song(song_id: str, title: str, artist: str, summary: str) -> dict[str, Any]:
    return {
        "chunk_id": f"spotify_track:{song_id}",
        "song_id": song_id,
        "title": title,
        "artist": artist,
        "artists": [artist],
        "album": "E2E Album",
        "release_date": "2024-01-01",
        "release_year": 2024,
        "duration_ms": 180000,
        "popularity": 50,
        "explicit": False,
        "metadata_summary": summary,
        "lyrics_summary": summary,
        "lyrics_available": True,
        "mood": ["sad", "healing"] if "sad" in summary else ["rare"],
        "genres": ["indie pop"],
        "tags": ["rain"] if "healing" in summary else ["unknown"],
        "preview_url": None,
        "spotify_url": f"https://open.spotify.com/track/{song_id}",
        "source_name": "E2E Test",
        "source_type": "mock",
        "search_query": summary,
        "mood_inferred": False,
        "data_origin": "test",
        "payload_version": "v1",
    }


def tool_wrapper(
    *,
    ok: bool,
    tool_name: str,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "tool_name": tool_name,
        "duration_ms": 1.0,
        "result": result,
        "error": error,
    }


def think_call_rag(query: str, mood_terms: list[str] | None = None) -> dict[str, Any]:
    return {
        "thought": "Need fixture RAG.",
        "action": "call_tool",
        "intent": AgentIntent.MUSIC_RECOMMENDATION,
        "entities": {
            "mood_terms": mood_terms or [],
            "genres": [],
            "tags": [],
            "constraints": [],
        },
        "tool_name": "music_rag_search",
        "tool_input": {
            "query": query,
            "mood_terms": mood_terms or [],
            "genres": [],
            "tags": [],
            "artist": None,
            "limit": 5,
        },
        "confidence": 0.86,
        "response": None,
    }


def think_call_web(search_intent: WebSearchIntent) -> dict[str, Any]:
    return {
        "thought": "Need web context.",
        "action": "call_tool",
        "intent": AgentIntent.ARTIST_DEEP_DIVE,
        "entities": {"mood_terms": [], "genres": [], "tags": [], "constraints": []},
        "tool_name": "web_search",
        "tool_input": {
            "query": "Local Echo artist background",
            "search_intent": search_intent,
            "limit": 5,
        },
        "confidence": 0.82,
        "response": None,
    }


def think_respond(intent: AgentIntent, response: str) -> dict[str, Any]:
    return {
        "thought": "Enough context.",
        "action": "respond",
        "intent": intent,
        "entities": {"mood_terms": [], "genres": [], "tags": [], "constraints": []},
        "tool_name": None,
        "tool_input": None,
        "confidence": 0.9,
        "response": response,
    }


def final_draft(
    answer: str,
    status: AgentStatus = AgentStatus.OK,
    recommendations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "answer": answer,
        "recommendations": recommendations or [],
    }
