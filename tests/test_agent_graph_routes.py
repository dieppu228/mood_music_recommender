import pytest

from music_agent.agent.graph import build_agent_graph
from music_agent.agent.state import AgentState
from music_agent.models import AgentIntent, AgentStatus, ToolName, WebSearchIntent


class FakeLlmClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete_json(self, system_prompt, user_prompt, response_model, temperature=0.0):
        self.calls.append(response_model.__name__)
        output = self.outputs.pop(0)
        return response_model.model_validate(output)


class FakeMcpClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def call_tool(self, tool_name, tool_input):
        self.calls.append({"tool_name": tool_name, "tool_input": tool_input})
        return self.outputs.pop(0)


@pytest.mark.asyncio
async def test_greeting_routes_think_to_final_without_tool() -> None:
    llm = FakeLlmClient([think_respond(AgentIntent.SMALLTALK, "Chào bạn."), final_draft("Chào bạn.")])
    mcp = FakeMcpClient([])
    graph = build_agent_graph(llm_client=llm, mcp_client=mcp)

    output = await graph.ainvoke(AgentState(user_message="hello").model_dump(mode="python"))
    state = AgentState.model_validate(output)

    assert llm.calls == ["ThinkDecision", "FinalAnswerDraft"]
    assert mcp.calls == []
    assert state.final_answer == "Chào bạn."
    assert state.status == AgentStatus.OK


@pytest.mark.asyncio
async def test_mood_recommendation_routes_rag_observe_think_final() -> None:
    llm = FakeLlmClient(
        [
            think_call_rag(),
            think_respond(AgentIntent.MUSIC_RECOMMENDATION, "Use RAG recommendations."),
            final_draft(
                "Có 1 bài hợp mood.",
                recommendations=[
                    {
                        "song_id": "mock-001",
                        "title": "After Rain",
                        "artist": "Local Echo",
                        "reason": "Hợp mood sad/healing.",
                    }
                ],
            ),
        ]
    )
    mcp = FakeMcpClient([rag_wrapper([rag_song(score=0.92)])])
    graph = build_agent_graph(llm_client=llm, mcp_client=mcp)

    output = await graph.ainvoke(
        AgentState(user_message="goi y nhac buon healing").model_dump(mode="python")
    )
    state = AgentState.model_validate(output)

    assert llm.calls == ["ThinkDecision", "ThinkDecision", "FinalAnswerDraft"]
    assert [call["tool_name"] for call in mcp.calls] == ["music_rag_search"]
    assert state.iteration_count == 1
    assert state.scratchpad["enough_context"] is True
    assert state.recommendations[0].reason == "Hợp mood sad/healing."


@pytest.mark.asyncio
async def test_artist_deep_dive_routes_web_observe_think_final() -> None:
    llm = FakeLlmClient(
        [
            think_call_web(),
            think_respond(AgentIntent.ARTIST_DEEP_DIVE, "Use web context."),
            final_draft("Dựa trên web context: Frank Ocean là singer-songwriter."),
        ]
    )
    mcp = FakeMcpClient(
        [web_wrapper([{"title": "Bio", "url": "https://example.com/frank", "content": "Bio."}])]
    )
    graph = build_agent_graph(llm_client=llm, mcp_client=mcp)

    output = await graph.ainvoke(
        AgentState(user_message="noi ve Frank Ocean").model_dump(mode="python")
    )
    state = AgentState.model_validate(output)

    assert [call["tool_name"] for call in mcp.calls] == ["web_search"]
    assert state.scratchpad["last_tool"] == ToolName.WEB_SEARCH
    assert state.scratchpad["enough_context"] is True
    assert "web context" in state.final_answer


@pytest.mark.asyncio
async def test_empty_rag_falls_back_to_web_before_final() -> None:
    llm = FakeLlmClient(
        [
            think_call_rag(),
            think_call_web(search_intent=WebSearchIntent.FALLBACK_RECOMMENDATION),
            think_respond(AgentIntent.MUSIC_RECOMMENDATION, "Use web fallback."),
            final_draft("Không có RAG đủ tốt, dựa trên web context để trả lời."),
        ]
    )
    mcp = FakeMcpClient(
        [
            rag_wrapper([]),
            web_wrapper([{"title": "Fallback", "url": "https://example.com/music", "content": "Songs."}]),
        ]
    )
    graph = build_agent_graph(llm_client=llm, mcp_client=mcp)

    output = await graph.ainvoke(
        AgentState(user_message="goi y nhac rat hiem").model_dump(mode="python")
    )
    state = AgentState.model_validate(output)

    assert [call["tool_name"] for call in mcp.calls] == ["music_rag_search", "web_search"]
    assert llm.calls == ["ThinkDecision", "ThinkDecision", "ThinkDecision", "FinalAnswerDraft"]
    assert state.iteration_count == 2
    assert state.scratchpad["last_tool"] == ToolName.WEB_SEARCH


@pytest.mark.asyncio
async def test_tool_error_reaches_final_failure_response() -> None:
    llm = FakeLlmClient(
        [
            think_call_rag(),
            think_respond(AgentIntent.MUSIC_RECOMMENDATION, "Tool failed; answer safely."),
            final_draft("Mình chưa có đủ context đáng tin cậy.", status=AgentStatus.FAILED),
        ]
    )
    mcp = FakeMcpClient([tool_error_wrapper("music_rag_search")])
    graph = build_agent_graph(llm_client=llm, mcp_client=mcp)

    output = await graph.ainvoke(
        AgentState(user_message="goi y nhac buon").model_dump(mode="python")
    )
    state = AgentState.model_validate(output)

    assert state.status == AgentStatus.FAILED
    assert state.scratchpad["tool_ok"] is False
    assert state.errors


@pytest.mark.asyncio
async def test_graph_never_executes_more_than_two_tool_calls() -> None:
    llm = FakeLlmClient(
        [
            think_call_rag(),
            think_call_web(),
            think_call_rag(),
            final_draft("Hard stop sau 2 tool calls.", status=AgentStatus.FAILED),
        ]
    )
    mcp = FakeMcpClient(
        [
            rag_wrapper([]),
            web_wrapper([]),
        ]
    )
    graph = build_agent_graph(llm_client=llm, mcp_client=mcp)

    output = await graph.ainvoke(AgentState(user_message="goi y nhac").model_dump(mode="python"))
    state = AgentState.model_validate(output)

    assert [call["tool_name"] for call in mcp.calls] == ["music_rag_search", "web_search"]
    assert state.iteration_count == 2
    assert state.status == AgentStatus.FAILED


def think_call_rag() -> dict:
    return {
        "thought": "Need RAG.",
        "action": "call_tool",
        "intent": AgentIntent.MUSIC_RECOMMENDATION,
        "entities": {"mood_terms": ["sad"], "genres": [], "tags": [], "constraints": []},
        "tool_name": "music_rag_search",
        "tool_input": {
            "query": "sad healing songs",
            "mood_terms": ["sad"],
            "genres": [],
            "tags": [],
            "artist": None,
            "limit": 5,
        },
        "confidence": 0.8,
        "response": None,
    }


def think_call_web(search_intent=WebSearchIntent.ARTIST_DEEP_DIVE) -> dict:
    return {
        "thought": "Need web context.",
        "action": "call_tool",
        "intent": AgentIntent.ARTIST_DEEP_DIVE,
        "entities": {"mood_terms": [], "genres": [], "tags": [], "constraints": []},
        "tool_name": "web_search",
        "tool_input": {
            "query": "Frank Ocean career",
            "search_intent": search_intent,
            "limit": 5,
        },
        "confidence": 0.8,
        "response": None,
    }


def think_respond(intent: AgentIntent, response: str) -> dict:
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


def final_draft(answer: str, status=AgentStatus.OK, recommendations=None) -> dict:
    return {
        "status": status,
        "answer": answer,
        "recommendations": recommendations or [],
    }


def rag_wrapper(results) -> dict:
    return {
        "ok": True,
        "tool_name": "music_rag_search",
        "duration_ms": 12.0,
        "result": {"ok": True, "results": results, "result_count": len(results), "diagnostics": {}},
        "error": None,
    }


def web_wrapper(results) -> dict:
    return {
        "ok": True,
        "tool_name": "web_search",
        "duration_ms": 10.0,
        "result": {
            "ok": True,
            "results": results,
            "sources": [item["url"] for item in results if "url" in item],
            "diagnostics": {},
        },
        "error": None,
    }


def tool_error_wrapper(tool_name: str) -> dict:
    return {
        "ok": False,
        "tool_name": tool_name,
        "duration_ms": 8.0,
        "result": None,
        "error": {"error_code": "mcp_transport_error", "error": "connection failed"},
    }


def rag_song(score=0.92) -> dict:
    return {
        "song_id": "mock-001",
        "title": "After Rain",
        "artist": "Local Echo",
        "mood": ["sad", "healing"],
        "genres": ["indie pop"],
        "tags": ["rain"],
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/mock-001",
        "score": score,
    }
