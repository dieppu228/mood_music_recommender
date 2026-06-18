import pytest

from music_agent.agent.nodes import act_node, final_node, observe_node, think_node
from music_agent.agent.state import AgentState, FinalAnswerDraft
from music_agent.llm_client import LlmOutputError
from music_agent.models import AgentIntent, AgentStatus, ToolName


class FakeLlmClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete_json(self, system_prompt, user_prompt, response_model, temperature=0.0):
        self.calls.append(
            {
                "response_model": response_model.__name__,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, response_model):
            return output
        return response_model.model_validate(output)


class FakeMcpClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def call_tool(self, tool_name, tool_input):
        self.calls.append({"tool_name": tool_name, "tool_input": tool_input})
        return self.outputs.pop(0)


def think_call(tool_name=ToolName.MUSIC_RAG_SEARCH, tool_input=None) -> dict:
    return {
        "thought": "Need tool evidence.",
        "action": "call_tool",
        "intent": AgentIntent.MUSIC_RECOMMENDATION,
        "entities": {"mood_terms": ["sad"], "genres": [], "tags": [], "constraints": []},
        "tool_name": tool_name,
        "tool_input": tool_input
        or {
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


def think_respond(response="Use existing evidence.") -> dict:
    return {
        "thought": "Enough context.",
        "action": "respond",
        "intent": AgentIntent.MUSIC_RECOMMENDATION,
        "entities": {"mood_terms": ["sad"], "genres": [], "tags": [], "constraints": []},
        "tool_name": None,
        "tool_input": None,
        "confidence": 0.9,
        "response": response,
    }


def final_ok(answer="Đây là vài bài phù hợp.") -> dict:
    return {"status": AgentStatus.OK, "answer": answer, "recommendations": []}


def rag_wrapper(results, ok=True) -> dict:
    return {
        "ok": ok,
        "tool_name": "music_rag_search",
        "duration_ms": 12.0,
        "result": {"ok": ok, "results": results, "result_count": len(results), "diagnostics": {}},
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


@pytest.mark.asyncio
async def test_think_sets_planned_tool_for_call_tool() -> None:
    state = AgentState(user_message="goi y nhac buon")
    llm = FakeLlmClient([think_call()])

    result = await think_node(state, llm_client=llm)

    assert result.planned_tool is not None
    assert result.planned_tool.tool_name == ToolName.MUSIC_RAG_SEARCH
    assert result.intent == AgentIntent.MUSIC_RECOMMENDATION
    assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_think_sets_planned_tool_none_when_responding() -> None:
    state = AgentState(user_message="hello")
    llm = FakeLlmClient([think_respond("Chào bạn.")])

    result = await think_node(state, llm_client=llm)

    assert result.planned_tool is None
    assert result.scratchpad["final_directive"] == "Chào bạn."


@pytest.mark.asyncio
async def test_think_llm_output_error_fails_safely() -> None:
    state = AgentState(user_message="goi y nhac")
    llm = FakeLlmClient([LlmOutputError("bad", raw_text="{", details="invalid json")])

    result = await think_node(state, llm_client=llm)

    assert result.planned_tool is None
    assert result.errors == ["think_llm_output_error: invalid json"]


@pytest.mark.asyncio
async def test_act_does_not_execute_without_planned_tool() -> None:
    state = AgentState(user_message="goi y nhac")
    mcp = FakeMcpClient([])

    result = await act_node(state, mcp_client=mcp)

    assert mcp.calls == []
    assert result.tool_result["ok"] is False
    assert result.tool_result["error"]["error_code"] == "missing_planned_tool"
    assert result.iteration_count == 0


@pytest.mark.asyncio
async def test_act_stores_mcp_wrapper_and_tool_trace() -> None:
    state = AgentState(user_message="goi y nhac")
    llm = FakeLlmClient([think_call()])
    state = await think_node(state, llm_client=llm)
    mcp = FakeMcpClient([rag_wrapper([rag_song()])])

    result = await act_node(state, mcp_client=mcp)

    assert result.tool_result["result"]["result_count"] == 1
    assert result.tool_calls[0].tool_name == "music_rag_search"
    assert result.iteration_count == 1


@pytest.mark.asyncio
async def test_observe_sets_scratchpad_flags_and_recommendations_for_rag() -> None:
    state = AgentState(
        user_message="goi y nhac",
        tool_result=rag_wrapper([rag_song(score=0.91)]),
    )

    result = await observe_node(state)

    assert result.scratchpad["tool_ok"] is True
    assert result.scratchpad["enough_context"] is True
    assert result.scratchpad["should_fallback_to_web"] is False
    assert result.recommendations[0].title == "After Rain"


@pytest.mark.asyncio
async def test_observe_sets_fallback_for_empty_rag() -> None:
    state = AgentState(user_message="goi y nhac", tool_result=rag_wrapper([]))

    result = await observe_node(state)

    assert result.scratchpad["tool_ok"] is True
    assert result.scratchpad["enough_context"] is False
    assert result.scratchpad["should_fallback_to_web"] is True


@pytest.mark.asyncio
async def test_observe_reads_web_payload_from_result_key() -> None:
    state = AgentState(
        user_message="tell me about frank ocean",
        tool_result=web_wrapper(
            [{"title": "Bio", "url": "https://example.com/frank", "content": "Career overview."}]
        ),
    )

    result = await observe_node(state)

    assert result.scratchpad["tool_ok"] is True
    assert result.scratchpad["enough_context"] is True
    assert result.scratchpad["sources"] == ["https://example.com/frank"]


@pytest.mark.asyncio
async def test_final_maps_draft_to_state_and_merges_reasons() -> None:
    state = AgentState(user_message="goi y nhac", recommendations=[rag_song_as_recommendation()])
    llm = FakeLlmClient(
        [
            FinalAnswerDraft(
                status=AgentStatus.OK,
                answer="Nghe thử After Rain.",
                recommendations=[
                    rag_song_as_recommendation(reason="Hợp mood sad/healing và có vibe mưa đêm.")
                ],
            )
        ]
    )

    result = await final_node(state, llm_client=llm)

    assert result.status == AgentStatus.OK
    assert result.final_answer == "Nghe thử After Rain."
    assert result.recommendations[0].reason == "Hợp mood sad/healing và có vibe mưa đêm."


@pytest.mark.asyncio
async def test_final_llm_output_error_fails_safely() -> None:
    state = AgentState(user_message="goi y nhac")
    llm = FakeLlmClient([LlmOutputError("bad", raw_text="{", details="invalid json")])

    result = await final_node(state, llm_client=llm)

    assert result.status == AgentStatus.FAILED
    assert "chưa có đủ context" in result.final_answer.lower()


def rag_song_as_recommendation(reason=""):
    from music_agent.models import Recommendation

    return Recommendation(
        song_id="mock-001",
        title="After Rain",
        artist="Local Echo",
        mood=["sad", "healing"],
        genres=["indie pop"],
        tags=["rain"],
        reason=reason,
        score=0.92,
    )
