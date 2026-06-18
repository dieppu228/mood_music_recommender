import pytest
from pydantic import ValidationError

from music_agent.agent.state import AgentState, FinalAnswerDraft, Observation, ThinkDecision
from music_agent.models import AgentIntent, AgentStatus, ToolName, WebSearchIntent


def test_agent_state_defaults_keep_observation_flags_in_scratchpad() -> None:
    state = AgentState(user_message="goi y nhac healing")

    assert state.iteration_count == 0
    assert state.scratchpad == {}
    assert state.recommendations == []
    assert state.errors == []

    with pytest.raises(ValidationError):
        AgentState(user_message="goi y nhac healing", tool_ok=True)


def test_think_decision_accepts_music_rag_tool_call() -> None:
    decision = ThinkDecision(
        thought="User asks for a mood recommendation.",
        action="call_tool",
        intent=AgentIntent.MUSIC_RECOMMENDATION,
        tool_name=ToolName.MUSIC_RAG_SEARCH,
        tool_input={
            "query": "sad healing indie songs",
            "mood_terms": ["sad", "healing"],
            "genres": ["indie"],
            "tags": ["night"],
            "artist": None,
            "limit": 5,
        },
        confidence=0.84,
        response=None,
    )

    assert decision.tool_name == ToolName.MUSIC_RAG_SEARCH
    assert decision.tool_input["limit"] == 5


def test_think_decision_accepts_web_search_tool_call() -> None:
    decision = ThinkDecision(
        thought="User asks for artist background.",
        action="call_tool",
        intent=AgentIntent.ARTIST_DEEP_DIVE,
        tool_name=ToolName.WEB_SEARCH,
        tool_input={
            "query": "Frank Ocean career background",
            "search_intent": WebSearchIntent.ARTIST_DEEP_DIVE,
            "limit": 3,
        },
        confidence=0.78,
        response=None,
    )

    assert decision.tool_name == ToolName.WEB_SEARCH


def test_think_decision_rejects_unknown_tool_name() -> None:
    with pytest.raises(ValidationError):
        ThinkDecision(
            thought="Need a tool.",
            action="call_tool",
            intent=AgentIntent.MUSIC_RECOMMENDATION,
            tool_name="unknown_tool",
            tool_input={"query": "sad songs", "limit": 5},
            confidence=0.7,
            response=None,
        )


def test_think_decision_accepts_null_tool_when_responding() -> None:
    decision = ThinkDecision(
        thought="Greeting can be answered directly.",
        action="respond",
        intent=AgentIntent.SMALLTALK,
        tool_name=None,
        tool_input=None,
        confidence=0.95,
        response="Chào bạn, bạn muốn nghe mood nào hôm nay?",
    )

    assert decision.tool_name is None
    assert decision.tool_input is None


def test_think_decision_rejects_tool_fields_when_responding() -> None:
    with pytest.raises(ValidationError):
        ThinkDecision(
            thought="Greeting can be answered directly.",
            action="respond",
            intent=AgentIntent.SMALLTALK,
            tool_name=ToolName.MUSIC_RAG_SEARCH,
            tool_input={"query": "hello", "limit": 5},
            confidence=0.9,
            response="Chào bạn.",
        )


def test_think_decision_rejects_missing_tool_name_when_calling_tool() -> None:
    with pytest.raises(ValidationError):
        ThinkDecision(
            thought="Need retrieval.",
            action="call_tool",
            intent=AgentIntent.MUSIC_RECOMMENDATION,
            tool_name=None,
            tool_input={"query": "healing songs", "limit": 5},
            confidence=0.8,
            response=None,
        )


def test_think_decision_rejects_invalid_tool_limit() -> None:
    with pytest.raises(ValidationError):
        ThinkDecision(
            thought="Need retrieval.",
            action="call_tool",
            intent=AgentIntent.MUSIC_RECOMMENDATION,
            tool_name=ToolName.MUSIC_RAG_SEARCH,
            tool_input={"query": "healing songs", "limit": 99},
            confidence=0.8,
            response=None,
        )


def test_final_answer_draft_rejects_trace_from_llm() -> None:
    with pytest.raises(ValidationError):
        FinalAnswerDraft(
            status=AgentStatus.OK,
            answer="Có vài bài phù hợp.",
            recommendations=[],
            trace={"scratchpad": "hidden"},
        )


def test_observation_models_scratchpad_keys() -> None:
    observation = Observation(
        tool_ok=True,
        enough_context=False,
        summary="RAG returned no result.",
        last_tool=ToolName.MUSIC_RAG_SEARCH,
        should_fallback_to_web=True,
    )

    assert observation.tool_ok is True
    assert observation.enough_context is False
    assert observation.should_fallback_to_web is True
