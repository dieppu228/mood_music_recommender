from music_agent.agent.mood_policy import apply_mood_policy
from music_agent.agent.state import ThinkDecision
from music_agent.models import AgentIntent, ToolName


def decision(*, action="call_tool", current=None, target=None, query="sad songs") -> ThinkDecision:
    is_call = action == "call_tool"
    return ThinkDecision.model_validate(
        {
            "thought": "Initial model decision.",
            "action": action,
            "intent": AgentIntent.MUSIC_RECOMMENDATION,
            "entities": {
                "mood_terms": current or [],
                "target_mood_terms": target or [],
            },
            "tool_name": ToolName.MUSIC_RAG_SEARCH if is_call else None,
            "tool_input": (
                {"query": query, "mood_terms": target or [], "limit": 5} if is_call else None
            ),
            "confidence": 0.9,
            "response": None if is_call else "Direct response.",
        }
    )


def test_current_sadness_without_music_preference_targets_calm_and_happy() -> None:
    result = apply_mood_policy(
        "Hiện tại tao đang buồn quá",
        decision(current=["sad"], target=["sad"]),
    )

    assert [mood.value for mood in result.entities.target_mood_terms] == ["calm", "happy"]
    assert result.tool_input["mood_terms"] == ["calm", "happy"]
    assert "sad" not in result.tool_input["query"]


def test_explicit_sad_music_preference_keeps_sad_target() -> None:
    result = apply_mood_policy(
        "Tao buồn, cho tao nghe nhạc buồn",
        decision(current=["sad"], target=["sad"]),
    )

    assert [mood.value for mood in result.entities.target_mood_terms] == ["sad"]
    assert result.tool_input["mood_terms"] == ["sad"]


def test_explicit_multiword_vietnamese_mood_maps_to_canonical_target() -> None:
    result = apply_mood_policy(
        "Cho tao nghe nhạc lãng mạn",
        decision(action="respond"),
    )

    assert [mood.value for mood in result.entities.target_mood_terms] == ["romantic"]
    assert result.tool_input["mood_terms"] == ["romantic"]


def test_insult_forces_calm_rag_and_apology_even_from_direct_decision() -> None:
    result = apply_mood_policy("Bot ngu quá", decision(action="respond"))

    assert result.action == "call_tool"
    assert result.tool_name == ToolName.MUSIC_RAG_SEARCH
    assert result.intent == AgentIntent.MUSIC_RECOMMENDATION
    assert result.entities.requires_apology is True
    assert result.entities.mood_terms == ["angry", "frustrated"]
    assert [mood.value for mood in result.entities.target_mood_terms] == ["calm"]
    assert result.tool_input["mood_terms"] == ["calm"]


def test_stress_targets_calm_and_tiredness_targets_energetic() -> None:
    stressed = apply_mood_policy(
        "Tao đang căng thẳng và lo lắng",
        decision(current=["stressed"], target=["stressed"]),
    )
    tired = apply_mood_policy(
        "Hôm nay tao mệt quá",
        decision(current=["tired"], target=[]),
    )

    assert [mood.value for mood in stressed.entities.target_mood_terms] == ["calm"]
    assert stressed.tool_input["mood_terms"] == ["calm"]
    assert [mood.value for mood in tired.entities.target_mood_terms] == ["energetic"]
    assert tired.tool_input["mood_terms"] == ["energetic"]
