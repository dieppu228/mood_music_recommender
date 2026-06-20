"""Think node: classify intent, extract entities, and plan the next action."""

from music_agent.agent.prompts import render_node_prompt, render_system_prompt
from music_agent.agent.mood_policy import apply_mood_policy
from music_agent.agent.state import AgentState, ThinkDecision
from music_agent.llm_client import LlmClient, LlmOutputError
from music_agent.models import ExtractedEntities, PlannedToolCall


async def think_node(state: AgentState, llm_client: LlmClient | None = None) -> AgentState:
    client = llm_client or LlmClient()

    try:
        decision = await client.complete_json(
            render_system_prompt(state),
            render_node_prompt("think", state),
            ThinkDecision,
            temperature=0.0,
        )
    except LlmOutputError as exc:
        state.errors.append(f"think_llm_output_error: {exc.details}")
        state.planned_tool = None
        state.scratchpad["final_directive"] = (
            "Không thể phân tích yêu cầu đủ tin cậy. Trả lời fail an toàn, không bịa dữ liệu."
        )
        return state

    if state.iteration_count == 0:
        decision = apply_mood_policy(state.user_message, decision)

    state.intent = decision.intent
    state.entities = merge_entities(state.entities, decision.entities)
    state.confidence = decision.confidence

    if decision.action == "call_tool":
        state.planned_tool = PlannedToolCall(
            tool_name=decision.tool_name,
            tool_input=decision.tool_input or {},
            reason=decision.thought,
            confidence=decision.confidence,
        )
        state.scratchpad.pop("final_directive", None)
        return state

    state.planned_tool = None
    if decision.response:
        state.scratchpad["final_directive"] = decision.response
    return state


def merge_entities(
    current: ExtractedEntities,
    incoming: ExtractedEntities,
) -> ExtractedEntities:
    """Preserve extracted context when later loop decisions omit unchanged entities."""

    return ExtractedEntities(
        mood_terms=incoming.mood_terms or current.mood_terms,
        target_mood_terms=incoming.target_mood_terms or current.target_mood_terms,
        requires_apology=incoming.requires_apology or current.requires_apology,
        genres=incoming.genres or current.genres,
        tags=incoming.tags or current.tags,
        artist=incoming.artist or current.artist,
        song_title=incoming.song_title or current.song_title,
        constraints=incoming.constraints or current.constraints,
    )
