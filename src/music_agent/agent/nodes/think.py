"""Think node: classify intent, extract entities, and plan the next action."""

from music_agent.agent.prompts import render_node_prompt, render_system_prompt
from music_agent.agent.state import AgentState, ThinkDecision
from music_agent.llm_client import LlmClient, LlmOutputError
from music_agent.models import PlannedToolCall


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

    state.intent = decision.intent
    state.entities = decision.entities
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
