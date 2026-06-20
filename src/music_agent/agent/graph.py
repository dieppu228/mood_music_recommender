"""LangGraph assembly for the music mood agent."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from music_agent.agent.nodes import act_node, final_node, observe_node, think_node
from music_agent.agent.state import AgentState
from music_agent.agent.trace import write_trace_event
from music_agent.llm_client import LlmClient
from music_agent.tools.mcp_client import McpToolClient

MAX_TOOL_CALLS = 2


class AgentGraphState(TypedDict, total=False):
    user_message: str
    language: str
    status: str | None
    intent: str | None
    entities: dict[str, Any]
    planned_tool: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    scratchpad: dict[str, Any]
    recommendations: list[dict[str, Any]]
    final_answer: str | None
    confidence: float
    iteration_count: int
    errors: list[str]


def build_agent_graph(
    llm_client: LlmClient | None = None,
    mcp_client: McpToolClient | None = None,
):
    graph = StateGraph(AgentGraphState)

    async def think(state: AgentGraphState) -> Command:
        parsed = coerce_state(state)
        write_trace_event("node_started", node="think", state=trace_state(parsed))
        updated = await think_node(parsed, llm_client=llm_client)
        next_node = route_after_think(updated)
        rewritten_query = None
        if updated.planned_tool is not None:
            rewritten_query = updated.planned_tool.tool_input.get("query")
        write_trace_event(
            "node_completed",
            node="think",
            next_node=next_node,
            rewritten_query=rewritten_query,
            state=trace_state(updated),
        )
        return Command(update=state_update(updated), goto=next_node)

    async def act(state: AgentGraphState) -> Command:
        parsed = coerce_state(state)
        write_trace_event("node_started", node="act", state=trace_state(parsed))
        updated = await act_node(parsed, mcp_client=mcp_client)
        write_trace_event(
            "node_completed",
            node="act",
            next_node="observe",
            state=trace_state(updated),
        )
        return Command(update=state_update(updated), goto="observe")

    async def observe(state: AgentGraphState) -> Command:
        parsed = coerce_state(state)
        write_trace_event("node_started", node="observe", state=trace_state(parsed))
        updated = await observe_node(parsed)
        write_trace_event(
            "node_completed",
            node="observe",
            next_node="think",
            state=trace_state(updated),
        )
        return Command(update=state_update(updated), goto="think")

    async def final(state: AgentGraphState) -> Command:
        parsed = coerce_state(state)
        write_trace_event("node_started", node="final", state=trace_state(parsed))
        updated = await final_node(parsed, llm_client=llm_client)
        write_trace_event(
            "node_completed",
            node="final",
            next_node="end",
            state=trace_state(updated),
        )
        return Command(update=state_update(updated), goto=END)

    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("observe", observe)
    graph.add_node("final", final)
    graph.add_edge(START, "think")
    return graph.compile()


def route_after_think(state: AgentState | dict[str, Any]) -> str:
    parsed = coerce_state(state)
    if parsed.iteration_count >= MAX_TOOL_CALLS:
        return "final"
    return "act" if parsed.planned_tool is not None else "final"


def coerce_state(state: AgentState | dict[str, Any]) -> AgentState:
    return state if isinstance(state, AgentState) else AgentState.model_validate(state)


def state_update(state: AgentState) -> dict[str, Any]:
    return state.model_dump(mode="python")


def trace_state(state: AgentState) -> dict[str, Any]:
    return state.model_dump(mode="json")
