"""Act node: execute the selected MCP tool."""

from music_agent.agent.state import AgentState
from music_agent.models import ToolCallTrace
from music_agent.tools.mcp_client import McpToolClient


async def act_node(state: AgentState, mcp_client: McpToolClient | None = None) -> AgentState:
    planned_tool = state.planned_tool
    if planned_tool is None or planned_tool.tool_name is None:
        state.tool_result = {
            "ok": False,
            "tool_name": None,
            "duration_ms": 0.0,
            "result": None,
            "error": {
                "error_code": "missing_planned_tool",
                "error": "act node requires planned_tool",
            },
        }
        state.errors.append("missing_planned_tool")
        return state

    client = mcp_client or McpToolClient()
    wrapper = await client.call_tool(str(planned_tool.tool_name), planned_tool.tool_input)
    state.tool_result = wrapper
    state.tool_calls.append(
        ToolCallTrace(
            tool_name=str(planned_tool.tool_name),
            tool_input=planned_tool.tool_input,
            reason=planned_tool.reason,
            confidence=planned_tool.confidence,
            ok=wrapper.get("ok"),
            duration_ms=wrapper.get("duration_ms"),
        )
    )
    state.iteration_count += 1
    return state
