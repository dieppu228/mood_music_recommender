"""FastAPI entrypoint for the music mood agent."""

from typing import Any

from fastapi import Depends, FastAPI

from music_agent.agent.graph import build_agent_graph
from music_agent.agent.state import AgentState
from music_agent.llm_client import LlmClient
from music_agent.models import AgentStatus, ChatRequest, ChatResponse
from music_agent.tools.mcp_client import McpToolClient

app = FastAPI(title="Music Mood Agent")
llm_client = LlmClient()
mcp_client = McpToolClient()
agent_graph = build_agent_graph(llm_client=llm_client, mcp_client=mcp_client)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def get_agent_graph() -> Any:
    return agent_graph


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, graph: Any = Depends(get_agent_graph)) -> ChatResponse:
    initial_state = AgentState(user_message=request.message).model_dump(mode="python")
    try:
        output = await graph.ainvoke(initial_state)
        state = AgentState.model_validate(output)
    except Exception:  # noqa: BLE001 - API boundary returns a structured failure.
        trace = {"errors": ["unexpected_agent_error"]} if request.debug else None
        return ChatResponse(
            status=AgentStatus.FAILED,
            answer="Mình chưa xử lý được yêu cầu này do lỗi hệ thống.",
            trace=trace,
        )

    return build_chat_response(state, request)


def build_chat_response(state: AgentState, request: ChatRequest) -> ChatResponse:
    status = state.status or infer_status(state)
    answer = state.final_answer or "Mình chưa có đủ context đáng tin cậy để trả lời yêu cầu này."
    return ChatResponse(
        status=status,
        answer=answer,
        recommendations=state.recommendations[: request.max_results],
        tool_calls=state.tool_calls,
        trace=build_trace(state) if request.debug else None,
    )


def infer_status(state: AgentState) -> AgentStatus:
    if state.final_answer or state.recommendations:
        return AgentStatus.OK
    return AgentStatus.FAILED


def build_trace(state: AgentState) -> dict[str, Any]:
    return {
        "scratchpad": state.scratchpad,
        "intent": state.intent,
        "entities": state.entities.model_dump(mode="json"),
        "confidence": state.confidence,
        "iteration_count": state.iteration_count,
        "errors": state.errors,
        "tool_calls": [tool_call.model_dump(mode="json") for tool_call in state.tool_calls],
    }
