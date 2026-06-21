"""FastAPI entrypoint for the music mood agent."""

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from music_agent.agent.graph import build_agent_graph
from music_agent.agent.state import AgentState
from music_agent.agent.trace import request_trace, write_trace_event
from music_agent.config import get_settings
from music_agent.llm_client import LlmClient
from music_agent.models import AgentStatus, ChatRequest, ChatResponse
from music_agent.tools.mcp_client import McpToolClient

app = FastAPI(title="Music Mood Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
llm_client = LlmClient()
mcp_client = McpToolClient()
agent_graph = build_agent_graph(llm_client=llm_client, mcp_client=mcp_client)
trace_log_path = get_settings().agent_trace_log_path


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def get_agent_graph() -> Any:
    return agent_graph


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, graph: Any = Depends(get_agent_graph)) -> ChatResponse:
    request_id = str(uuid4())
    started = time.perf_counter()
    with request_trace(request_id, trace_log_path):
        write_trace_event("request_received", request=request.model_dump(mode="json"))
        initial_state = AgentState(user_message=request.message).model_dump(mode="python")
        try:
            output = await graph.ainvoke(initial_state)
            state = AgentState.model_validate(output)
            response = build_chat_response(state, request)
        except Exception:  # noqa: BLE001 - API boundary returns a structured failure.
            trace = {"errors": ["unexpected_agent_error"]} if request.debug else None
            response = ChatResponse(
                status=AgentStatus.FAILED,
                answer="Mình chưa xử lý được yêu cầu này do lỗi hệ thống.",
                trace=trace,
            )
        write_trace_event(
            "request_completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            response=response.model_dump(mode="json"),
        )
        return response


def build_chat_response(state: AgentState, request: ChatRequest) -> ChatResponse:
    status = state.status or infer_status(state)
    answer = clean_answer_text(
        state.final_answer or "Mình chưa có đủ context đáng tin cậy để trả lời yêu cầu này."
    )
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


def clean_answer_text(answer: str) -> str:
    return answer.replace("**", "").replace("__", "")


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


dashboard_source_path = Path(__file__).resolve().parents[3] / "app"
dashboard_dist_path = dashboard_source_path / "dist"


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    index_path = dashboard_dist_path / "index.html"
    if not index_path.exists():
        index_path = dashboard_source_path / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/assets/{asset_path:path}")
async def dashboard_asset(asset_path: str) -> FileResponse:
    assets_root = (dashboard_dist_path / "assets").resolve()
    target = (assets_root / asset_path).resolve()
    if not target.is_relative_to(assets_root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)
