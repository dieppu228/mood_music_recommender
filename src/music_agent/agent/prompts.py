"""Prompt rendering helpers for agent nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from music_agent.agent.state import AgentState

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(relative_path: str) -> str:
    return (PROMPT_ROOT / relative_path).read_text(encoding="utf-8")


def render_system_prompt(state: AgentState, *, history: str = "Không có lịch sử") -> str:
    scratchpad = json.dumps(state.scratchpad, ensure_ascii=False, indent=2)
    values = {
        "agent_name": "Music Mood Agent",
        "agent_description": "Agent gợi ý bài hát theo mood, genre, tag và hỗ trợ tra cứu nghệ sĩ.",
        "agent_system_prompt": (
            "Ưu tiên RAG nội bộ cho gợi ý nhạc. Dùng web_search khi cần facts nghệ sĩ "
            "hoặc khi RAG không đủ context."
        ),
        "history": history,
        "scratchpad": scratchpad,
        "user_message": state.user_message,
    }
    return render_template(load_prompt("system.md"), values)


def render_node_prompt(node_name: str, state: AgentState) -> str:
    state_payload = {
        "user_message": state.user_message,
        "language": state.language,
        "intent": state.intent,
        "entities": state.entities.model_dump(),
        "planned_tool": state.planned_tool.model_dump() if state.planned_tool else None,
        "tool_result": state.tool_result,
        "scratchpad": state.scratchpad,
        "recommendations": [item.model_dump() for item in state.recommendations],
        "confidence": state.confidence,
        "iteration_count": state.iteration_count,
        "errors": state.errors,
    }
    return (
        load_prompt(f"nodes/{node_name}.md")
        + "\n\n## State hiện tại\n"
        + json.dumps(state_payload, ensure_ascii=False, indent=2, default=str)
    )


def render_template(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered
