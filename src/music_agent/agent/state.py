"""Typed state models for the LangGraph agent."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from music_agent.models import (
    AgentIntent,
    AgentStatus,
    ExtractedEntities,
    MusicRagSearchInput,
    PlannedToolCall,
    Recommendation,
    ToolCallTrace,
    ToolName,
    WebSearchInput,
)


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_message: str
    language: str = "auto"
    status: AgentStatus | None = None
    intent: AgentIntent | None = None
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    planned_tool: PlannedToolCall | None = None
    tool_result: dict[str, Any] | None = None
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[Recommendation] = Field(default_factory=list)
    final_answer: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    iteration_count: int = 0
    errors: list[str] = Field(default_factory=list)


class ThinkDecision(BaseModel):
    """Validated JSON output from the think node LLM call."""

    model_config = ConfigDict(extra="forbid")

    thought: str
    action: Literal["call_tool", "respond"]
    intent: AgentIntent
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    tool_name: ToolName | None = None
    tool_input: dict[str, Any] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    response: str | None = None

    @model_validator(mode="after")
    def validate_action_contract(self) -> "ThinkDecision":
        if self.action == "respond":
            if self.tool_name is not None or self.tool_input is not None:
                raise ValueError("respond action must not include tool_name or tool_input")
            return self

        if self.tool_name is None:
            raise ValueError("call_tool action requires tool_name")
        if self.tool_input is None:
            raise ValueError("call_tool action requires tool_input")

        if self.tool_name == ToolName.MUSIC_RAG_SEARCH:
            MusicRagSearchInput.model_validate(self.tool_input)
        elif self.tool_name == ToolName.WEB_SEARCH:
            WebSearchInput.model_validate(self.tool_input)
        return self


class FinalAnswerDraft(BaseModel):
    """Validated JSON output from the final node LLM call."""

    model_config = ConfigDict(extra="forbid")

    status: AgentStatus
    answer: str = Field(min_length=1)
    recommendations: list[Recommendation] = Field(default_factory=list)


class Observation(BaseModel):
    """Internal helper for deterministic observe-node scratchpad updates."""

    tool_ok: bool
    enough_context: bool
    summary: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    last_tool: ToolName | None = None
    should_fallback_to_web: bool = False
    errors: list[str] = Field(default_factory=list)
