"""Shared Pydantic contracts for API, agent state, retrieval, and MCP tools."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    OUT_OF_DOMAIN = "out_of_domain"


class AgentIntent(StrEnum):
    SMALLTALK = "smalltalk"
    MUSIC_RECOMMENDATION = "music_recommendation"
    ARTIST_DEEP_DIVE = "artist_deep_dive"
    OUT_OF_DOMAIN = "out_of_domain"


class CanonicalMood(StrEnum):
    HAPPY = "happy"
    SAD = "sad"
    CALM = "calm"
    ENERGETIC = "energetic"
    ROMANTIC = "romantic"
    STRESSED = "stressed"


class ToolName(StrEnum):
    MUSIC_RAG_SEARCH = "music_rag_search"
    WEB_SEARCH = "web_search"


class WebSearchIntent(StrEnum):
    ARTIST_DEEP_DIVE = "artist_deep_dive"
    FALLBACK_RECOMMENDATION = "fallback_recommendation"


class SongPayload(BaseModel):
    """Payload stored per song chunk and returned by retrieval tools."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    song_id: str
    title: str
    artist: str
    album: str | None = None
    metadata_summary: str
    lyrics_summary: str | None = None
    mood: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    preview_url: str | None = None
    spotify_url: str | None = None
    payload_version: str = "v1"

    @field_validator("title", "artist", "metadata_summary")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

class Recommendation(BaseModel):
    song_id: str
    title: str
    artist: str
    mood: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reason: str = ""
    preview_url: str | None = None
    spotify_url: str | None = None
    score: float | None = None


class ToolCallTrace(BaseModel):
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ok: bool | None = None
    duration_ms: float | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    max_results: int = Field(default=5, ge=1, le=10)
    debug: bool = False


class ChatResponse(BaseModel):
    status: AgentStatus
    answer: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    trace: dict[str, Any] | None = None


class ExtractedEntities(BaseModel):
    mood_terms: list[str] = Field(default_factory=list)
    target_mood_terms: list[CanonicalMood] = Field(default_factory=list)
    requires_apology: bool = False
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    artist: str | None = None
    song_title: str | None = None
    constraints: list[str] = Field(default_factory=list)


class PlannedToolCall(BaseModel):
    tool_name: ToolName | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MusicRagSearchInput(BaseModel):
    query: str = Field(min_length=1)
    mood_terms: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    artist: str | None = None
    limit: int = Field(default=5, ge=1, le=10)


class MusicRagSearchResult(BaseModel):
    ok: bool
    results: list[SongPayload] = Field(default_factory=list)
    result_count: int = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1)
    search_intent: WebSearchIntent
    limit: int = Field(default=5, ge=1, le=10)


class WebSearchResult(BaseModel):
    ok: bool
    results: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
