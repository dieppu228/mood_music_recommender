import pytest
from pydantic import ValidationError

from music_agent.models import (
    AgentStatus,
    CanonicalMood,
    ChatRequest,
    ChatResponse,
    ExtractedEntities,
    MusicRagSearchInput,
    PlannedToolCall,
    SongPayload,
    ToolName,
    WebSearchInput,
    WebSearchIntent,
)


def test_chat_request_defaults() -> None:
    request = ChatRequest(message="goi y nhac chill")
    assert request.max_results == 5
    assert request.debug is False


def test_chat_request_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_response_serializes_trace_null() -> None:
    response = ChatResponse(status=AgentStatus.OK, answer="ok")
    assert response.model_dump()["trace"] is None


def test_music_rag_search_input_rejects_unsafe_limit() -> None:
    with pytest.raises(ValidationError):
        MusicRagSearchInput(query="sad healing", limit=0)

    with pytest.raises(ValidationError):
        MusicRagSearchInput(query="sad healing", limit=11)


def test_planned_tool_call_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        PlannedToolCall(tool_name="unknown_tool")


def test_web_search_input_uses_known_search_intent() -> None:
    request = WebSearchInput(
        query="Taylor Swift career",
        search_intent=WebSearchIntent.ARTIST_DEEP_DIVE,
    )
    assert request.search_intent == WebSearchIntent.ARTIST_DEEP_DIVE


def test_song_payload_validates_chunk_payload() -> None:
    payload = SongPayload(
        chunk_id="spotify_track:mock-001",
        song_id="mock-001",
        title="After Rain",
        artist="Local Echo",
        metadata_summary="A reflective song about sadness and recovery.",
        lyrics_summary="A song about becoming lighter after grief.",
        mood=["sad", "healing"],
        genres=["indie pop"],
        tags=["rain", "night"],
    )
    assert payload.title == "After Rain"
    assert payload.tags == ["rain", "night"]
    assert set(payload.model_dump()) == {
        "chunk_id",
        "song_id",
        "title",
        "artist",
        "album",
        "metadata_summary",
        "lyrics_summary",
        "mood",
        "genres",
        "tags",
        "preview_url",
        "spotify_url",
        "payload_version",
    }


def test_extracted_entities_accepts_only_canonical_target_moods() -> None:
    entities = ExtractedEntities(
        mood_terms=["buon"],
        target_mood_terms=["calm", "happy"],
        requires_apology=True,
    )

    assert entities.target_mood_terms == [CanonicalMood.CALM, CanonicalMood.HAPPY]
    assert entities.requires_apology is True

    with pytest.raises(ValidationError):
        ExtractedEntities(target_mood_terms=["healing"])


def test_planned_tool_call_accepts_known_tool_name() -> None:
    planned = PlannedToolCall(tool_name=ToolName.MUSIC_RAG_SEARCH, confidence=0.8)
    assert planned.tool_name == ToolName.MUSIC_RAG_SEARCH
