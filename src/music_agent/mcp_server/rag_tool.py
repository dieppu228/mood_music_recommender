"""MCP RAG tool wrapper."""

from typing import Any

from pydantic import ValidationError

from music_agent.config import get_settings
from music_agent.models import MusicRagSearchInput
from music_agent.retrieval.fixture_store import FixtureSongStore, FixtureStoreError

_song_store: FixtureSongStore | None = None


def get_song_store() -> FixtureSongStore:
    """Return the process-wide fixture song store singleton."""

    global _song_store
    if _song_store is None:
        settings = get_settings()
        _song_store = FixtureSongStore(settings.mock_song_path, settings=settings)
    return _song_store


def set_song_store_for_testing(store: FixtureSongStore | None) -> None:
    """Override the singleton store in tests."""

    global _song_store
    _song_store = store


def warm_up_song_store() -> dict[str, Any]:
    """Build the singleton store index ahead of first tool call."""

    try:
        get_song_store().ensure_index()
    except FixtureStoreError as exc:
        return {
            "ok": False,
            "diagnostics": {
                "error_code": "fixture_store_warmup_failed",
                "error": str(exc),
            },
        }
    return {"ok": True, "diagnostics": {}}


def music_rag_search(
    tool_input: MusicRagSearchInput | dict[str, Any],
    store: FixtureSongStore | None = None,
) -> dict[str, Any]:
    """Search the fixture song store and return an MCP-friendly payload."""

    try:
        parsed_input = MusicRagSearchInput.model_validate(tool_input)
    except ValidationError as exc:
        return {
            "ok": False,
            "results": [],
            "result_count": 0,
            "diagnostics": {
                "error_code": "invalid_music_rag_search_input",
                "error": str(exc),
            },
        }

    try:
        result = (store or get_song_store()).search(parsed_input)
    except FixtureStoreError as exc:
        return {
            "ok": False,
            "results": [],
            "result_count": 0,
            "diagnostics": {
                "error_code": "fixture_store_error",
                "error": str(exc),
            },
        }

    score_details = result.diagnostics.get("score_details", {})
    normalized_results = []
    for item in result.results:
        score = score_details.get(item.song_id, {}).get("score")
        normalized_results.append(
            {
                "song_id": item.song_id,
                "title": item.title,
                "artist": item.artist,
                "mood": item.mood,
                "genres": item.genres,
                "tags": item.tags,
                "preview_url": item.preview_url,
                "spotify_url": item.spotify_url,
                "score": score,
            }
        )

    return {
        "ok": result.ok,
        "results": normalized_results,
        "result_count": len(normalized_results),
        "diagnostics": {
            "record_count": result.diagnostics.get("record_count", 0),
            "score_details": score_details,
            "embedding_model": result.diagnostics.get("embedding_model"),
            "document_task_type": result.diagnostics.get("document_task_type"),
            "query_task_type": result.diagnostics.get("query_task_type"),
        },
    }
