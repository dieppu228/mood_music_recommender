import pytest

from music_agent.models import MusicRagSearchInput, MusicRagSearchResult, SongPayload
from music_agent.mcp_server.rag_tool import music_rag_search
from music_agent.retrieval.fixture_store import FixtureStoreError


class SuccessfulStore:
    def search(self, search_input: MusicRagSearchInput) -> MusicRagSearchResult:
        assert search_input.query == "sad healing"
        payload = SongPayload(
            chunk_id="spotify_track:mock-001",
            song_id="mock-001",
            title="After Rain",
            artist="Local Echo",
            artists=["Local Echo"],
            metadata_summary="sad healing recovery",
            mood=["sad", "healing"],
            genres=["indie pop"],
            tags=["rain"],
            preview_url=None,
            spotify_url="https://open.spotify.com/track/mock-001",
        )
        return MusicRagSearchResult(
            ok=True,
            results=[payload],
            result_count=1,
            diagnostics={
                "record_count": 1,
                "embedding_model": "gemini-embedding-001",
                "document_task_type": "RETRIEVAL_DOCUMENT",
                "query_task_type": "RETRIEVAL_QUERY",
                "score_details": {
                    "mock-001": {
                        "score": 0.92,
                        "semantic_score": 0.84,
                        "metadata_boost": 0.08,
                    }
                },
            },
        )


class FailingStore:
    def search(self, search_input: MusicRagSearchInput) -> MusicRagSearchResult:
        raise FixtureStoreError("fixture failed")


def test_rag_tool_validates_required_input() -> None:
    result = music_rag_search({"mood_terms": ["sad"]}, store=SuccessfulStore())

    assert result["ok"] is False
    assert result["results"] == []
    assert result["diagnostics"]["error_code"] == "invalid_music_rag_search_input"


def test_rag_tool_returns_normalized_results_with_score() -> None:
    result = music_rag_search(
        {
            "query": "sad healing",
            "mood_terms": ["sad", "healing"],
            "genres": [],
            "tags": ["rain"],
            "artist": None,
            "limit": 5,
        },
        store=SuccessfulStore(),
    )

    assert result["ok"] is True
    assert result["result_count"] == 1
    assert result["results"] == [
        {
            "song_id": "mock-001",
            "title": "After Rain",
            "artist": "Local Echo",
            "mood": ["sad", "healing"],
            "genres": ["indie pop"],
            "tags": ["rain"],
            "preview_url": None,
            "spotify_url": "https://open.spotify.com/track/mock-001",
            "score": 0.92,
        }
    ]
    assert result["diagnostics"]["document_task_type"] == "RETRIEVAL_DOCUMENT"
    assert result["diagnostics"]["query_task_type"] == "RETRIEVAL_QUERY"


def test_rag_tool_returns_ok_false_when_fixture_store_fails() -> None:
    result = music_rag_search({"query": "sad healing"}, store=FailingStore())

    assert result["ok"] is False
    assert result["results"] == []
    assert result["result_count"] == 0
    assert result["diagnostics"]["error_code"] == "fixture_store_error"
    assert "fixture failed" in result["diagnostics"]["error"]


@pytest.mark.asyncio
async def test_mcp_server_registers_expected_tools() -> None:
    from music_agent.mcp_server.server import mcp

    tools = await mcp.list_tools()

    assert sorted(tool.name for tool in tools) == ["music_rag_search", "web_search"]
