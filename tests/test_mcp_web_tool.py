from music_agent.config import Settings
from music_agent.mcp_server.web_tool import web_search
from music_agent.models import SongPayload


def catalog_song(song_id: str, title: str, artist: str, **extra) -> SongPayload:
    return SongPayload(
        chunk_id=f"spotify_track:{song_id}",
        song_id=song_id,
        title=title,
        artist=artist,
        metadata_summary=f"{title} by {artist}",
        **extra,
    )


class ContentTavilyClient:
    """Tavily stub returning a configurable answer + result contents."""

    def __init__(self, answer: str = "", contents: tuple[str, ...] = ()) -> None:
        self.answer = answer
        self.contents = list(contents)

    def search(self, query: str, **kwargs):
        return {
            "answer": self.answer,
            "results": [
                {
                    "title": "",
                    "url": f"https://example.com/{index}",
                    "content": content,
                    "score": 0.5,
                }
                for index, content in enumerate(self.contents)
            ],
        }


class FakeTavilyClient:
    def search(self, query: str, **kwargs):
        assert query == "Taylor Swift career"
        assert kwargs["max_results"] == 2
        return {
            "answer": "Taylor Swift is a singer-songwriter.",
            "results": [
                {
                    "title": "Taylor Swift Biography",
                    "url": "https://example.com/taylor",
                    "content": "Career overview and discography.",
                    "score": 0.95,
                },
                {
                    "title": "Taylor Swift Albums",
                    "url": "https://example.com/albums",
                    "content": "Album timeline.",
                    "score": 0.82,
                },
            ],
        }


def test_web_tool_returns_ok_false_when_tavily_key_is_missing() -> None:
    settings = Settings(tavily_api_key="")

    result = web_search(
        {
            "query": "Taylor Swift career",
            "search_intent": "artist_deep_dive",
            "limit": 2,
        },
        settings=settings,
    )

    assert result["ok"] is False
    assert result["results"] == []
    assert result["sources"] == []
    assert result["diagnostics"]["error_code"] == "missing_tavily_api_key"


def test_web_tool_maps_mocked_tavily_response_into_sources() -> None:
    settings = Settings(tavily_api_key="test-key")

    result = web_search(
        {
            "query": "Taylor Swift career",
            "search_intent": "artist_deep_dive",
            "limit": 2,
        },
        client=FakeTavilyClient(),
        settings=settings,
    )

    assert result["ok"] is True
    assert result["sources"] == ["https://example.com/taylor", "https://example.com/albums"]
    assert result["results"] == [
        {
            "title": "Taylor Swift Biography",
            "url": "https://example.com/taylor",
            "content": "Career overview and discography.",
            "score": 0.95,
        },
        {
            "title": "Taylor Swift Albums",
            "url": "https://example.com/albums",
            "content": "Album timeline.",
            "score": 0.82,
        },
    ]
    assert result["diagnostics"]["search_intent"] == "artist_deep_dive"
    assert result["diagnostics"]["result_count"] == 2


def test_web_tool_validates_required_input() -> None:
    result = web_search({"search_intent": "artist_deep_dive"}, client=FakeTavilyClient())

    assert result["ok"] is False
    assert result["diagnostics"]["error_code"] == "invalid_web_search_input"


def test_web_tool_surfaces_catalog_song_mentioned_in_web_text() -> None:
    catalog = [
        catalog_song("sad-001", "After Rain", "Local Echo", preview_url="https://p/sad.mp3"),
        catalog_song("energy-001", "Neon Drive", "Pulse City"),
    ]
    client = ContentTavilyClient(
        answer="After Rain by Local Echo is a soothing healing track.",
        contents=("Some unrelated review text.",),
    )

    result = web_search(
        {"query": "healing songs", "search_intent": "fallback_recommendation", "limit": 5},
        client=client,
        settings=Settings(tavily_api_key="k"),
        catalog_records=catalog,
    )

    assert result["ok"] is True
    assert [m["song_id"] for m in result["catalog_matches"]] == ["sad-001"]
    assert result["catalog_matches"][0]["preview_url"] == "https://p/sad.mp3"
    assert result["catalog_matches"][0]["score"] is None
    assert result["diagnostics"]["catalog_match_count"] == 1


def test_web_tool_catalog_match_is_intent_agnostic() -> None:
    catalog = [catalog_song("sad-001", "After Rain", "Local Echo")]
    client = ContentTavilyClient(answer="After Rain by Local Echo, a classic.", contents=())

    result = web_search(
        {"query": "Local Echo", "search_intent": "artist_deep_dive", "limit": 5},
        client=client,
        settings=Settings(tavily_api_key="k"),
        catalog_records=catalog,
    )

    assert [m["song_id"] for m in result["catalog_matches"]] == ["sad-001"]


def test_web_tool_requires_both_title_and_artist_to_match() -> None:
    catalog = [catalog_song("sad-001", "After Rain", "Local Echo")]
    client = ContentTavilyClient(
        answer="After Rain is a nice phrase about the weather.",
        contents=("No artist named in this snippet.",),
    )

    result = web_search(
        {"query": "weather", "search_intent": "fallback_recommendation", "limit": 5},
        client=client,
        settings=Settings(tavily_api_key="k"),
        catalog_records=catalog,
    )

    assert result["catalog_matches"] == []
    assert result["diagnostics"]["catalog_match_count"] == 0


def test_web_tool_no_catalog_matches_without_records() -> None:
    client = ContentTavilyClient(answer="After Rain by Local Echo.", contents=())

    result = web_search(
        {"query": "x", "search_intent": "artist_deep_dive", "limit": 5},
        client=client,
        settings=Settings(tavily_api_key="k"),
    )

    assert result["catalog_matches"] == []
    assert result["diagnostics"]["catalog_match_count"] == 0
