from music_agent.config import Settings
from music_agent.mcp_server.web_tool import web_search


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
