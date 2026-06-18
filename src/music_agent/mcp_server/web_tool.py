"""MCP web search tool wrapper."""

from typing import Any, Protocol

from pydantic import ValidationError

from music_agent.config import Settings, get_settings
from music_agent.models import WebSearchInput


class TavilySearchClient(Protocol):
    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Run a Tavily search."""


def make_tavily_client(settings: Settings | None = None) -> TavilySearchClient:
    """Create a Tavily client from settings."""

    settings = settings or get_settings()
    if not settings.tavily_api_key:
        raise ValueError("missing_tavily_api_key")
    from tavily import TavilyClient

    return TavilyClient(api_key=settings.tavily_api_key)


def web_search(
    tool_input: WebSearchInput | dict[str, Any],
    client: TavilySearchClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run Tavily web search and return a normalized MCP payload."""

    try:
        parsed_input = WebSearchInput.model_validate(tool_input)
    except ValidationError as exc:
        return {
            "ok": False,
            "results": [],
            "sources": [],
            "diagnostics": {
                "error_code": "invalid_web_search_input",
                "error": str(exc),
            },
        }

    settings = settings or get_settings()
    if client is None and not settings.tavily_api_key:
        return {
            "ok": False,
            "results": [],
            "sources": [],
            "diagnostics": {
                "error_code": "missing_tavily_api_key",
                "error": "TAVILY_API_KEY is required for web_search",
            },
        }

    try:
        search_client = client or make_tavily_client(settings)
        raw_response = search_client.search(
            parsed_input.query,
            max_results=parsed_input.limit,
            search_depth="basic",
            include_answer=True,
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary must not crash agent loop.
        return {
            "ok": False,
            "results": [],
            "sources": [],
            "diagnostics": {
                "error_code": "tavily_search_error",
                "error": str(exc),
            },
        }

    raw_results = raw_response.get("results") or []
    normalized_results = []
    sources = []
    for item in raw_results[: parsed_input.limit]:
        url = item.get("url")
        if url:
            sources.append(url)
        normalized_results.append(
            {
                "title": item.get("title") or "",
                "url": url,
                "content": item.get("content") or item.get("snippet") or "",
                "score": item.get("score"),
            }
        )

    return {
        "ok": True,
        "results": normalized_results,
        "sources": sources,
        "diagnostics": {
            "search_intent": parsed_input.search_intent,
            "answer": raw_response.get("answer"),
            "result_count": len(normalized_results),
        },
    }
