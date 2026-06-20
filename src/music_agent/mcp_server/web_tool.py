"""MCP web search tool wrapper."""

import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import ValidationError

from music_agent.config import Settings, get_settings
from music_agent.models import SongPayload, WebSearchInput


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
    catalog_records: Sequence[SongPayload] | None = None,
) -> dict[str, Any]:
    """Run Tavily web search and return a normalized MCP payload.

    When ``catalog_records`` is provided, any catalog song whose title AND artist both
    appear in the web text is surfaced under ``catalog_matches`` as a playable
    recommendation. This applies to every ``search_intent`` (artist deep-dive and RAG
    fallback alike), so the trigger logic in the agent stays untouched.
    """

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

    web_text = build_web_text(raw_response.get("answer"), normalized_results)
    catalog_matches = (
        match_catalog_songs(web_text, catalog_records, parsed_input.limit)
        if catalog_records
        else []
    )

    return {
        "ok": True,
        "results": normalized_results,
        "sources": sources,
        "catalog_matches": catalog_matches,
        "diagnostics": {
            "search_intent": parsed_input.search_intent,
            "answer": raw_response.get("answer"),
            "result_count": len(normalized_results),
            "catalog_match_count": len(catalog_matches),
        },
    }


def build_web_text(answer: Any, normalized_results: Sequence[dict[str, Any]]) -> str:
    """Join the searchable free text from a Tavily response for catalog matching."""

    parts = [str(answer or "")]
    for item in normalized_results:
        parts.append(str(item.get("title") or ""))
        parts.append(str(item.get("content") or ""))
    return "\n".join(part for part in parts if part)


def match_catalog_songs(
    web_text: str,
    catalog_records: Sequence[SongPayload],
    limit: int,
) -> list[dict[str, Any]]:
    """Return catalog songs whose title AND artist both appear in the web text.

    Matching is deterministic phrase matching on normalized tokens. Requiring the artist
    to co-occur keeps precision high (avoids common one-word titles matching everything).
    """

    haystack = normalize_tokens(web_text)
    if not haystack:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in catalog_records:
        if record.song_id in seen:
            continue
        title_tokens = normalize_tokens(record.title)
        artist_tokens = normalize_tokens(record.artist)
        if not title_tokens or not artist_tokens:
            continue
        if phrase_in_tokens(haystack, title_tokens) and phrase_in_tokens(haystack, artist_tokens):
            matches.append(catalog_song_dict(record))
            seen.add(record.song_id)
            if len(matches) >= limit:
                break
    return matches


def catalog_song_dict(record: SongPayload) -> dict[str, Any]:
    """Normalize a catalog song into the same shape rag_tool emits (score is None)."""

    return {
        "song_id": record.song_id,
        "title": record.title,
        "artist": record.artist,
        "mood": list(record.mood),
        "genres": list(record.genres),
        "tags": list(record.tags),
        "preview_url": record.preview_url,
        "spotify_url": record.spotify_url,
        "score": None,
    }


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, strip accents, and split into alphanumeric tokens."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", ascii_text)


def phrase_in_tokens(haystack: Sequence[str], phrase: Sequence[str]) -> bool:
    """Return True if ``phrase`` appears as a contiguous run inside ``haystack``."""

    width = len(phrase)
    if width == 0 or width > len(haystack):
        return False
    phrase = list(phrase)
    return any(
        list(haystack[index : index + width]) == phrase
        for index in range(len(haystack) - width + 1)
    )
