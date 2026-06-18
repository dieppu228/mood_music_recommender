"""HTTP MCP server entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from music_agent.mcp_server.rag_tool import music_rag_search, warm_up_song_store
from music_agent.mcp_server.web_tool import web_search


@asynccontextmanager
async def server_lifespan(_mcp: FastMCP) -> AsyncIterator[dict]:
    """Warm up the singleton song store without crashing MCP startup."""

    yield {"warmup": warm_up_song_store()}


mcp = FastMCP("music-mood-agent", lifespan=server_lifespan)


@mcp.tool(name="music_rag_search")
def music_rag_search_tool(
    query: str,
    mood_terms: list[str] | None = None,
    genres: list[str] | None = None,
    tags: list[str] | None = None,
    artist: str | None = None,
    limit: int = 5,
) -> dict:
    """Search internal song data for mood-oriented recommendations."""

    return music_rag_search(
        {
            "query": query,
            "mood_terms": mood_terms or [],
            "genres": genres or [],
            "tags": tags or [],
            "artist": artist,
            "limit": limit,
        }
    )


@mcp.tool(name="web_search")
def web_search_tool(query: str, search_intent: str, limit: int = 5) -> dict:
    """Search the web for artist deep-dives or RAG fallback context."""

    return web_search(
        {
            "query": query,
            "search_intent": search_intent,
            "limit": limit,
        }
    )


app = mcp.streamable_http_app()
