"""Observe node: normalize tool output into agent state."""

from typing import Any

from music_agent.agent.state import AgentState
from music_agent.models import Recommendation, ToolName

MIN_RAG_TOP_SCORE = 0.5


async def observe_node(state: AgentState) -> AgentState:
    wrapper = state.tool_result or {}
    tool_name = wrapper.get("tool_name")
    payload = wrapper.get("result") if isinstance(wrapper.get("result"), dict) else {}
    error = wrapper.get("error")

    if not wrapper.get("ok"):
        state.scratchpad.update(
            {
                "tool_ok": False,
                "enough_context": False,
                "summary": "Tool execution failed.",
                "evidence": [],
                "last_tool": tool_name,
                "should_fallback_to_web": tool_name == ToolName.MUSIC_RAG_SEARCH,
            }
        )
        if error:
            state.errors.append(format_tool_error(error))
        return state

    if tool_name == ToolName.MUSIC_RAG_SEARCH:
        observe_rag_result(state, payload)
    elif tool_name == ToolName.WEB_SEARCH:
        observe_web_result(state, payload)
    else:
        state.scratchpad.update(
            {
                "tool_ok": False,
                "enough_context": False,
                "summary": f"Unknown tool result: {tool_name}",
                "evidence": [],
                "last_tool": tool_name,
                "should_fallback_to_web": False,
            }
        )
        state.errors.append(f"unknown_tool_result: {tool_name}")
    return state


def observe_rag_result(state: AgentState, payload: dict[str, Any]) -> None:
    raw_results = payload.get("results") or []
    result_count = int(payload.get("result_count") or len(raw_results))
    top_score = max((float(item.get("score") or 0.0) for item in raw_results), default=0.0)
    enough_context = result_count > 0 and top_score >= MIN_RAG_TOP_SCORE
    recommendations = [recommendation_from_rag_item(item) for item in raw_results]

    state.recommendations = recommendations
    state.scratchpad.update(
        {
            "tool_ok": bool(payload.get("ok", True)),
            "enough_context": enough_context,
            "summary": build_rag_summary(result_count, top_score, enough_context),
            "evidence": raw_results,
            "last_tool": ToolName.MUSIC_RAG_SEARCH,
            "should_fallback_to_web": not enough_context,
        }
    )


def observe_web_result(state: AgentState, payload: dict[str, Any]) -> None:
    raw_results = payload.get("results") or []
    evidence = [item for item in raw_results if item.get("url") or item.get("content")]
    sources = payload.get("sources") or [
        item.get("url") for item in evidence if isinstance(item.get("url"), str)
    ]
    catalog_matches = payload.get("catalog_matches") or []
    enough_context = bool(evidence or sources)

    # Surface catalog songs mentioned in the web text as structured, playable recommendations.
    if catalog_matches:
        state.recommendations = [recommendation_from_rag_item(item) for item in catalog_matches]

    state.scratchpad.update(
        {
            "tool_ok": bool(payload.get("ok", True)),
            "enough_context": enough_context,
            "summary": (
                f"Web search returned {len(evidence)} useful result(s); "
                f"{len(catalog_matches)} catalog match(es)."
                if enough_context
                else "Web search returned no useful result."
            ),
            "evidence": evidence,
            "sources": sources,
            "catalog_match_count": len(catalog_matches),
            "last_tool": ToolName.WEB_SEARCH,
            "should_fallback_to_web": False,
        }
    )


def recommendation_from_rag_item(item: dict[str, Any]) -> Recommendation:
    return Recommendation(
        song_id=str(item.get("song_id") or ""),
        title=str(item.get("title") or ""),
        artist=str(item.get("artist") or ""),
        mood=list(item.get("mood") or []),
        genres=list(item.get("genres") or []),
        tags=list(item.get("tags") or []),
        reason="",
        preview_url=item.get("preview_url"),
        spotify_url=item.get("spotify_url"),
        score=item.get("score"),
    )


def build_rag_summary(result_count: int, top_score: float, enough_context: bool) -> str:
    if enough_context:
        return f"RAG returned {result_count} result(s); top score {top_score:.3f}."
    if result_count > 0:
        return f"RAG returned low-confidence result(s); top score {top_score:.3f}."
    return "RAG returned no result."


def format_tool_error(error: Any) -> str:
    if isinstance(error, dict):
        code = error.get("error_code") or "tool_error"
        message = error.get("error") or error
        return f"{code}: {message}"
    return str(error)
