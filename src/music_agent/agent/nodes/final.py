"""Final node: produce the user-facing response."""

from music_agent.agent.prompts import render_node_prompt, render_system_prompt
from music_agent.agent.state import AgentState, FinalAnswerDraft
from music_agent.llm_client import LlmClient, LlmOutputError
from music_agent.models import AgentStatus, Recommendation


async def final_node(state: AgentState, llm_client: LlmClient | None = None) -> AgentState:
    client = llm_client or LlmClient()

    try:
        draft = await client.complete_json(
            render_system_prompt(state),
            render_node_prompt("final", state),
            FinalAnswerDraft,
            temperature=0.2,
        )
    except LlmOutputError as exc:
        state.errors.append(f"final_llm_output_error: {exc.details}")
        state.status = AgentStatus.FAILED
        state.final_answer = build_safe_failure_answer(state)
        return state

    state.status = draft.status
    state.final_answer = draft.answer
    state.recommendations = merge_recommendation_reasons(state.recommendations, draft.recommendations)
    return state


def merge_recommendation_reasons(
    existing: list[Recommendation],
    drafted: list[Recommendation],
) -> list[Recommendation]:
    if not existing:
        return drafted

    reasons_by_song_id = {item.song_id: item.reason for item in drafted if item.song_id and item.reason}
    reasons_by_title = {
        (item.title.lower(), item.artist.lower()): item.reason
        for item in drafted
        if item.title and item.artist and item.reason
    }
    merged = []
    for item in existing:
        reason = reasons_by_song_id.get(item.song_id) or reasons_by_title.get(
            (item.title.lower(), item.artist.lower())
        )
        if reason:
            item = item.model_copy(update={"reason": reason})
        merged.append(item)
    return merged


def build_safe_failure_answer(state: AgentState) -> str:
    directive = state.scratchpad.get("final_directive")
    if isinstance(directive, str) and directive.strip():
        return directive.strip()
    if state.errors:
        return "Mình chưa có đủ context đáng tin cậy để trả lời yêu cầu này."
    if state.recommendations:
        return "Mình tìm được vài bài phù hợp, nhưng chưa tạo được câu trả lời cuối đáng tin cậy."
    return "Mình chưa tìm được đủ context phù hợp để trả lời chính xác."
