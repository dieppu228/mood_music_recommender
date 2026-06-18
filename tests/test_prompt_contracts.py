from pathlib import Path


PROMPT_ROOT = Path("src/music_agent/prompts")


def read_prompt(relative_path: str) -> str:
    return (PROMPT_ROOT / relative_path).read_text(encoding="utf-8")


def test_system_prompt_documents_both_tools_and_schemas() -> None:
    prompt = read_prompt("system.md")

    assert "music_rag_search" in prompt
    assert "web_search" in prompt
    assert '"query"' in prompt
    assert '"mood_terms"' in prompt
    assert '"genres"' in prompt
    assert '"tags"' in prompt
    assert '"artist"' in prompt
    assert '"search_intent"' in prompt
    assert "artist_deep_dive" in prompt
    assert "fallback_recommendation" in prompt


def test_system_prompt_defines_global_safety_and_smalltalk_rules() -> None:
    prompt = read_prompt("system.md").lower()

    assert "greeting" in prompt
    assert "smalltalk" in prompt
    assert "không bịa bài hát" in prompt
    assert "artist facts" in prompt
    assert "không lộ scratchpad" in prompt
    assert "debug=false" in prompt


def test_think_prompt_allows_only_call_tool_or_respond_and_json_only() -> None:
    prompt = read_prompt("nodes/think.md")

    assert "`call_tool`" in prompt
    assert "`respond`" in prompt
    assert "`action` chỉ được là `call_tool` hoặc `respond`" in prompt
    assert "Chỉ trả về JSON hợp lệ" in prompt
    assert "Không bọc Markdown" in prompt
    assert "tool_name" in prompt
    assert "tool_input" in prompt


def test_act_prompt_is_code_contract_for_planned_tool_only() -> None:
    prompt = read_prompt("nodes/act.md").lower()

    assert "không tự quyết định tool mới" in prompt
    assert "planned_tool" in prompt
    assert "gọi đúng một tool" in prompt
    assert "không retry" in prompt
    assert "tool_result" in prompt
    assert "iteration_count" in prompt


def test_observe_prompt_defines_tool_ok_and_enough_context_rules() -> None:
    prompt = read_prompt("nodes/observe.md")

    assert "Không gọi tool" in prompt
    assert "`enough_context=true` nếu có result và top score đủ tốt" in prompt
    assert "`enough_context=true` nếu có ít nhất một source/snippet hữu ích" in prompt
    assert "`tool_ok=false`" in prompt
    assert "should_fallback_to_web" in prompt


def test_final_prompt_hides_scratchpad_and_disallows_llm_trace() -> None:
    prompt = read_prompt("nodes/final.md").lower()

    assert "không lộ scratchpad" in prompt
    assert "không tự sinh `trace`" in prompt
    assert "trace/debug do code gắn" in prompt
    assert '"status"' in prompt
    assert '"answer"' in prompt
    assert '"recommendations"' in prompt
