"""Deterministic mood-target policy applied to the first think decision."""

from __future__ import annotations

import re
import unicodedata

from music_agent.agent.state import ThinkDecision
from music_agent.models import AgentIntent, ToolName

_MOOD_ALIASES = {
    "happy": {"happy", "vui", "tuoi"},
    "sad": {"sad", "buon"},
    "calm": {"calm", "chill", "thu gian"},
    "energetic": {"energetic", "energy", "nang luong", "soi dong"},
    "romantic": {"romantic", "lang man", "tinh yeu"},
    "stressed": {"stressed", "stress", "cang thang"},
}
_INSULT_WORDS = {"ngu", "idiot", "stupid", "dumb"}
_MUSIC_WORDS = {"nhac", "bai", "music", "song", "songs"}


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def apply_mood_policy(user_message: str, decision: ThinkDecision) -> ThinkDecision:
    normalized = normalize_text(user_message)
    explicit_targets = extract_explicit_music_moods(normalized)
    insult = contains_insult(normalized)
    inferred_moods = infer_current_moods(normalized, insult=insult)
    current_moods = inferred_moods or (
        decision.entities.mood_terms if explicit_targets else []
    )

    if insult:
        return build_rag_decision(
            decision,
            current_moods=current_moods or ["angry", "frustrated"],
            target_moods=["calm"],
            query="calm soothing relaxing music",
            tags=["soothing", "relaxing"],
            requires_apology=True,
        )

    if explicit_targets:
        return build_rag_decision(
            decision,
            current_moods=current_moods,
            target_moods=explicit_targets,
            query=decision.tool_input.get("query", "") if decision.tool_input else "",
            tags=decision.entities.tags,
            requires_apology=decision.entities.requires_apology,
        )

    regulation = regulation_target(current_moods)
    if regulation is None:
        return synchronize_target_moods(decision)

    target_moods, query, tags = regulation
    return build_rag_decision(
        decision,
        current_moods=current_moods,
        target_moods=target_moods,
        query=query,
        tags=tags,
        requires_apology=decision.entities.requires_apology,
    )


def extract_explicit_music_moods(normalized: str) -> list[str]:
    tokens = normalized.split()
    targets = []
    for mood, aliases in _MOOD_ALIASES.items():
        positions = [
            position
            for alias in aliases
            for position in phrase_positions(tokens, alias.split())
        ]
        if any(
            abs(mood_index - music_index) <= 2
            for mood_index in positions
            for music_index, token in enumerate(tokens)
            if token in _MUSIC_WORDS
        ):
            targets.append(mood)
    return targets


def phrase_positions(tokens: list[str], phrase: list[str]) -> list[int]:
    width = len(phrase)
    return [
        index
        for index in range(len(tokens) - width + 1)
        if tokens[index : index + width] == phrase
    ]


def contains_insult(normalized: str) -> bool:
    return bool(set(normalized.split()) & _INSULT_WORDS)


def infer_current_moods(normalized: str, *, insult: bool) -> list[str]:
    if insult:
        return ["angry", "frustrated"]
    if contains_any_term(normalized, ("buon", "sad", "down")):
        return ["sad"]
    if contains_any_term(normalized, ("stress", "cang thang", "lo lang", "anxious")):
        return ["stressed"]
    if contains_any_term(normalized, ("tuc", "gian", "angry", "frustrated")):
        return ["angry", "frustrated"]
    if contains_any_term(normalized, ("met", "tired", "low energy")):
        return ["tired"]
    return []


def contains_any_term(normalized: str, terms: tuple[str, ...]) -> bool:
    padded = f" {normalized} "
    return any(f" {term} " in padded for term in terms)


def regulation_target(current_moods: list[str]):
    normalized = {normalize_text(mood) for mood in current_moods}
    if normalized & {"sad", "buon"}:
        return ["calm", "happy"], "calm happy healing uplifting music", ["healing", "uplifting"]
    if normalized & {"stressed", "stress", "anxious", "lo lang", "cang thang"}:
        return ["calm"], "calm grounding relaxing music", ["grounding", "relaxing"]
    if normalized & {"angry", "frustrated", "tuc", "gian"}:
        return ["calm"], "calm soothing relaxing music", ["soothing", "relaxing"]
    if normalized & {"tired", "met", "low energy"}:
        return ["energetic"], "energetic motivating music", ["motivating"]
    return None


def build_rag_decision(
    decision: ThinkDecision,
    *,
    current_moods: list[str],
    target_moods: list[str],
    query: str,
    tags: list[str],
    requires_apology: bool,
) -> ThinkDecision:
    existing_input = decision.tool_input or {}
    entities = type(decision.entities).model_validate(
        {
            **decision.entities.model_dump(),
            "mood_terms": current_moods,
            "target_mood_terms": target_moods,
            "requires_apology": requires_apology,
            "tags": tags or decision.entities.tags,
        }
    )
    tool_input = {
        "query": query or " ".join([*target_moods, "music"]),
        "mood_terms": target_moods,
        "genres": existing_input.get("genres", decision.entities.genres),
        "tags": tags or existing_input.get("tags", decision.entities.tags),
        "artist": existing_input.get("artist", decision.entities.artist),
        "limit": existing_input.get("limit", 5),
    }
    return ThinkDecision.model_validate(
        {
            **decision.model_dump(),
            "action": "call_tool",
            "intent": AgentIntent.MUSIC_RECOMMENDATION,
            "entities": entities.model_dump(),
            "tool_name": ToolName.MUSIC_RAG_SEARCH,
            "tool_input": tool_input,
            "response": None,
        }
    )


def synchronize_target_moods(decision: ThinkDecision) -> ThinkDecision:
    if decision.tool_name != ToolName.MUSIC_RAG_SEARCH or not decision.entities.target_mood_terms:
        return decision
    tool_input = dict(decision.tool_input or {})
    tool_input["mood_terms"] = [mood.value for mood in decision.entities.target_mood_terms]
    return ThinkDecision.model_validate({**decision.model_dump(), "tool_input": tool_input})
