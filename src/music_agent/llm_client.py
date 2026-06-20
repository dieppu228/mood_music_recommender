"""Gemini direct async LLM client adapter."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from music_agent.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LlmOutputError(ValueError):
    """Raised when an LLM response cannot be parsed into the expected model."""

    def __init__(self, message: str, raw_text: str, details: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.details = details


class LlmClient:
    """Thin async adapter for Gemini structured and text completions."""

    def __init__(self, client: Any | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or genai.Client(api_key=self.settings.gemini_api_key or "unused")
        self.model = self.settings.llm_model

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.0,
    ) -> T:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        raw_text = extract_message_text(response)
        json_text = strip_json_code_fence(raw_text)
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise LlmOutputError(
                "LLM returned malformed JSON",
                raw_text=raw_text,
                details=str(exc),
            ) from exc

        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            raise LlmOutputError(
                "LLM JSON did not match expected schema",
                raw_text=raw_text,
                details=str(exc),
            ) from exc

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )
        return extract_message_text(response)


def extract_message_text(response: object) -> str:
    """Extract plain text from a Gemini generate_content response."""

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return str(text or "")


def strip_json_code_fence(text: str) -> str:
    """Remove a surrounding Markdown JSON code fence if present."""

    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped
