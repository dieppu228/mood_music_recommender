from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from music_agent.config import Settings
from music_agent.llm_client import LlmClient, LlmOutputError


class Decision(BaseModel):
    action: str
    response: str | None = None


class FakeModels:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.content)


class FakeGeminiClient:
    def __init__(self, content: str) -> None:
        self.models = FakeModels(content)
        self.aio = SimpleNamespace(models=self.models)


def test_llm_client_initializes_gemini_client_from_settings() -> None:
    settings = Settings(
        gemini_api_key="test-key",
        llm_model="test-model",
    )

    client = LlmClient(settings=settings)

    assert client.model == "test-model"


@pytest.mark.asyncio
async def test_llm_client_parses_valid_json_with_code_fence() -> None:
    fake = FakeGeminiClient('```json\n{"action":"respond","response":"ok"}\n```')
    client = LlmClient(client=fake, settings=Settings(llm_model="test-model"))

    result = await client.complete_json("system", "user", Decision)

    assert result == Decision(action="respond", response="ok")
    call = fake.models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == "user"
    assert call["config"].system_instruction == "system"
    assert call["config"].temperature == 0.0
    assert call["config"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_llm_client_raises_output_error_for_malformed_json() -> None:
    fake = FakeGeminiClient("not json")
    client = LlmClient(client=fake)

    with pytest.raises(LlmOutputError) as exc:
        await client.complete_json("system", "user", Decision)

    assert exc.value.raw_text == "not json"
    assert "Expecting value" in exc.value.details


@pytest.mark.asyncio
async def test_llm_client_raises_output_error_for_validation_failure() -> None:
    fake = FakeGeminiClient('{"response":"missing action"}')
    client = LlmClient(client=fake)

    with pytest.raises(LlmOutputError) as exc:
        await client.complete_json("system", "user", Decision)

    assert "Field required" in exc.value.details


@pytest.mark.asyncio
async def test_llm_client_complete_text_returns_plain_text() -> None:
    fake = FakeGeminiClient("plain answer")
    client = LlmClient(client=fake, settings=Settings(llm_model="text-model"))

    result = await client.complete_text("system", "user", temperature=0.4)

    assert result == "plain answer"
    call = fake.models.calls[0]
    assert call["model"] == "text-model"
    assert call["contents"] == "user"
    assert call["config"].system_instruction == "system"
    assert call["config"].temperature == 0.4
    assert call["config"].response_mime_type is None
