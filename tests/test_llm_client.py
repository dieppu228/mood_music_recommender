from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from music_agent.config import Settings
from music_agent.llm_client import LlmClient, LlmOutputError


class Decision(BaseModel):
    action: str
    response: str | None = None


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeAsyncOpenAI:
    def __init__(self, content: str) -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_llm_client_initializes_openai_client_from_settings() -> None:
    settings = Settings(
        llm_base_url="http://localhost:9999/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )

    client = LlmClient(settings=settings)

    assert str(client.client.base_url) == "http://localhost:9999/v1/"
    assert client.model == "test-model"


@pytest.mark.asyncio
async def test_llm_client_parses_valid_json_with_code_fence() -> None:
    fake = FakeAsyncOpenAI('```json\n{"action":"respond","response":"ok"}\n```')
    client = LlmClient(client=fake, settings=Settings(llm_model="test-model"))

    result = await client.complete_json("system", "user", Decision)

    assert result == Decision(action="respond", response="ok")
    call = fake.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.0
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


@pytest.mark.asyncio
async def test_llm_client_raises_output_error_for_malformed_json() -> None:
    fake = FakeAsyncOpenAI("not json")
    client = LlmClient(client=fake)

    with pytest.raises(LlmOutputError) as exc:
        await client.complete_json("system", "user", Decision)

    assert exc.value.raw_text == "not json"
    assert "Expecting value" in exc.value.details


@pytest.mark.asyncio
async def test_llm_client_raises_output_error_for_validation_failure() -> None:
    fake = FakeAsyncOpenAI('{"response":"missing action"}')
    client = LlmClient(client=fake)

    with pytest.raises(LlmOutputError) as exc:
        await client.complete_json("system", "user", Decision)

    assert "Field required" in exc.value.details


@pytest.mark.asyncio
async def test_llm_client_complete_text_returns_plain_text() -> None:
    fake = FakeAsyncOpenAI("plain answer")
    client = LlmClient(client=fake, settings=Settings(llm_model="text-model"))

    result = await client.complete_text("system", "user", temperature=0.4)

    assert result == "plain answer"
    call = fake.completions.calls[0]
    assert call["model"] == "text-model"
    assert call["temperature"] == 0.4
    assert "response_format" not in call
