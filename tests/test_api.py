import pytest
import httpx

from music_agent.api.main import app, get_agent_graph
from music_agent.agent.state import AgentState
from music_agent.models import AgentIntent, AgentStatus, Recommendation, ToolCallTrace


class FakeGraph:
    def __init__(self, output=None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.inputs = []

    async def ainvoke(self, initial_state):
        self.inputs.append(initial_state)
        if self.error is not None:
            raise self.error
        return self.output


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    async with make_test_client() as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_rejects_empty_message() -> None:
    async with make_test_client() as client:
        response = await client.post("/v1/chat", json={"message": ""})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_returns_successful_recommendations_and_no_trace_by_default() -> None:
    graph = FakeGraph(
        output=successful_state(
            recommendations=[
                recommendation("mock-001", "After Rain"),
                recommendation("mock-002", "Night Drive"),
            ]
        ).model_dump(mode="python")
    )
    override_agent_graph(graph)

    async with make_test_client() as client:
        response = await client.post("/v1/chat", json={"message": "goi y nhac buon"})

    assert response.status_code == 200
    payload = response.json()
    assert graph.inputs[0]["user_message"] == "goi y nhac buon"
    assert payload["status"] == "ok"
    assert payload["answer"] == "Nghe thử mấy bài này."
    assert [item["title"] for item in payload["recommendations"]] == ["After Rain", "Night Drive"]
    assert payload["tool_calls"][0]["tool_name"] == "music_rag_search"
    assert payload["trace"] is None


@pytest.mark.asyncio
async def test_chat_debug_true_returns_trace() -> None:
    graph = FakeGraph(output=successful_state().model_dump(mode="python"))
    override_agent_graph(graph)

    async with make_test_client() as client:
        response = await client.post("/v1/chat", json={"message": "goi y nhac", "debug": True})

    assert response.status_code == 200
    trace = response.json()["trace"]
    assert trace["scratchpad"]["enough_context"] is True
    assert trace["intent"] == "music_recommendation"
    assert trace["entities"]["mood_terms"] == ["sad"]
    assert trace["confidence"] == 0.87
    assert trace["iteration_count"] == 1
    assert trace["errors"] == []
    assert trace["tool_calls"][0]["tool_name"] == "music_rag_search"


@pytest.mark.asyncio
async def test_chat_clamps_recommendations_to_max_results() -> None:
    graph = FakeGraph(
        output=successful_state(
            recommendations=[
                recommendation("mock-001", "After Rain"),
                recommendation("mock-002", "Night Drive"),
                recommendation("mock-003", "Soft Static"),
            ]
        ).model_dump(mode="python")
    )
    override_agent_graph(graph)

    async with make_test_client() as client:
        response = await client.post(
            "/v1/chat",
            json={"message": "goi y nhac", "max_results": 2},
        )

    assert response.status_code == 200
    assert [item["song_id"] for item in response.json()["recommendations"]] == [
        "mock-001",
        "mock-002",
    ]


@pytest.mark.asyncio
async def test_chat_maps_graph_failure_to_failed_response_without_stack_trace() -> None:
    graph = FakeGraph(error=RuntimeError("database password leaked in exception"))
    override_agent_graph(graph)

    async with make_test_client() as client:
        response = await client.post("/v1/chat", json={"message": "goi y nhac", "debug": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert "database password" not in payload["answer"]
    assert payload["trace"] == {"errors": ["unexpected_agent_error"]}


def successful_state(recommendations: list[Recommendation] | None = None) -> AgentState:
    return AgentState(
        user_message="goi y nhac buon",
        status=AgentStatus.OK,
        intent=AgentIntent.MUSIC_RECOMMENDATION,
        entities={"mood_terms": ["sad"], "genres": [], "tags": [], "constraints": []},
        scratchpad={"tool_ok": True, "enough_context": True},
        recommendations=recommendations or [recommendation("mock-001", "After Rain")],
        final_answer="Nghe thử mấy bài này.",
        confidence=0.87,
        iteration_count=1,
        tool_calls=[
            ToolCallTrace(
                tool_name="music_rag_search",
                tool_input={"query": "sad songs"},
                reason="Need RAG.",
                confidence=0.8,
                ok=True,
                duration_ms=12.0,
            )
        ],
    )


def recommendation(song_id: str, title: str) -> Recommendation:
    return Recommendation(
        song_id=song_id,
        title=title,
        artist="Local Echo",
        mood=["sad", "healing"],
        genres=["indie pop"],
        tags=["rain"],
        reason="Hợp mood.",
        score=0.9,
    )


def make_test_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def override_agent_graph(graph: FakeGraph) -> None:
    async def override() -> FakeGraph:
        return graph

    app.dependency_overrides[get_agent_graph] = override
