# Music Mood Agent V1 - Kế hoạch triển khai theo phase

## 0. Tổng quan kiến trúc

Xây dựng một Python API service cho agent gợi ý bài hát từ query hoặc mood của user.

V1 đi theo hướng mock-first:
- Agent, API và MCP contract chạy thật.
- RAG tool dùng fixture JSONL và Gemini embedding API.
- Chưa ingest 10k bài hát vào Qdrant trong V1, nhưng retrieval interface phải được thiết kế để sau này thay bằng Qdrant mà không cần đổi agent.

Kiến trúc runtime:
- API service: FastAPI, expose `POST /v1/chat`.
- Agent loop: LangGraph, explicit nodes `think -> act -> observe -> think/final`.
- Tool layer: HTTP MCP server, expose 2 tools `music_rag_search` và `web_search`.
- LLM: OpenAI-compatible custom runtime qua `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
- Embedding cho mock retrieval: Gemini `gemini-embedding-001`.
- Web search: Tavily.

Tài liệu tham chiếu:
- MCP Streamable HTTP transport: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
- LangGraph: https://github.com/langchain-ai/langgraph
- Gemini Embeddings: https://ai.google.dev/gemini-api/docs/embeddings

### 0.1 Lý do chọn Gemini embedding

RAG của hệ thống không chỉ so khớp một đoạn văn bản dài như `lyrics_summary`. Một bài hát
được retrieve từ nhiều nhóm tín hiệu:
- Semantics dài: `lyrics_summary`, `metadata_summary`, title/artist/album context.
- Keyword/metadata ngắn: `mood`, `tags`, `genres`, `artist`, source category.
- Query tự nhiên của user: thường là mood/vibe/situation, không luôn trùng literal keyword.

Vì vậy V1 dùng `gemini-embedding-001` thay vì local embedding model nhẹ. Lý do chính là
model này hỗ trợ `task_type`, cho phép encode khác nhau theo vai trò retrieval:
- `RETRIEVAL_DOCUMENT`: dùng khi embed song document/chunk cần được truy xuất.
- `RETRIEVAL_QUERY`: dùng khi embed user query hoặc query đã rewrite.

Thiết kế quan trọng: không embed `mood/tags/genres` riêng bằng `RETRIEVAL_QUERY` để lưu như
document vector. `RETRIEVAL_QUERY` dành cho phía query. Các field keyword như `mood`,
`tags`, `genres` sẽ:
- được đưa vào `document_text` của chunk và embed bằng `RETRIEVAL_DOCUMENT`;
- được lưu nguyên trong payload để filter/boost/rerank bằng exact match.

Nếu sau này cần multi-vector nâng cao, có thể thêm vector phụ:
- `song_document_vector`: full song text, task type `RETRIEVAL_DOCUMENT`.
- `facet_document_vector`: text tổng hợp từ mood/tags/genres/source, vẫn dùng
  `RETRIEVAL_DOCUMENT` vì đây là tài liệu/facet được retrieve.
- `query_vector`: user query, task type `RETRIEVAL_QUERY`, không lưu trong Qdrant payload.

## 0.2 Quyết định đã chốt cho V1

Các quyết định này override mọi mô tả mâu thuẫn ở phần dưới:

1. **Prompt theo node, đủ 4 node.** Giữ `system.md` + 4 node prompt (`think/act/observe/final`).
   Mỗi node có một file mô tả nhiệm vụ + việc tiếp theo. Tool guide/schema nằm trong `system.md`,
   không tách thành file prompt tool riêng.
2. **observe là code thuần, KHÔNG gọi LLM.** observe chỉ đóng gói tool result, đếm kết quả và so
   threshold score để set `enough_context`/`tool_ok`, rồi trả về `think`. Mọi suy nghĩ nằm ở `think`.
3. **V1 dùng mock fixture retrieval.** `FixtureSongStore` + JSONL (~30 bài) embed bằng Gemini API
   (xem 0.1). Qdrant + 10k bài để Phase 9.
4. **Bỏ reranker khỏi V1.** Chỉ cosine + deterministic boost. Gỡ `RERANKER_MODEL` khỏi `config.py`
   và `.env.example`. Thêm lại rerank stage khi lên Qdrant thật.
5. **Async end-to-end.** FastAPI handler `async`, LangGraph `ainvoke`, node async, `AsyncOpenAI`,
   MCP `ClientSession` async. Không trộn sync/async. (`fixture_store` Phase 2 vẫn sync — chỉ chạy
   trong tiến trình MCP server, không nằm trên đường async của agent.)

## 0.3 Vai trò prompt của từng node (quan trọng)

Prompt của một node có thể đóng 1 trong 2 vai:
- **LLM prompt** = được gửi thật vào model; node đó "suy nghĩ" bằng LLM.
- **Code contract** = chỉ là bản mô tả/hợp đồng cho code làm theo; node KHÔNG gọi LLM.

| Node | Gọi LLM? | Vai trò prompt |
|------|----------|----------------|
| `think` | Có | LLM prompt — phân tích query, extract entities, chọn tool hoặc finalize |
| `act` | Không | Code contract — cầm `planned_tool` của think rồi gọi MCP, không suy nghĩ |
| `observe` | Không | Code contract — đóng gói result + set `enough_context`/`tool_ok` |
| `final` | Có | LLM prompt — viết câu trả lời tự nhiên cho user |

→ Chỉ `think.md` và `final.md` là prompt bắn vào LLM. `act.md` và `observe.md` là contract cho code.

## 1. Public contracts

### 1.1 API contract

Endpoint:
- `POST /v1/chat`

Request:
```json
{
  "message": "gợi ý bài nghe lúc buồn nhưng vẫn muốn healing",
  "session_id": "optional-string",
  "max_results": 5,
  "debug": false
}
```

Response:
```json
{
  "status": "ok | failed | out_of_domain",
  "answer": "string, trả lời cùng ngôn ngữ với user",
  "recommendations": [
    {
      "song_id": "string",
      "title": "string",
      "artist": "string",
      "mood": ["string"],
      "genres": ["string"],
      "tags": ["string"],
      "reason": "string",
      "preview_url": "string|null",
      "score": 0.87
    }
  ],
  "tool_calls": [],
  "trace": null
}
```

### 1.2 MCP tool contracts

Tool `music_rag_search`:
```json
{
  "query": "string",
  "mood_terms": ["string"],
  "genres": ["string"],
  "tags": ["string"],
  "artist": "string|null",
  "limit": 5
}
```

Tool `web_search`:
```json
{
  "query": "string",
  "search_intent": "artist_deep_dive | fallback_recommendation",
  "limit": 5
}
```

### 1.3 Agent state contract

Các field trong state:
- `user_message`: input gốc của user.
- `language`: ngôn ngữ agent sẽ dùng để trả lời.
- `intent`: `smalltalk | music_recommendation | artist_deep_dive | out_of_domain`.
- `entities`: mood, genres, tags, artist, song title và các constraint đã extract.
- `planned_tool`: tool được chọn, input, lý do chọn tool, confidence.
- `tool_result`: wrapper từ `McpToolClient` (`{ok, tool_name, duration_ms, result, error}`); payload tool ở `result`.
- `scratchpad`: evidence đã được chuẩn hóa từ node `observe`.
- `recommendations`: danh sách bài hát cuối cùng.
- `final_answer`: câu trả lời tự nhiên cuối cùng.
- `confidence`: confidence hiện tại của agent.
- `iteration_count`: số lần gọi tool.
- `errors`: các lỗi recoverable.

Luật chạy loop:
- V1 gọi tối đa `2` tool calls.
- Route fallback hợp lệ: `music_rag_search -> web_search -> final`.
- V1 không retry lại cùng một tool.
- Greeting và smalltalk bypass tool, đi thẳng tới `final`.

## Phase 1 - Project scaffold, config và shared models

### Mục tiêu

Tạo cấu trúc Python project nền tảng, dependency config, environment settings và các Pydantic models dùng chung cho API, agent và MCP tools.

### Module cần làm
- Project packaging và dependency management.
- Load environment/config.
- Shared request/response/domain models.
- Test wiring cơ bản.

### File cần tạo
- `pyproject.toml`
- `.env.example`
- `README.md`
- `src/music_agent/__init__.py`
- `src/music_agent/config.py`
- `src/music_agent/models.py`
- `tests/conftest.py`
- `tests/test_config.py`
- `tests/test_models.py`

### Chi tiết triển khai

`pyproject.toml`:
- Runtime deps:
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `pydantic-settings`
  - `langgraph`
  - `openai`
  - `mcp`
  - `httpx`
  - `google-genai`
  - `numpy`
  - `tavily-python`
  - `python-dotenv`
- Dev deps:
  - `pytest`
  - `pytest-asyncio`
  - `respx`
  - `ruff`

`src/music_agent/config.py`:
- Define `Settings` gồm:
  - `llm_base_url`
  - `llm_api_key`
  - `llm_model`
  - `gemini_api_key`
  - `embedding_provider`
  - `embedding_model`
  - `embedding_output_dimensionality`
  - `embedding_document_task_type`
  - `embedding_query_task_type`
  - `tavily_api_key`
  - `mcp_server_url`
  - `mcp_host`
  - `mcp_port`
  - `api_host`
  - `api_port`
  - `mock_song_path`
- Default:
  - `LLM_BASE_URL=http://localhost:4000/v1`
  - `LLM_MODEL=gemini-2.5-flash-lite`
  - `LLM_API_KEY` để trong env
  - `GEMINI_API_KEY` để trong env
  - `EMBEDDING_PROVIDER=gemini`
  - `EMBEDDING_MODEL=gemini-embedding-001`
  - `EMBEDDING_OUTPUT_DIMENSIONALITY=768`
  - `EMBEDDING_DOCUMENT_TASK_TYPE=RETRIEVAL_DOCUMENT`
  - `EMBEDDING_QUERY_TASK_TYPE=RETRIEVAL_QUERY`
  - `MCP_HOST=localhost`
  - `MCP_PORT=8001`
  - `MCP_SERVER_URL=http://localhost:8001/mcp`
  - `API_HOST=localhost`
  - `API_PORT=8000`
  - `MOCK_SONG_PATH=data/mock_songs.jsonl`

`src/music_agent/models.py`:
- Define Pydantic models:
  - `ChatRequest`
  - `ChatResponse`
  - `Recommendation`
  - `ToolCallTrace`
  - `ExtractedEntities`
  - `PlannedToolCall`
  - `MusicRagSearchInput`
  - `MusicRagSearchResult`
  - `WebSearchInput`
  - `WebSearchResult`
  - `AgentStatus`
  - `AgentIntent`

### Test checklist
- [ ] `Settings` load default khi không có env vars.
- [ ] `Settings` nhận env override đúng.
- [ ] `ChatRequest` reject `message` rỗng.
- [ ] `ChatRequest` default `max_results=5` và `debug=false`.
- [ ] `Recommendation` chấp nhận `preview_url` optional.
- [ ] `MusicRagSearchInput` validate hoặc clamp `limit` vào safe range.
- [ ] `ChatResponse` serialize được sang JSON với `trace=null`.

### Done criteria
- `pytest tests/test_config.py tests/test_models.py -q` pass.
- `ruff check .` pass.

## Phase 2 - Mock song data và fixture retrieval module

### Mục tiêu

Xây dựng local semantic retrieval module có behavior giống future Qdrant retrieval, nhưng V1 đọc từ fixture JSONL.

### Module cần làm
- Spotify crawl dataset đã normalize.
- Load và validate song records.
- Tạo search document.
- Local vector cache/index cho embeddings sinh bởi Gemini API.
- Cosine similarity ranking.

### File cần tạo
- `data/spotify_hybrid_tracks_raw.csv`
- `data/spotify_hybrid_processed.csv`
- `data/spotify_processing_summary.json`
- `data/spotify_songs.jsonl`
- `data/README.md`
- `data/mock_songs.jsonl`
- `src/music_agent/retrieval/__init__.py`
- `src/music_agent/retrieval/fixture_store.py`
- `tests/test_fixture_store.py`

### Chi tiết triển khai

`data/spotify_songs.jsonl`:
- Normalized từ partner Spotify crawl trong `spocrawl/`.
- Mỗi dòng là một JSON object.
- Field bắt buộc (khớp `SongPayload`):
  - `chunk_id`
  - `song_id`
  - `title`
  - `artist`
  - `metadata_summary`
- Field optional:
  - `lyrics_summary`
  - `mood`
  - `genres`
  - `tags`
  - `preview_url`
  - `spotify_url`
  - `album`
  - `release_year`
  - `source_name` / `search_query` ...
- Lưu ý: crawl hiện chưa có lyric summary thật. `metadata_summary` (required) là summary tạm thời
  từ title, artist, album, mood, tags và source category; `lyrics_summary` (optional) để trống cho
  tới khi có lyric thật.

`data/mock_songs.jsonl`:
- Giữ lại fixture nhỏ để unit test nhanh khi không muốn load toàn bộ Spotify dataset.

Document text format để embed bằng `RETRIEVAL_DOCUMENT`:
```text
title: <title>
artist: <artist>
album: <album>
metadata_summary: <metadata_summary>
lyrics_summary: <lyrics_summary>
mood: <mood joined>
genres: <genres joined>
tags: <tags joined>
source: <source_name>
search_query: <search_query>
```

Query text format để embed bằng `RETRIEVAL_QUERY`:
```text
query: <rewritten user query>
mood_terms: <extracted mood terms>
genres: <extracted genres>
tags: <extracted tags>
artist: <artist if any>
```

`fixture_store.py`:
- Define `SongRecord`.
- Define `FixtureSongStore`.
- Trách nhiệm:
  - Load JSONL.
  - Validate records.
  - Build document embeddings lazy ở lần search đầu tiên bằng Gemini task type `RETRIEVAL_DOCUMENT`.
  - Embed query bằng Gemini task type `RETRIEVAL_QUERY`.
  - Search bằng query vector và extracted metadata.
  - Return normalized `MusicRagSearchResult`.
- Ranking:
  - Main score: cosine similarity.
  - Boost nhỏ, deterministic cho exact mood/genre/tag/artist match.
  - Boost keyword chỉ là bổ trợ; không thay thế semantic score.
  - Sort theo final score giảm dần.

### Test checklist
- [ ] Load được valid JSONL records.
- [ ] Raise lỗi rõ ràng khi JSONL malformed.
- [ ] Build searchable text từ `metadata_summary`/`lyrics_summary`, `mood`, `genres`, `tags`.
- [ ] Return top N theo `limit`.
- [ ] Với query sad/healing, bài sad/healing rank cao hơn bài energetic không liên quan.
- [ ] Apply artist filter/boost khi có `artist`.
- [ ] Handle `preview_url` optional hoặc rỗng.
- [ ] Return empty result list khi không có records.

### Done criteria
- `pytest tests/test_fixture_store.py -q` pass.
- Retrieval module không import API, MCP server hoặc LangGraph modules.

## Phase 3 - MCP tool server

### Mục tiêu

Expose `music_rag_search` và `web_search` thành MCP tools qua HTTP để agent chỉ gọi tool thông qua MCP contracts.

### Quyết định Phase 3 (chốt trước khi code)

1. **MCP thật, tối giản.** Server dùng `mcp` SDK (`FastMCP`) với streamable-http transport,
   KHÔNG tự định nghĩa route POST. App = `mcp.streamable_http_app()` (tự mount ở `/mcp`).
   `McpToolClient` (Phase 4) phải nói đúng MCP protocol (initialize + `tools/call`), không
   `httpx.post` tay. Mục đích: agent gọi tool sạch + dễ scale, không phát minh protocol riêng.
2. **`FixtureSongStore` là singleton.** Tạo 1 lần ở module-level/lifespan và warm-up index lúc
   startup. KHÔNG tạo store mới mỗi request (tránh re-load JSONL + re-embed Gemini mỗi lần gọi).
3. **Score join ở `rag_tool`.** `fixture_store` giữ nguyên (trả `SongPayload` + score trong
   `diagnostics`). `rag_tool` join score vào từng result để trả item có `song_id/title/artist/score`.
4. **Tool không crash.** `rag_tool` catch `FixtureStoreError` (gồm thiếu `GEMINI_API_KEY`) →
   `ok=false` + diagnostic. `web_tool` đã xử lý thiếu `TAVILY_API_KEY` tương tự.
5. **Test offline.** RAG test inject fake embedder / mock store; web test mock `TavilyClient`.
   Không test nào chạm network. Không in API key vào `diagnostics`.

### Module cần làm
- MCP server app.
- RAG tool wrapper.
- Tavily web search wrapper.
- Tool schemas và normalization.

### File cần tạo
- `src/music_agent/mcp_server/__init__.py`
- `src/music_agent/mcp_server/server.py`
- `src/music_agent/mcp_server/rag_tool.py`
- `src/music_agent/mcp_server/web_tool.py`
- `tests/test_mcp_rag_tool.py`
- `tests/test_mcp_web_tool.py`

### Chi tiết triển khai

`server.py`:
- Tạo `FastMCP` app, register 2 tool `music_rag_search` và `web_search`.
- Expose `app = mcp.streamable_http_app()` để `uvicorn server:app` mount đúng `/mcp`.
- Khởi tạo `FixtureSongStore` singleton + warm-up index ở startup/lifespan.
- Giữ transport path compatible với `MCP_SERVER_URL=http://localhost:8001/mcp`.

`rag_tool.py`:
- Nhận `MusicRagSearchInput`.
- Call `FixtureSongStore.search` (dùng store singleton).
- Join score từ `diagnostics.score_details` vào từng result.
- Catch `FixtureStoreError` → `ok=false` + diagnostic, không raise.
- Return:
  - `ok`
  - `results` (mỗi item có `song_id`, `title`, `artist`, `score`)
  - `result_count`
  - `diagnostics`

`web_tool.py`:
- Nhận `WebSearchInput`.
- Call Tavily client.
- Return:
  - `ok`
  - `results`
  - `sources`
  - `diagnostics`
- Nếu thiếu `TAVILY_API_KEY`:
  - Return `ok=false`.
  - Return error code rõ ràng `missing_tavily_api_key`.
  - Không crash server.

### Test checklist
- [ ] RAG tool validate input bắt buộc.
- [ ] RAG tool return normalized results có `song_id`, `title`, `artist`, `score`.
- [ ] RAG tool return `ok=true` khi fixture search thành công.
- [ ] RAG tool return `ok=false` kèm diagnostic error nếu fixture loading fail.
- [ ] Web tool return `ok=false` khi thiếu Tavily key.
- [ ] Web tool map mocked Tavily response thành normalized sources.
- [ ] MCP server register đúng 2 tools: `music_rag_search` và `web_search`.

### Done criteria
- `pytest tests/test_mcp_rag_tool.py tests/test_mcp_web_tool.py -q` pass.
- Manual server boot hoạt động:
```bash
uvicorn music_agent.mcp_server.server:app --host localhost --port 8001
```

## Phase 4 - MCP client và LLM client

### Mục tiêu

Tạo adapter ổn định để agent gọi MCP tools và OpenAI-compatible LLM runtime mà không leak provider-specific logic vào graph nodes.

### Module cần làm
- MCP client wrapper.
- LLM client wrapper.
- JSON parsing và validation cho LLM outputs.

### File cần tạo
- `src/music_agent/tools/__init__.py`
- `src/music_agent/tools/mcp_client.py`
- `src/music_agent/llm_client.py`
- `tests/test_mcp_client.py`
- `tests/test_llm_client.py`

### Chi tiết triển khai

`mcp_client.py` (async — xem 0.2):
- Define `McpToolClient(server_url: str | None = None)`.
- Method:
  - `async def call_tool(tool_name: str, tool_input: dict) -> dict`
- Transport: dùng `mcp.client.streamable_http.streamablehttp_client(server_url)` + `ClientSession`.
  Mỗi call mở session → `await session.initialize()` → `await session.call_tool(...)` → đóng
  (per-call connect chấp nhận được vì V1 tối đa 2 tool call).
- Gỡ kết quả (quyết định B):
  - Ưu tiên `result.structuredContent` (tool của ta return `dict` → nằm ở đây).
  - Fallback parse `result.content[0].text` thành JSON.
  - `result.isError == True` → coi như tool error.
- Return wrapper (quyết định C):
  ```json
  {
    "ok": true,
    "tool_name": "music_rag_search",
    "duration_ms": 12.3,
    "result": { "...": "payload dict của tool" },
    "error": null
  }
  ```
  - Transport failure (connect refused/timeout) → `ok=false`, `result=null`,
    `error={"error_code": "mcp_transport_error", "error": "..."}`, vẫn đo `duration_ms`.
  - Tool báo lỗi (`isError`) → `ok=false`, `error={"error_code": "mcp_tool_error", ...}`.

`llm_client.py` (async — xem 0.2):
- Define `LlmClient(client: AsyncOpenAI | None = None, settings: Settings | None = None)`.
  Cho phép inject client để test offline (giống `embedding_client` ở fixture_store).
- Dùng `AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)`, model
  `settings.llm_model`.
- Methods:
  - `async def complete_json(system_prompt, user_prompt, response_model: type[T], temperature=0.0) -> T`
  - `async def complete_text(system_prompt, user_prompt, temperature=0.7) -> str`
- `complete_json` (quyết định D — parse phòng thủ):
  - Best-effort `response_format={"type": "json_object"}`.
  - Strip code fence ```json ... ``` trước khi `json.loads`.
  - Validate bằng Pydantic `response_model`.
  - JSON/validation fail → raise `LlmOutputError` (kèm raw text + chi tiết) để node bắt và
    finalize fail. V1 KHÔNG retry reformat.

### Test checklist
- [ ] MCP client gọi đúng tool name + input và trả `result` payload đúng (test bằng in-memory
      client/server pair của SDK với 2 tool thật, KHÔNG mock — xem 0.2/quyết định E).
- [ ] MCP client trả `ok=false` + `error.error_code` khi transport failure.
- [ ] MCP client trả `duration_ms` trong wrapper.
- [ ] MCP client map `isError` của tool → `ok=false`, `error_code="mcp_tool_error"`.
- [ ] LLM client pass `base_url`, `api_key`, `model` từ settings (inject fake AsyncOpenAI).
- [ ] LLM client parse valid JSON (kể cả khi bọc trong code fence) thành Pydantic model.
- [ ] LLM client raise `LlmOutputError` khi JSON malformed / validation fail.
- [ ] LLM client `complete_text` trả plain text.

### Done criteria
- `pytest tests/test_mcp_client.py tests/test_llm_client.py -q` pass.
- Agent graph code không import Tavily, fixture store hoặc MCP server internals trực tiếp.
- Mọi I/O ở client đều async; không có `asyncio.run` bên trong client.

## Phase 5 - System prompt, node prompts và agent state

### Mục tiêu

Định nghĩa prompt architecture mới: tool guide/schema/instruction nằm trong system prompt,
còn mỗi node trong graph có prompt riêng để xác định nhiệm vụ và output contract của node đó.

### Quyết định Phase 5 (chốt trước khi code)

1. **State tái dùng contract `models.py`.** `AgentState` không định nghĩa lại entity/tool;
   nó nhúng `ExtractedEntities` và `PlannedToolCall` từ `models.py`. `tool_ok`/`enough_context`
   KHÔNG phải field top-level của `AgentState` — chúng là key trong `scratchpad` dict (xem 1.3,
   state chỉ có `scratchpad/recommendations/errors`).
2. **Chỉ `think`/`final` có LLM-output model.** `ThinkDecision` (parse output `think.md`) và
   `FinalAnswerDraft` (parse output `final.md`) là model để validate JSON LLM trả về.
   `Observation` KHÔNG phải LLM schema — observe là code thuần (0.2), ghi thẳng vào `scratchpad`.
   `Observation` chỉ là typed helper nội bộ tuỳ chọn, không bắt buộc.
3. **`ThinkDecision` khớp đúng `think.md`.** Field: `thought`, `action` (`call_tool|respond`),
   `intent` (`AgentIntent`), `entities` (`ExtractedEntities`), `tool_name` (`ToolName | None`),
   `tool_input` (`dict | None`), `confidence` (0..1), `response` (`str | None`). Không có
   `should_finalize`; "finalize" = `action == "respond"`.

### Module cần làm
- System prompt chứa agent context, runtime context, tool guide và ràng buộc toàn hệ thống.
- Prompt riêng cho node `think`.
- Prompt riêng cho node `act`.
- Prompt riêng cho node `observe`.
- Prompt riêng cho node `final`.
- Agent state và LLM-output models cho `think`/`final` (observe là code thuần, không có schema LLM).

### File cần tạo
- `src/music_agent/prompts/system.md`
- `src/music_agent/prompts/nodes/think.md`
- `src/music_agent/prompts/nodes/act.md`
- `src/music_agent/prompts/nodes/observe.md`
- `src/music_agent/prompts/nodes/final.md`
- `src/music_agent/agent/__init__.py`
- `src/music_agent/agent/state.py`
- `tests/test_agent_state.py`
- `tests/test_prompt_contracts.py`

### Chi tiết triển khai

`system.md`:
- Chứa agent context placeholders: `agent_name`, `agent_description`, `agent_system_prompt`.
- Chứa runtime placeholders: `history`, `scratchpad`, `user_message`.
- Chứa toàn bộ tool guide cho `music_rag_search` và `web_search`.
- Với mỗi tool, define mục đích, khi nên dùng, khi không nên dùng, input schema và output kỳ vọng.
- Define rule toàn hệ thống: match ngôn ngữ user, không hallucinate, đọc history khi có tham chiếu ngược, không lộ scratchpad khi `debug=false`.

`prompts/nodes/think.md`:
- Define nhiệm vụ classify intent, extract entities, chọn `call_tool` hoặc `respond`.
- Output bắt buộc là JSON hợp lệ (không bọc Markdown), gồm các field:
  `thought`, `action`, `intent`, `entities`, `tool_name`, `tool_input`, `confidence`, `response`.
- `action` chỉ được là `call_tool` hoặc `respond`.
- Khi `call_tool`, `tool_name` phải nằm trong tool guide của system prompt; `tool_input` khớp
  schema tool (vd `music_rag_search` → `query/mood_terms/genres/tags/artist/limit`).
- Khi `respond`, `tool_name` và `tool_input` phải là null; `response` chứa chỉ dẫn/câu trả lời
  cho node `final`.
- `entities` khớp `ExtractedEntities`; `intent` khớp `AgentIntent`.

`prompts/nodes/act.md` (CODE CONTRACT — không bắn vào LLM, xem 0.3):
- Mô tả nhiệm vụ cho code: execute đúng một `planned_tool`.
- Không tự chọn tool mới.
- Không retry cùng tool trong V1.
- Output cập nhật `tool_result`, `tool_calls`, `iteration_count`.

`prompts/nodes/observe.md` (CODE CONTRACT — không bắn vào LLM, xem 0.3):
- Mô tả nhiệm vụ cho code: normalize raw `tool_result` thành scratchpad/evidence (deterministic).
- Với RAG, xác định `enough_context` dựa trên result count và top score (so threshold trong code).
- Với web, giữ source/snippet hữu ích.
- Với tool error, set `tool_ok=false` và ghi errors.

`prompts/nodes/final.md`:
- Define nhiệm vụ tạo final answer từ state, scratchpad và recommendations.
- Với recommendation, trả 3-5 bài và lý do ngắn gọn.
- Với web fallback, nói rõ trả lời dựa trên web context.
- Với failure, không bịa bài hát hoặc artist facts.

`state.py`:
- `AgentState` (đã có): `user_message`, `language`, `intent` (`AgentIntent | None`),
  `entities` (`ExtractedEntities`), `planned_tool` (`PlannedToolCall | None`),
  `tool_result` (`dict | None`, là wrapper từ `McpToolClient`), `scratchpad` (`dict`),
  `recommendations` (`list[Recommendation]`), `final_answer` (`str | None`),
  `confidence` (0..1), `iteration_count` (int), `errors` (`list[str]`).
  → `tool_ok`/`enough_context`/`last_tool`/`should_fallback_to_web` là KEY trong `scratchpad`,
  không phải field top-level.
- `ThinkDecision` (LLM-output của `think.md`): `thought`, `action` (`Literal["call_tool","respond"]`),
  `intent` (`AgentIntent`), `entities` (`ExtractedEntities`), `tool_name` (`ToolName | None`),
  `tool_input` (`dict | None`), `confidence` (0..1), `response` (`str | None`).
- `FinalAnswerDraft` (LLM-output của `final.md`): `status` (`AgentStatus`), `answer` (str),
  `recommendations` (`list[Recommendation]`). `trace` do code gắn, không để LLM tự sinh.
- `Observation` (tuỳ chọn, helper nội bộ cho observe — KHÔNG phải LLM schema): mô tả các key
  observe ghi vào `scratchpad` (`tool_ok`, `enough_context`, `summary`, `evidence`,
  `last_tool`, `should_fallback_to_web`).
- Validators:
  - `ThinkDecision`: `tool_name` ∈ `ToolName` (enum tự reject unknown); khi `action="respond"`
    thì `tool_name` và `tool_input` phải null; khi `action="call_tool"` thì `tool_name` bắt buộc.
  - confidence range `0.0 <= confidence <= 1.0`.
  - max result limits theo `MusicRagSearchInput.limit` / `ChatRequest.max_results`.

### Test checklist
- [x] System prompt nhắc đúng cả 2 tools.
- [x] System prompt chứa input schema cho `music_rag_search`.
- [x] System prompt chứa input schema cho `web_search`.
- [x] System prompt nói rõ greetings/smalltalk không cần gọi tool.
- [x] System prompt cấm fabricate songs nếu không có evidence.
- [x] `think.md` chỉ cho phép `call_tool` hoặc `respond`.
- [x] `think.md` yêu cầu JSON-only output.
- [x] `act.md` nói rõ không tự chọn tool mới và chỉ execute `planned_tool`.
- [x] `observe.md` nói rõ cách set `tool_ok` và `enough_context`.
- [x] `final.md` nói rõ không lộ scratchpad khi trả lời user.
- [x] `ThinkDecision` reject unknown tool names.
- [x] `ThinkDecision` accept `tool_name=null` khi `action="respond"`.
- [x] `ThinkDecision` reject `tool_name`/`tool_input` khác null khi `action="respond"`.
- [x] `ThinkDecision` reject thiếu `tool_name` khi `action="call_tool"`.
- [x] `AgentState` init với `iteration_count=0` và `scratchpad={}`.

### Done criteria
- `pytest tests/test_agent_state.py tests/test_prompt_contracts.py -q` pass.

## Phase 6 - LangGraph agent nodes

### Mục tiêu

Implement core loop `think -> act -> observe -> think/final` với deterministic routing và không có hidden retry.

### Quyết định Phase 6 (chốt trước khi code)

1. **Node async.** Cả 4 node là `async def` (0.2). `act`/`final` await `McpToolClient`/`LlmClient`;
   `think` await `LlmClient.complete_json`; `observe` đồng bộ logic nhưng vẫn khai báo async cho
   đồng nhất graph.
2. **`tool_result` = wrapper của `McpToolClient`.** `act` lưu nguyên wrapper
   `{ok, tool_name, duration_ms, result, error}` vào `state.tool_result`. Payload tool nằm ở
   `tool_result["result"]` (vd `{ok, results, result_count, diagnostics}` của RAG).
   `observe` đọc `tool_result["ok"]`, `tool_result["result"]`, `tool_result["error"]`.
   (`act.md` đã dùng đúng key `result` — khớp wrapper thật.)
3. **observe ghi vào `scratchpad`, không tạo field mới.** `tool_ok`/`enough_context`/`summary`/
   `evidence`/`last_tool`/`should_fallback_to_web` là key của `scratchpad`. `recommendations` và
   `errors` là field thật của `AgentState`.
4. **Routing theo `planned_tool`, không theo `should_finalize`.** `think` set `planned_tool=None`
   khi `action="respond"`. Edge `think -> final` khi `planned_tool is None`; `think -> act` khi có
   `planned_tool`. Hard stop `-> final` khi `iteration_count >= 2`.

### Module cần làm
- Node `think`.
- Node `act`.
- Node `observe`.
- Node `final`.
- LangGraph assembly và routing.

### File cần tạo
- `src/music_agent/agent/nodes/__init__.py`
- `src/music_agent/agent/nodes/think.py`
- `src/music_agent/agent/nodes/act.py`
- `src/music_agent/agent/nodes/observe.py`
- `src/music_agent/agent/nodes/final.py`
- `src/music_agent/agent/graph.py`
- `tests/test_agent_nodes.py`
- `tests/test_agent_graph_routes.py`

### Chi tiết triển khai

`think` node (async, gọi LLM):
- Input: current `AgentState`.
- Render `system.md` + `think.md`, gọi `LlmClient.complete_json(..., ThinkDecision)`.
- Map `ThinkDecision` vào state: luôn set `intent`, `entities`, `confidence`.
- Nếu `action="call_tool"`: set `planned_tool = PlannedToolCall(tool_name, tool_input, reason=thought, confidence)`.
- Nếu `action="respond"`: set `planned_tool=None`, lưu `response` vào `scratchpad["final_directive"]`
  để node `final` dùng.
- Logic loop nằm trong prompt + scratchpad: sau `observe`, nếu RAG fail/không đủ context và chưa
  gọi web (`scratchpad.should_fallback_to_web`), think chọn `web_search`; nếu đủ thì `respond`.
- `LlmOutputError` → ghi `errors`, set `planned_tool=None` để đi tới `final` (fail an toàn).

`act` node (async):
- Execute đúng một tool từ `planned_tool` (không tự chọn tool mới, không retry).
- `await McpToolClient.call_tool(planned_tool.tool_name, planned_tool.tool_input)`.
- Lưu nguyên wrapper trả về vào `state.tool_result` (`{ok, tool_name, duration_ms, result, error}`).
- Append một `ToolCallTrace` (tool_name, tool_input, reason/confidence từ `planned_tool`,
  `ok`/`duration_ms` từ wrapper).
- `iteration_count += 1`.

`observe` node (async, code thuần — không gọi LLM):
- Đọc `state.tool_result` (wrapper). Payload tool = `tool_result["result"]`.
- Ghi vào `scratchpad`: `tool_ok`, `enough_context`, `summary`, `evidence`, `last_tool`,
  `should_fallback_to_web`.
- Cập nhật `state.recommendations` và `state.errors` (field thật của state).
- Với RAG (`tool_result["result"]` = `{ok, results, result_count, diagnostics}`):
  - `enough_context=true` khi `result_count > 0` và top `score` vượt threshold (so trong code).
  - Map `results` → `Recommendation` (`song_id/title/artist/mood/genres/tags/preview_url/spotify_url/score`;
    `reason` để trống, `final` điền sau).
  - Nếu rỗng/low-confidence: `should_fallback_to_web=true`.
- Với web (`tool_result["result"]` = `{ok, results, sources, diagnostics}`):
  - `enough_context=true` khi có ít nhất một source/snippet hữu ích; giữ source URL trong evidence.
- Với tool error (`tool_result["ok"]=false`): `tool_ok=false`, ghi `error` vào `errors`.

`final` node (async, gọi LLM):
- Render `system.md` + `final.md` với state/scratchpad/recommendations, gọi
  `LlmClient.complete_json(..., FinalAnswerDraft)`.
- Map `FinalAnswerDraft` vào state: `final_answer` (= `answer`), `status`, và bổ sung `reason`
  cho từng `Recommendation`.
- Với RAG recommendations: return 3-5 bài, lý do gắn mood/query, include preview URL nếu có.
- Với web fallback: nói rõ answer đến từ web context.
- Với failure (`tool_ok=false` hoặc không đủ context / `LlmOutputError`): trả `status=failed`,
  nói không đủ context, KHÔNG invent song/artist/preview.

`graph.py`:
- Build LangGraph state graph (async nodes, invoke bằng `ainvoke`).
- Nodes: `think`, `act`, `observe`, `final`.
- Conditional edges:
  - `think -> final` khi `planned_tool is None` (tức `action="respond"`).
  - `think -> act` khi có `planned_tool`.
  - `act -> observe`.
  - `observe -> think`.
  - hard stop tới `final` khi `iteration_count >= 2` (chặn ngay ở routing sau `observe`/`think`).

### Test checklist
- [x] Greeting đi theo route `think -> final`.
- [x] Mood recommendation đi theo route `think -> act(music_rag_search) -> observe -> think -> final` khi RAG succeed.
- [x] Artist deep-dive đi theo route `think -> act(web_search) -> observe -> think -> final`.
- [x] Empty RAG result đi theo route `think -> act(RAG) -> observe -> think -> act(web_search) -> observe -> think -> final`.
- [x] Tool error đi tới final failure response.
- [x] Graph không bao giờ vượt quá 2 tool calls.
- [x] `act` không execute khi thiếu `planned_tool` (đi tới final fail).
- [x] `act` lưu wrapper `{ok, result, error, ...}` vào `tool_result`; `observe` đọc `tool_result["result"]`.
- [x] `observe` set `tool_ok`/`enough_context` trong `scratchpad` (không phải field top-level).
- [x] `think` set `planned_tool=None` khi `action="respond"`.
- [x] `LlmOutputError` ở `think`/`final` → fail an toàn, không crash graph.
- [x] `final` ẩn scratchpad khi `debug=false`.

### Done criteria
- `pytest tests/test_agent_nodes.py tests/test_agent_graph_routes.py -q` pass.
- Graph invoke bằng `ainvoke` từ plain async test, inject fake `LlmClient`/`McpToolClient`,
  không cần start FastAPI hay MCP server thật.

## Phase 7 - FastAPI service

### Mục tiêu

Expose agent graph qua HTTP API và cung cấp health/debug behavior phù hợp cho local development.

### Quyết định Phase 7 (chốt trước khi code)

1. **Async handler + `ainvoke`.** `POST /v1/chat` là `async def`. Build graph bằng
   `build_agent_graph(llm_client, mcp_client)` rồi `await graph.ainvoke(initial_state)`.
   `initial_state` = `AgentState(user_message=..., ...).model_dump(mode="python")`; output dict
   parse lại bằng `AgentState.model_validate(output)` (graph dùng TypedDict, vào/ra là dict).
2. **Graph + client khởi tạo 1 lần.** Tạo `LlmClient` + `McpToolClient` (trỏ `MCP_SERVER_URL`) và
   compile graph ở startup (lifespan/module-level), không dựng lại mỗi request. API và MCP server
   là 2 tiến trình tách biệt — API gọi tool qua MCP protocol.
3. **Map state → `ChatResponse` bằng field thật.** `status` ← `AgentState.status` (None →
   suy ra: có `recommendations`/`final_answer` → `ok`, ngược lại `failed`); `answer` ←
   `final_answer`; `recommendations` ← `AgentState.recommendations` (cắt còn `max_results`);
   `tool_calls` ← `AgentState.tool_calls`.
4. **`max_results` clamp ở API.** `ChatRequest.max_results` không tự chảy vào graph; API cắt
   `recommendations[:max_results]` khi shape response. (think LLM tự chọn `limit` trong tool_input.)
5. **`trace` chỉ khi `debug=true`.** `debug=false` → `trace=None`. `debug=true` → `trace` gồm
   `scratchpad`, `intent`, `entities`, `confidence`, `iteration_count`, `errors`.

### Module cần làm
- FastAPI app.
- Request validation.
- Agent invocation.
- Response shaping.
- Health endpoint.

### File cần tạo
- `src/music_agent/api/__init__.py`
- `src/music_agent/api/main.py`
- `tests/test_api.py`

### Chi tiết triển khai

`main.py`:
- Tạo FastAPI app (`/health` đã có sẵn, giữ nguyên).
- Endpoints:
  - `GET /health` → `{"status": "ok"}`.
  - `POST /v1/chat` (async).
- `POST /v1/chat` flow:
  - Validate `ChatRequest` (FastAPI tự trả `422` nếu sai).
  - Build initial `AgentState(user_message=request.message, ...)` → `model_dump(mode="python")`.
  - `await graph.ainvoke(initial_state)` (graph compile sẵn ở startup).
  - `AgentState.model_validate(output)` rồi map sang `ChatResponse` (xem quyết định #3, #4).
  - `trace=None` khi `debug=false`; gắn trace dict khi `debug=true` (quyết định #5).
- Error behavior:
  - Validation errors → HTTP `422` (mặc định FastAPI).
  - Agent fail (graph chạy xong nhưng `status=failed` hoặc có `errors`) → HTTP `200` với
    `status=failed`, không leak stack trace. Graph đã fail-safe nội bộ (think/final bắt
    `LlmOutputError`), nên API chỉ cần bọc try/except phòng lỗi ngoài dự kiến → `status=failed`.

### Test checklist
- [x] `GET /health` return `{"status": "ok"}`.
- [x] `POST /v1/chat` reject empty message.
- [x] `POST /v1/chat` return `status=ok` cho successful recommendation.
- [x] `POST /v1/chat` include recommendations array.
- [x] `debug=false` return `trace=null`.
- [x] `debug=true` return route/tool trace (scratchpad/intent/errors...).
- [x] `recommendations` bị cắt còn `max_results`.
- [x] API map graph failure thành `status=failed` và không leak stack trace.

### Done criteria
- `pytest tests/test_api.py -q` pass (inject fake `LlmClient`/`McpToolClient` qua dependency
  override, không cần MCP server hay Gemini thật).
- Manual API boot hoạt động:
```bash
uvicorn music_agent.api.main:app --host localhost --port 8000
```

## Phase 8 - End-to-end scenarios và documentation

### Mục tiêu

Verify full local system và document cách run, test, extend.

### Module cần làm
- E2E tests.
- README usage.
- Developer runbook.
- Future Qdrant extension notes.

### File cần sửa
- `README.md`

### File cần tạo
- `tests/test_e2e_agent_flow.py`

### Chi tiết triển khai

README phải có:
- Install command.
- Required env vars.
- Cách run MCP server.
- Cách run API server.
- Example curl calls.
- Expected response examples.
- Debug mode example.
- Cách thêm mock songs mới.
- Cách future Qdrant adapter replace `FixtureSongStore`.

E2E test approach (offline hoàn toàn, không cần key/network):
- Mock LLM outputs để deterministic `think` và `final` (fake `LlmClient` như ở
  `test_agent_graph_routes.py`).
- Hai tầng tùy mục tiêu test:
  - **HTTP contract**: inject fake `McpToolClient` qua dependency override → kiểm tra
    `/v1/chat` shape response, status, trace, clamp `max_results`. Không cần MCP server.
  - **Real RAG path** (muốn chạy fixture thật): inject deterministic `EmbeddingClient`
    (kiểu `KeywordEmbeddingClient` ở `test_fixture_store.py`) vào `FixtureSongStore` qua
    `set_song_store_for_testing`, chạy `rag_tool` qua MCP in-memory protocol (kiểu
    `test_mcp_client.py`). KHÔNG để E2E phụ thuộc `GEMINI_API_KEY`.
- Mock Tavily: inject fake `TavilySearchClient` vào `web_tool`.
- Dùng FastAPI test client (httpx ASGI transport) cho `/v1/chat`.
- Lưu ý độ phủ đã có: graph wiring (`test_agent_graph_routes`), MCP protocol thật
  (`test_mcp_client`), retrieval logic (`test_fixture_store`). E2E Phase 8 tập trung vào hợp đồng
  HTTP + response shaping + debug trace, không lặp lại các tầng đã test.

### Test checklist
- [x] E2E greeting không có tool calls.
- [x] E2E query sad/healing return song recommendations từ fixture RAG.
- [x] E2E unknown recommendation fallback từ RAG sang web search.
- [x] E2E artist deep-dive dùng web search đầu tiên.
- [x] E2E tool failure return non-fabricated failure answer.
- [x] README commands copy-paste chạy được.

### Done criteria
- Full suite pass:
```bash
pytest -q
ruff check .
```

## Phase 9 - Future production Qdrant phase, không thuộc V1

### Mục tiêu

Thay fixture retrieval bằng Qdrant-backed retrieval sau khi V1 agent contract đã ổn định.

### Module cần làm
- Song ingestion.
- Embedding generation.
- Qdrant collection management.
- Qdrant retrieval adapter.

### File sẽ tạo sau
- `src/music_agent/retrieval/qdrant_store.py`
- `src/music_agent/ingestion/song_loader.py`
- `src/music_agent/ingestion/index_qdrant.py`
- `tests/test_qdrant_store.py`
- `tests/test_ingestion.py`

### Chi tiết triển khai

Qdrant payload cho mỗi point = đúng `SongPayload` (Phase 1), KHÔNG đặt lại tên field. Tối thiểu:
- `chunk_id`, `song_id`, `title`, `artist`, `metadata_summary` (required theo `SongPayload`).
- `lyrics_summary`, `mood`, `genres`, `tags`, `preview_url`, `spotify_url`, ... (optional).

Document text để embed phải dùng `FixtureSongStore.build_document_text` (cùng format
`RETRIEVAL_DOCUMENT`), query dùng `build_query_text` — tái dùng, không viết lại, để behavior khớp.

Collection requirements:
- Một vector point cho mỗi chunk (V1: 1 chunk/bài → point ID deterministic từ `chunk_id`/`song_id`).
- Idempotent upsert.
- Payload filters cho `artist`, `genres`, `tags`, `mood`.

`QdrantStore` phải implement cùng interface `FixtureSongStore`:
- `ensure_index()`, `search(MusicRagSearchInput) -> MusicRagSearchResult`.
- `search` trả `results: list[SongPayload]` + `diagnostics.score_details[song_id] = {score, semantic_score, metadata_boost}`
  để `rag_tool` join score y như fixture. `rag_tool`/agent KHÔNG đổi khi swap store.

### Test checklist
- [ ] Ingestion đọc raw scraped song records.
- [ ] Ingestion reject record thiếu field required của `SongPayload`
      (`chunk_id`/`song_id`/`title`/`artist`/`metadata_summary`).
- [ ] Qdrant upsert idempotent theo `chunk_id`/`song_id`.
- [ ] `QdrantStore.search` return `MusicRagSearchResult` cùng shape `FixtureSongStore`
      (gồm `diagnostics.score_details`).
- [ ] `rag_tool` + agent tests không đổi khi swap `FixtureSongStore` sang `QdrantStore`
      (chỉ đổi binding ở `get_song_store`).

## Final acceptance checklist

- [ ] Project install thành công.
- [ ] MCP server start local được.
- [ ] API server start local được.
- [ ] `POST /v1/chat` handle greetings không gọi tool.
- [ ] `POST /v1/chat` recommend 3-5 bài hát cho mood queries.
- [ ] Agent emit tool selection reason và confidence trong debug mode.
- [ ] RAG no-context case fallback sang web search đúng một lần.
- [ ] Artist deep-dive dùng web search.
- [ ] Không response nào fabricate song recommendations khi thiếu RAG/web evidence.
- [ ] `pytest -q` pass.
- [ ] `ruff check .` pass.
