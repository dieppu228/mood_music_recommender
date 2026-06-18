# Music Mood Agent

Python service for mood-oriented music recommendations using an agent loop
(`think -> act -> observe -> final`), MCP tools, and Gemini embeddings.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

All local services default to `localhost`:

- API: `http://localhost:8000`
- MCP server: `http://localhost:8001/mcp`
- LLM gateway: `http://localhost:4000/v1`
- Future Qdrant: `http://localhost:6333`

## Environment

Required for live local runs:

```bash
LLM_BASE_URL=http://localhost:4000/v1
LLM_MODEL=gemini-2.5-flash-lite
LLM_API_KEY=
GEMINI_API_KEY=your-gemini-key
MCP_SERVER_URL=http://localhost:8001/mcp
TAVILY_API_KEY=your-tavily-key
MOCK_SONG_PATH=data/mock_songs.jsonl
```

Notes:

- Tests do not require real API keys or network access.
- `GEMINI_API_KEY` is used by `gemini-embedding-001` for live fixture retrieval.
- `TAVILY_API_KEY` is only needed when `web_search` should call Tavily.
- V1 has no reranker. Retrieval uses cosine score plus deterministic metadata boost.

## Run Services

Terminal 1, run MCP tools:

```bash
source .venv/bin/activate
uvicorn music_agent.mcp_server.server:app --host localhost --port 8001
```

Terminal 2, run API:

```bash
source .venv/bin/activate
uvicorn music_agent.api.main:app --host localhost --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

## Chat API

Mood recommendation:

```bash
curl -s http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"goi y nhac buon healing","max_results":3}'
```

Expected shape:

```json
{
  "status": "ok",
  "answer": "Natural-language answer from the final node",
  "recommendations": [
    {
      "song_id": "mock-001",
      "title": "After Rain",
      "artist": "Local Echo",
      "mood": ["sad", "healing"],
      "genres": ["indie pop"],
      "tags": ["rain"],
      "reason": "Short reason tied to the user mood",
      "preview_url": null,
      "spotify_url": "https://open.spotify.com/track/mock-001",
      "score": 0.92
    }
  ],
  "tool_calls": [
    {
      "tool_name": "music_rag_search",
      "tool_input": {},
      "reason": "Need RAG.",
      "confidence": 0.8,
      "ok": true,
      "duration_ms": 12.0
    }
  ],
  "trace": null
}
```

Artist deep-dive:

```bash
curl -s http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"noi sau ve Frank Ocean"}'
```

Debug mode:

```bash
curl -s http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"goi y nhac buon healing","debug":true}'
```

When `debug=true`, `trace` includes:

```json
{
  "scratchpad": {},
  "intent": "music_recommendation",
  "entities": {},
  "confidence": 0.87,
  "iteration_count": 1,
  "errors": [],
  "tool_calls": []
}
```

When `debug=false`, `trace` is always `null`.

## Mock Song Data

Default mock data lives at:

```text
data/mock_songs.jsonl
```

Each line must be one `SongPayload` JSON object. Required fields:

```json
{
  "chunk_id": "spotify_track:mock-001",
  "song_id": "mock-001",
  "title": "After Rain",
  "artist": "Local Echo",
  "artists": ["Local Echo"],
  "metadata_summary": "sad healing recovery",
  "mood": ["sad", "healing"],
  "genres": ["indie pop"],
  "tags": ["rain"],
  "data_origin": "mock",
  "payload_version": "v1"
}
```

Useful optional fields:

- `lyrics_summary`
- `preview_url`
- `spotify_url`
- `release_date`
- `release_year`
- `search_query`

After adding songs, restart the MCP server so the fixture store reloads records.

## Tests

Run the full offline suite:

```bash
pytest -q
ruff check .
```

Focused checks:

```bash
pytest tests/test_e2e_agent_flow.py -q
pytest tests/test_api.py -q
pytest tests/test_agent_graph_routes.py -q
```

## Future Qdrant Adapter

V1 uses `FixtureSongStore` as the retrieval backend. The future Qdrant adapter should preserve the
same contracts:

- Store one point per song chunk.
- Use `SongPayload` as the point payload without renaming fields.
- Reuse `FixtureSongStore.build_document_text(record)` for document text.
- Reuse `FixtureSongStore.build_query_text(search_input)` for query text.
- Keep Gemini `gemini-embedding-001` with:
  - `RETRIEVAL_DOCUMENT` for song document embeddings.
  - `RETRIEVAL_QUERY` for user query embeddings.
- Use Qdrant payload filters for `artist`, `genres`, `tags`, and `mood`.

The adapter should replace retrieval behind the MCP `music_rag_search` tool without changing the
agent graph, API response model, or prompt contracts.
