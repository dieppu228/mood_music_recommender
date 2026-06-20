# Music Mood Agent

Python service for mood-oriented music recommendations using an agent loop
(`think -> act -> observe -> final`), MCP tools, and Gemini embeddings.

## Project Overview

Music Mood Agent recommends songs from natural-language requests about emotions, activities,
genres, or listening preferences. Instead of treating the user's current emotion as the desired
music mood, the agent separates **current mood** from **target mood** before retrieval.

For example, a message such as "I feel sad" can be rewritten toward calm, happy, healing music,
while an explicit request such as "I feel sad and want sad songs" keeps `sad` as the target. This
prevents semantic search from automatically reinforcing a negative mood when the user did not ask
for that outcome. Target moods are constrained to six canonical values: `happy`, `sad`, `calm`,
`energetic`, `romantic`, and `stressed`.

The current V1 corpus contains 2,261 Spotify tracks enriched with mood labels, tags, Deezer preview
audio, and Spotify URLs. Each track is represented as one retrieval document and embedded once with
`gemini-embedding-001`; the resulting `2261 x 768` matrix is stored as a validated NumPy artifact.

## Agent Workflow

```text
User request
    |
    v
Think: classify intent, extract entities, choose target mood, rewrite retrieval query
    |
    +---------------- direct response -------------------------------+
    |                                                               |
    v                                                               v
Act: execute music_rag_search or web_search over MCP              Final
    |
    v
Observe: validate tool output, collect evidence, decide fallback
    |
    +---------------- back to Think ---------------------------------+
```

- **Think** handles intent classification, mood regulation, entity extraction, query rewriting,
  tool selection, confidence, and routing reasons.
- **Act** executes the selected tool through an MCP streamable HTTP server.
- **Observe** normalizes tool results into scratchpad evidence and structured recommendations. A
  weak or empty RAG result can trigger one web-search fallback.
- **Final** generates a grounded response using only the accumulated evidence and recommendations.

Small talk and out-of-domain requests bypass retrieval. Artist deep-dives use web search directly.
If the user insults the bot, the agent can apologize while still extracting the underlying mood and
returning calming recommendations.

## Retrieval Design

The internal `music_rag_search` tool combines asymmetric semantic retrieval with deterministic
metadata scoring:

- Corpus documents use Gemini task type `RETRIEVAL_DOCUMENT`.
- Rewritten user queries use `RETRIEVAL_QUERY`.
- Final score weights are semantic `0.55`, mood `0.20`, genres `0.20`, and tags `0.05`.
- Keyword-field weight is divided across the requested terms, keeping the maximum total score at
  `1.0`.
- Artist is excluded from semantic text and scoring; it is an exact normalized payload filter.
- Web results can be mapped back to playable catalog songs when both title and artist are mentioned.

The corpus embedding artifact includes a manifest with the corpus SHA-256, row count, model, task
type, dimension, and dtype. A stale or incompatible artifact fails explicitly instead of silently
re-embedding the complete corpus during a user request.

## Main Components

- `src/music_agent/agent/`: LangGraph state, routing, prompts, mood policy, nodes, and JSONL tracing.
- `src/music_agent/retrieval/`: fixture-backed semantic retrieval and Gemini embedding adapter.
- `src/music_agent/mcp_server/`: MCP tools for catalog RAG and Tavily web search.
- `src/music_agent/api/`: FastAPI chat API and same-origin dashboard delivery.
- `app/`: static chat dashboard with recommendations, previews, Spotify links, and debug trace.
- `spocrawl/`: Spotify/Deezer collection, processing scripts, reports, and source snapshots.
- `data/`: lean song payloads, balanced mock data, and cached document embeddings.

## V1 Scope

V1 is text retrieval only. Audio-feature similarity, Qdrant persistence, reranking, authentication,
and durable conversation memory are intentionally deferred. The local fixture store follows the
same payload and result contracts planned for a future Qdrant adapter.

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
- Future Qdrant: `http://localhost:6333`

## Environment

Required for live local runs:

```bash
LLM_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=your-gemini-key
MCP_SERVER_URL=http://localhost:8001/mcp
TAVILY_API_KEY=your-tavily-key
MOCK_SONG_PATH=data/spotify_songs.jsonl
EMBEDDING_CACHE_PATH=data/spotify_songs.embeddings.npy
```

Notes:

- Tests do not require real API keys or network access.
- `GEMINI_API_KEY` is used for the direct Gemini LLM and `gemini-embedding-001` retrieval.
- `TAVILY_API_KEY` is only needed when `web_search` should call Tavily.
- V1 has no reranker. Retrieval uses normalized weights: semantic `0.55`, mood `0.20`, genres
  `0.20`, tags `0.05`. Artist is an exact payload filter and is not embedded or scored.

## Run Services

Build the document embedding artifact whenever the corpus or embedding configuration changes:

```bash
python scripts/build_embedding_cache.py
```

The MCP service validates and loads this `.npy` artifact at runtime. It does not re-embed the
corpus during startup or the first search.

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

Each `/v1/chat` request appends detailed agent-loop events to `logs/agent_loop.jsonl`.
Events share a `request_id` and include node input/output, rewritten retrieval queries,
MCP call input/result, routing, latency, and the final API response. Override the location with
`AGENT_TRACE_LOG_PATH`.

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

Default V1 corpus lives at:

```text
data/spotify_songs.jsonl
```

It is generated from the 2,261-row Spotify/Deezer snapshot:

```bash
python scripts/build_song_jsonl.py
```

This also regenerates `data/mock_songs.jsonl` as a balanced 30-song test sample. Each line is one
lean `SongPayload` JSON object:

```json
{
  "chunk_id": "spotify_track:mock-001",
  "song_id": "mock-001",
  "title": "After Rain",
  "artist": "Local Echo",
  "album": "Weather Inside",
  "metadata_summary": "After Rain - Local Echo. Album: Weather Inside. Mood: calm.",
  "lyrics_summary": null,
  "mood": ["calm"],
  "genres": [],
  "tags": ["healing"],
  "preview_url": "https://example.com/preview/mock-001",
  "spotify_url": "https://open.spotify.com/track/mock-001",
  "payload_version": "v1"
}
```

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
---------------------------------------------------------------------------
MCP : source .venv/bin/activate
uvicorn music_agent.mcp_server.server:app --host localhost --port 8001

API: source .venv/bin/activate
uvicorn music_agent.api.main:app --host localhost --port 8000

UI: cd app
python3 -m http.server 5173

## Public Demo

Cloudflare Quick Tunnel (temporary):

https://pine-choir-smtp-killing.trycloudflare.com
