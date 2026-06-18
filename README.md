# Music Mood Agent

Python service for mood-oriented music recommendations.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Local services

All services are configured to run on localhost by default:
- API: `http://localhost:8000`
- MCP server: `http://localhost:8001/mcp`
- LLM gateway: `http://localhost:4000/v1`
- Future Qdrant: `http://localhost:6333`

Copy `.env.example` to `.env` and fill secrets before running.

## Planned entrypoints

```bash
uvicorn music_agent.mcp_server.server:app --host localhost --port 8001
uvicorn music_agent.api.main:app --host localhost --port 8000
```
