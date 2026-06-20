# Music Mood Dashboard UI

Static chat dashboard for the local Music Mood Agent API.

## Run

Start backend services first:

```bash
uvicorn music_agent.mcp_server.server:app --host localhost --port 8001
uvicorn music_agent.api.main:app --host localhost --port 8000
```

Serve the UI:

```bash
cd app
python3 -m http.server 5173
```

Open:

```text
http://localhost:5173
```

The dashboard calls:

```text
http://localhost:8000/v1/chat
```

Use the API URL field if the backend runs on another local port.
