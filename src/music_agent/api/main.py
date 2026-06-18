"""FastAPI entrypoint for the music mood agent."""

from fastapi import FastAPI

app = FastAPI(title="Music Mood Agent")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
