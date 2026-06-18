"""Application settings for local-first services."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings shared across API, agent, and tools."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_host: str = "localhost"
    api_port: int = 8000

    mcp_host: str = "localhost"
    mcp_port: int = 8001
    mcp_server_url: str = "http://localhost:8001/mcp"

    llm_base_url: str = "http://localhost:4000/v1"
    llm_model: str = "gemini-2.5-flash-lite"
    llm_api_key: str = Field(default="", repr=False)

    gemini_api_key: str = Field(default="", repr=False)
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_output_dimensionality: int = 768
    embedding_document_task_type: str = "RETRIEVAL_DOCUMENT"
    embedding_query_task_type: str = "RETRIEVAL_QUERY"

    tavily_api_key: str = Field(default="", repr=False)
    mock_song_path: str = "data/mock_songs.jsonl"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "music_songs"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings()
