from music_agent.config import Settings


def test_settings_defaults_are_localhost() -> None:
    settings = Settings()
    assert settings.api_host == "localhost"
    assert settings.mcp_host == "localhost"
    assert settings.mcp_server_url == "http://localhost:8001/mcp"
    assert settings.llm_base_url == "http://localhost:4000/v1"
    assert settings.qdrant_url == "http://localhost:6333"


def test_model_defaults_follow_plan() -> None:
    settings = Settings()
    assert settings.llm_model == "gemini-2.5-flash-lite"
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.embedding_output_dimensionality == 768
    assert settings.embedding_document_task_type == "RETRIEVAL_DOCUMENT"
    assert settings.embedding_query_task_type == "RETRIEVAL_QUERY"
    assert settings.mock_song_path == "data/mock_songs.jsonl"
