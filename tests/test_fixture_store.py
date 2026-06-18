import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from music_agent.models import MusicRagSearchInput
from music_agent.retrieval.fixture_store import (
    FixtureSongStore,
    FixtureStoreError,
    cosine_similarity,
)


class KeywordEmbeddingClient:
    """Deterministic test embedder with dimensions for expected search concepts."""

    dimensions = {
        "sad": 0,
        "healing": 1,
        "energetic": 2,
        "workout": 3,
        "romantic": 4,
        "calm": 5,
        "anxiety": 6,
        "local echo": 7,
    }

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        vector = [0.0] * len(self.dimensions)
        for keyword, index in self.dimensions.items():
            if keyword in lowered:
                vector[index] = 1.0
        return vector


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def song(
    song_id: str,
    title: str,
    artist: str,
    metadata_summary: str,
    *,
    mood: list[str] | None = None,
    genres: list[str] | None = None,
    tags: list[str] | None = None,
    preview_url: str | None = None,
) -> dict:
    return {
        "chunk_id": f"spotify_track:{song_id}",
        "song_id": song_id,
        "title": title,
        "artist": artist,
        "artists": [artist],
        "album": "Test Album",
        "release_date": "2024-01-01",
        "release_year": 2024,
        "duration_ms": 180000,
        "popularity": 50,
        "explicit": False,
        "metadata_summary": metadata_summary,
        "lyrics_summary": metadata_summary,
        "lyrics_available": True,
        "mood": mood or [],
        "genres": genres or [],
        "tags": tags or [],
        "preview_url": preview_url,
        "spotify_url": f"https://open.spotify.com/track/{song_id}",
        "source_name": "Unit Test",
        "source_type": "mock",
        "search_query": "unit test",
        "mood_inferred": False,
        "data_origin": "test",
        "payload_version": "v1",
    }


def test_loads_valid_jsonl_records(tmp_path: Path) -> None:
    path = tmp_path / "songs.jsonl"
    write_jsonl(
        path,
        [
            song(
                "s1",
                "After Rain",
                "Local Echo",
                "sad healing rain recovery",
                mood=["sad", "healing"],
                preview_url=None,
            )
        ],
    )
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())

    records = store.load_records()

    assert len(records) == 1
    assert records[0].title == "After Rain"
    assert records[0].preview_url is None


def test_raises_clear_error_for_malformed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"song_id": "broken"\n', encoding="utf-8")
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())

    with pytest.raises(FixtureStoreError, match="Malformed JSONL"):
        store.load_records()


def test_raises_clear_error_for_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    write_jsonl(path, [{"song_id": "missing-required-fields"}])
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())

    with pytest.raises(FixtureStoreError, match="Invalid song payload"):
        store.load_records()


def test_builds_searchable_text_from_summary_mood_genres_and_tags(tmp_path: Path) -> None:
    path = tmp_path / "songs.jsonl"
    write_jsonl(
        path,
        [
            song(
                "s1",
                "Quiet Room",
                "Mina Vale",
                "late night memory comfort",
                mood=["lonely", "calm"],
                genres=["acoustic"],
                tags=["piano", "late night"],
            )
        ],
    )
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())
    record = store.load_records()[0]

    text = store.build_document_text(record)

    assert "metadata_summary: late night memory comfort" in text
    assert "lyrics_summary: late night memory comfort" in text
    assert "mood: lonely, calm" in text
    assert "genres: acoustic" in text
    assert "tags: piano, late night" in text


def test_returns_top_n_results_based_on_limit(tmp_path: Path) -> None:
    path = tmp_path / "songs.jsonl"
    write_jsonl(
        path,
        [
            song("s1", "After Rain", "Local Echo", "sad healing", mood=["sad"]),
            song("s2", "Neon Run", "Pulse City", "energetic workout", mood=["energetic"]),
            song("s3", "Quiet Room", "Mina Vale", "calm", mood=["calm"]),
        ],
    )
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())

    result = store.search(MusicRagSearchInput(query="sad healing", limit=2))

    assert result.ok is True
    assert result.result_count == 2
    assert len(result.results) == 2


def test_ranks_sad_healing_song_above_unrelated_energetic_song(tmp_path: Path) -> None:
    path = tmp_path / "songs.jsonl"
    write_jsonl(
        path,
        [
            song(
                "sad",
                "After Rain",
                "Local Echo",
                "sad healing recovery",
                mood=["sad", "healing"],
                tags=["rain", "recovery"],
            ),
            song(
                "energy",
                "Neon Run",
                "Pulse City",
                "energetic workout speed",
                mood=["energetic"],
                tags=["workout"],
            ),
        ],
    )
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())

    result = store.search(
        MusicRagSearchInput(query="sad healing song", mood_terms=["sad", "healing"], limit=2)
    )

    assert [record.song_id for record in result.results] == ["sad", "energy"]
    sad_score = result.diagnostics["score_details"]["sad"]["score"]
    energy_score = result.diagnostics["score_details"]["energy"]["score"]
    assert sad_score > energy_score


def test_applies_artist_boost_when_artist_is_provided(tmp_path: Path) -> None:
    path = tmp_path / "songs.jsonl"
    write_jsonl(
        path,
        [
            song("local", "After Rain", "Local Echo", "sad healing", mood=["sad"]),
            song("other", "Mirror Lake", "Other Artist", "sad healing", mood=["sad"]),
        ],
    )
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())

    result = store.search(
        MusicRagSearchInput(query="sad healing", mood_terms=["sad"], artist="Local Echo", limit=2)
    )

    assert result.results[0].song_id == "local"
    assert (
        result.diagnostics["score_details"]["local"]["metadata_boost"]
        > result.diagnostics["score_details"]["other"]["metadata_boost"]
    )


def test_handles_empty_optional_preview_url(tmp_path: Path) -> None:
    path = tmp_path / "songs.jsonl"
    write_jsonl(
        path,
        [song("s1", "Sunday Window", "June Harbor", "calm morning", preview_url=None)],
    )
    store = FixtureSongStore(path, embedding_client=KeywordEmbeddingClient())

    result = store.search(MusicRagSearchInput(query="calm", limit=1))

    assert result.results[0].preview_url is None


def test_returns_empty_result_list_when_no_records_exist(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    embedder = KeywordEmbeddingClient()
    store = FixtureSongStore(path, embedding_client=embedder)

    result = store.search(MusicRagSearchInput(query="anything", limit=5))

    assert result.ok is True
    assert result.results == []
    assert result.result_count == 0
    assert embedder.document_calls == 0


def test_builds_document_embeddings_lazily_once(tmp_path: Path) -> None:
    path = tmp_path / "songs.jsonl"
    write_jsonl(path, [song("s1", "After Rain", "Local Echo", "sad healing")])
    embedder = KeywordEmbeddingClient()
    store = FixtureSongStore(path, embedding_client=embedder)

    store.search(MusicRagSearchInput(query="sad", limit=1))
    store.search(MusicRagSearchInput(query="healing", limit=1))

    assert embedder.document_calls == 1
    assert embedder.query_calls == 2


def test_builds_query_text_from_extracted_metadata(tmp_path: Path) -> None:
    store = FixtureSongStore(tmp_path / "unused.jsonl", embedding_client=KeywordEmbeddingClient())

    text = store.build_query_text(
        MusicRagSearchInput(
            query="songs for recovery",
            mood_terms=["sad", "healing"],
            genres=["indie pop"],
            tags=["rain"],
            artist="Local Echo",
        )
    )

    assert "query: songs for recovery" in text
    assert "mood_terms: sad, healing" in text
    assert "genres: indie pop" in text
    assert "tags: rain" in text
    assert "artist: Local Echo" in text


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(FixtureStoreError, match="Vector dimension mismatch"):
        cosine_similarity([1.0, 0.0], [1.0])
