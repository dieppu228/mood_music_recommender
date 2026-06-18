"""Fixture-backed song retrieval for V1."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from music_agent.config import Settings, get_settings
from music_agent.models import MusicRagSearchInput, MusicRagSearchResult, SongPayload


class FixtureStoreError(RuntimeError):
    """Raised when the fixture song store cannot load or search records."""


class EmbeddingClient(Protocol):
    """Minimal embedding interface used by the fixture store."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document texts for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""


class GeminiEmbeddingClient:
    """Gemini embedding adapter.

    The store depends only on ``EmbeddingClient`` so tests can inject a deterministic fake
    and never call the network.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self.settings.embedding_document_task_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.settings.embedding_query_task_type)[0]

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.gemini_api_key:
            raise FixtureStoreError("GEMINI_API_KEY is required for Gemini embeddings")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.settings.gemini_api_key)
        result = client.models.embed_content(
            model=self.settings.embedding_model,
            contents=list(texts),
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.settings.embedding_output_dimensionality,
            ),
        )
        return [[float(value) for value in embedding.values] for embedding in result.embeddings]


class FixtureSongStore:
    """Load JSONL song payloads and perform vector + metadata boosted retrieval."""

    def __init__(
        self,
        path: str | Path,
        embedding_client: EmbeddingClient | None = None,
        settings: Settings | None = None,
        min_score: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.settings = settings or get_settings()
        self.embedding_client = embedding_client or GeminiEmbeddingClient(self.settings)
        self.min_score = min_score
        self._records: list[SongPayload] | None = None
        self._document_embeddings: list[list[float]] | None = None

    def load_records(self) -> list[SongPayload]:
        """Load and validate all song payloads from JSONL."""

        if self._records is not None:
            return self._records
        if not self.path.exists():
            raise FixtureStoreError(f"Song fixture file does not exist: {self.path}")

        records: list[SongPayload] = []
        with self.path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FixtureStoreError(
                        f"Malformed JSONL at {self.path}:{line_number}: {exc.msg}"
                    ) from exc
                try:
                    records.append(SongPayload.model_validate(raw))
                except ValidationError as exc:
                    raise FixtureStoreError(
                        f"Invalid song payload at {self.path}:{line_number}: {exc}"
                    ) from exc

        self._records = records
        return records

    def build_document_text(self, record: SongPayload) -> str:
        """Build the document text embedded with Gemini RETRIEVAL_DOCUMENT."""

        return "\n".join(
            [
                f"title: {record.title}",
                f"artist: {record.artist}",
                f"album: {record.album or ''}",
                f"metadata_summary: {record.metadata_summary}",
                f"lyrics_summary: {record.lyrics_summary or ''}",
                f"mood: {', '.join(record.mood)}",
                f"genres: {', '.join(record.genres)}",
                f"tags: {', '.join(record.tags)}",
                f"source: {record.source_name or ''}",
                f"search_query: {record.search_query or ''}",
            ]
        )

    def build_query_text(self, search_input: MusicRagSearchInput) -> str:
        """Build the query text embedded with Gemini RETRIEVAL_QUERY."""

        return "\n".join(
            [
                f"query: {search_input.query}",
                f"mood_terms: {', '.join(search_input.mood_terms)}",
                f"genres: {', '.join(search_input.genres)}",
                f"tags: {', '.join(search_input.tags)}",
                f"artist: {search_input.artist or ''}",
            ]
        )

    def ensure_index(self) -> None:
        """Build document embeddings lazily."""

        if self._document_embeddings is not None:
            return

        records = self.load_records()
        if not records:
            self._document_embeddings = []
            return

        texts = [self.build_document_text(record) for record in records]
        embeddings = self.embedding_client.embed_documents(texts)
        if len(embeddings) != len(records):
            raise FixtureStoreError(
                "Embedding count mismatch: "
                f"expected {len(records)}, got {len(embeddings)}"
            )
        self._document_embeddings = embeddings

    def search(
        self,
        search_input: MusicRagSearchInput | str,
        limit: int | None = None,
    ) -> MusicRagSearchResult:
        """Search songs by vector similarity with deterministic metadata boosts."""

        if isinstance(search_input, str):
            search_input = MusicRagSearchInput(query=search_input, limit=limit or 5)
        elif limit is not None:
            search_input = search_input.model_copy(update={"limit": limit})

        records = self.load_records()
        if not records:
            return MusicRagSearchResult(
                ok=True,
                results=[],
                result_count=0,
                diagnostics={"record_count": 0},
            )

        self.ensure_index()
        query_embedding = self.embedding_client.embed_query(self.build_query_text(search_input))
        scored: list[tuple[float, SongPayload, dict[str, float]]] = []

        for record, document_embedding in zip(records, self._document_embeddings or [], strict=True):
            semantic_score = cosine_similarity(query_embedding, document_embedding)
            boost = metadata_boost(record, search_input)
            final_score = semantic_score + boost
            if final_score >= self.min_score:
                scored.append(
                    (
                        final_score,
                        record,
                        {
                            "semantic_score": semantic_score,
                            "metadata_boost": boost,
                        },
                    )
                )

        scored.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].song_id))
        results = [record for score, record, _ in scored[: search_input.limit]]
        score_details = {
            record.song_id: {
                "score": round(score, 6),
                "semantic_score": round(details["semantic_score"], 6),
                "metadata_boost": round(details["metadata_boost"], 6),
            }
            for score, record, details in scored[: search_input.limit]
        }
        return MusicRagSearchResult(
            ok=True,
            results=results,
            result_count=len(results),
            diagnostics={
                "record_count": len(records),
                "score_details": score_details,
                "embedding_model": self.settings.embedding_model,
                "document_task_type": self.settings.embedding_document_task_type,
                "query_task_type": self.settings.embedding_query_task_type,
            },
        )


def metadata_boost(record: SongPayload, search_input: MusicRagSearchInput) -> float:
    """Return a small deterministic score boost for exact metadata matches."""

    boost = 0.0
    record_mood = normalized_set(record.mood)
    record_genres = normalized_set(record.genres)
    record_tags = normalized_set(record.tags)

    boost += 0.08 * len(record_mood & normalized_set(search_input.mood_terms))
    boost += 0.06 * len(record_genres & normalized_set(search_input.genres))
    boost += 0.04 * len(record_tags & normalized_set(search_input.tags))

    if search_input.artist:
        artist = normalize_token(search_input.artist)
        record_artists = normalized_set([record.artist, *record.artists])
        if artist in record_artists or artist in normalize_token(record.artist):
            boost += 0.12

    return min(boost, 0.3)


def normalized_set(values: Sequence[str]) -> set[str]:
    return {normalize_token(value) for value in values if normalize_token(value)}


def normalize_token(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two vectors."""

    if len(left) != len(right):
        raise FixtureStoreError(
            f"Vector dimension mismatch: left={len(left)}, right={len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
