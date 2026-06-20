"""Fixture-backed song retrieval for V1."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import ValidationError

from music_agent.config import Settings, get_settings
from music_agent.models import MusicRagSearchInput, MusicRagSearchResult, SongPayload

FIELD_WEIGHTS = {
    "semantic": 0.55,
    "mood": 0.20,
    "genres": 0.20,
    "tags": 0.05,
}


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

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        batch_size: int = 100,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self.settings.embedding_document_task_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.settings.embedding_query_task_type)[0]

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        if self._client is None and not self.settings.gemini_api_key:
            raise FixtureStoreError("GEMINI_API_KEY is required for Gemini embeddings")

        from google import genai
        from google.genai import types

        client = self._client or genai.Client(api_key=self.settings.gemini_api_key)
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            result = client.models.embed_content(
                model=self.settings.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.settings.embedding_output_dimensionality,
                ),
            )
            embeddings.extend(
                [float(value) for value in embedding.values]
                for embedding in result.embeddings
            )
        return embeddings


class FixtureSongStore:
    """Load JSONL song payloads and perform vector + metadata boosted retrieval."""

    def __init__(
        self,
        path: str | Path,
        embedding_client: EmbeddingClient | None = None,
        settings: Settings | None = None,
        min_score: float = 0.0,
        embedding_cache_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.settings = settings or get_settings()
        self.embedding_client = embedding_client or GeminiEmbeddingClient(self.settings)
        if embedding_cache_path is not None:
            self.embedding_cache_path = Path(embedding_cache_path)
        elif embedding_client is None:
            self.embedding_cache_path = Path(self.settings.embedding_cache_path)
        else:
            self.embedding_cache_path = None
        self.min_score = min_score
        self._records: list[SongPayload] | None = None
        self._document_embeddings: np.ndarray | list[list[float]] | None = None

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
                f"album: {record.album or ''}",
                f"metadata_summary: {record.metadata_summary}",
                f"lyrics_summary: {record.lyrics_summary or ''}",
                f"mood: {', '.join(record.mood)}",
                f"genres: {', '.join(record.genres)}",
                f"tags: {', '.join(record.tags)}",
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
            ]
        )

    def ensure_index(self) -> None:
        """Load the document cache, or build in memory for injected test embedders."""

        if self._document_embeddings is not None:
            return

        records = self.load_records()
        if not records:
            self._document_embeddings = []
            return

        if self.embedding_cache_path is not None:
            self._document_embeddings = self.load_embedding_cache()
            return

        texts = [self.build_document_text(record) for record in records]
        embeddings = self.embedding_client.embed_documents(texts)
        if len(embeddings) != len(records):
            raise FixtureStoreError(
                "Embedding count mismatch: "
                f"expected {len(records)}, got {len(embeddings)}"
            )
        self._document_embeddings = embeddings

    @property
    def embedding_manifest_path(self) -> Path:
        if self.embedding_cache_path is None:
            raise FixtureStoreError("Embedding cache path is not configured")
        return self.embedding_cache_path.with_suffix(".meta.json")

    def build_embedding_cache(self) -> dict[str, Any]:
        """Embed the complete corpus and persist a validated NumPy artifact."""

        if self.embedding_cache_path is None:
            raise FixtureStoreError("Embedding cache path is not configured")
        records = self.load_records()
        texts = [self.build_document_text(record) for record in records]
        embeddings = self.embedding_client.embed_documents(texts)
        matrix = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (len(records), self.settings.embedding_output_dimensionality)
        if matrix.shape != expected_shape:
            raise FixtureStoreError(
                f"Embedding shape mismatch: expected {expected_shape}, got {matrix.shape}"
            )

        manifest = self._expected_embedding_manifest(len(records))
        self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = self.embedding_cache_path.with_suffix(".tmp.npy")
        temporary_manifest = self.embedding_manifest_path.with_suffix(".tmp.json")
        with temporary_cache.open("wb") as file:
            np.save(file, matrix, allow_pickle=False)
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_cache.replace(self.embedding_cache_path)
        temporary_manifest.replace(self.embedding_manifest_path)
        self._document_embeddings = matrix
        return manifest

    def load_embedding_cache(self) -> np.ndarray:
        """Load an embedding artifact only when its manifest matches the corpus."""

        if self.embedding_cache_path is None:
            raise FixtureStoreError("Embedding cache path is not configured")
        if not self.embedding_cache_path.exists() or not self.embedding_manifest_path.exists():
            raise FixtureStoreError(
                "Embedding cache is missing; run scripts/build_embedding_cache.py"
            )
        try:
            manifest = json.loads(self.embedding_manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise FixtureStoreError(f"Invalid embedding cache manifest: {exc}") from exc

        expected = self._expected_embedding_manifest(len(self.load_records()))
        if manifest != expected:
            raise FixtureStoreError(
                "Embedding cache is stale or incompatible; "
                "run scripts/build_embedding_cache.py"
            )
        try:
            matrix = np.load(self.embedding_cache_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise FixtureStoreError(f"Invalid embedding cache: {exc}") from exc
        expected_shape = (expected["record_count"], expected["output_dimensionality"])
        if matrix.shape != expected_shape:
            raise FixtureStoreError(
                f"Embedding cache shape mismatch: expected {expected_shape}, got {matrix.shape}"
            )
        return matrix

    def _expected_embedding_manifest(self, record_count: int) -> dict[str, Any]:
        return {
            "corpus_path": str(self.path),
            "corpus_sha256": sha256(self.path.read_bytes()).hexdigest(),
            "record_count": record_count,
            "embedding_model": self.settings.embedding_model,
            "document_task_type": self.settings.embedding_document_task_type,
            "output_dimensionality": self.settings.embedding_output_dimensionality,
            "dtype": "float32",
        }

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

        document_embeddings = self._document_embeddings
        if document_embeddings is None:
            raise FixtureStoreError("Document embeddings are not loaded")
        for record, document_embedding in zip(records, document_embeddings, strict=True):
            if search_input.artist and not artist_matches(record.artist, search_input.artist):
                continue
            semantic_score = max(0.0, min(1.0, cosine_similarity(query_embedding, document_embedding)))
            semantic_component = FIELD_WEIGHTS["semantic"] * semantic_score
            components = metadata_components(record, search_input)
            metadata_score = sum(components.values())
            final_score = semantic_component + metadata_score
            if final_score >= self.min_score:
                scored.append(
                    (
                        final_score,
                        record,
                        {
                            "semantic_score": semantic_score,
                            "semantic_component": semantic_component,
                            "metadata_boost": metadata_score,
                            **components,
                        },
                    )
                )

        scored.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].song_id))
        results = [record for score, record, _ in scored[: search_input.limit]]
        score_details = {
            record.song_id: {
                "score": round(score, 6),
                "semantic_score": round(details["semantic_score"], 6),
                "semantic_component": round(details["semantic_component"], 6),
                "metadata_boost": round(details["metadata_boost"], 6),
                "mood_score": round(details["mood_score"], 6),
                "genre_score": round(details["genre_score"], 6),
                "tag_score": round(details["tag_score"], 6),
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


def metadata_components(
    record: SongPayload,
    search_input: MusicRagSearchInput,
) -> dict[str, float]:
    """Return bounded field scores whose maximum plus semantic weight is one."""

    return {
        "mood_score": weighted_overlap(record.mood, search_input.mood_terms, FIELD_WEIGHTS["mood"]),
        "genre_score": weighted_overlap(
            record.genres,
            search_input.genres,
            FIELD_WEIGHTS["genres"],
        ),
        "tag_score": weighted_overlap(record.tags, search_input.tags, FIELD_WEIGHTS["tags"]),
    }


def weighted_overlap(
    record_values: Sequence[str],
    query_values: Sequence[str],
    field_weight: float,
) -> float:
    query_terms = normalized_set(query_values)
    if not query_terms:
        return 0.0
    matches = normalized_set(record_values) & query_terms
    return field_weight * len(matches) / len(query_terms)


def artist_matches(record_artist: str, query_artist: str) -> bool:
    record_value = normalize_token(record_artist)
    query_value = normalize_token(query_artist)
    return query_value == record_value or query_value in record_value


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
