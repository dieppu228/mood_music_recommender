"""Build the Gemini document embedding cache for the song corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_agent.config import get_settings
from music_agent.retrieval.fixture_store import FixtureSongStore


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(settings.mock_song_path))
    parser.add_argument("--output", type=Path, default=Path(settings.embedding_cache_path))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    store = FixtureSongStore(
        args.corpus,
        settings=settings,
        embedding_cache_path=args.output,
    )
    manifest = store.build_embedding_cache()
    print(
        f"Wrote {manifest['record_count']} x {manifest['output_dimensionality']} "
        f"embeddings to {args.output}"
    )


if __name__ == "__main__":
    main()
