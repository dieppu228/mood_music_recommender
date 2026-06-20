# Data files

## Spotify crawl snapshot

`spocrawl/` is kept in the repository root for reporting and partner handoff context. The
files below are copied or normalized from that crawl for the music agent.

- `spotify_songs.jsonl`: production V1 retrieval corpus generated from
  `spocrawl/data/processed/spotify_audio_training.csv` (2,261 songs with preview URLs).
- `mock_songs.jsonl`: deterministic balanced sample with five songs per canonical mood,
  used by focused tests and smoke checks.

Regenerate both files from the repository root:

```bash
python scripts/build_song_jsonl.py
```

## Normalized JSONL schema

Each line in `spotify_songs.jsonl` is one song object with the retrieval-facing fields:

- `chunk_id`
- `song_id`
- `title`
- `artist`
- `album`
- `metadata_summary`
- `lyrics_summary`
- `mood`
- `genres`
- `tags`
- `preview_url`
- `spotify_url`
- `payload_version`

The crawl does not contain real lyric summaries yet. `metadata_summary` is built from title,
album, release year, canonical mood, and tags. Artist remains payload-only for display/filter and
is excluded from semantic text. `lyrics_summary` remains null until a
separate lyric pipeline is available.

`FixtureSongStore.build_document_text` builds the canonical retrieval document from this payload.
Corpus text uses Gemini `RETRIEVAL_DOCUMENT`; user queries use `RETRIEVAL_QUERY`, both at 768
dimensions. Audio features from the source CSV are intentionally not included in V1 retrieval.
