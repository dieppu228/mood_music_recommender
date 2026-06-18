# Data files

## Spotify crawl snapshot

`spocrawl/` is kept in the repository root for reporting and partner handoff context. The
files below are copied or normalized from that crawl for the music agent.

- `spotify_hybrid_tracks_raw.csv`: raw Spotify crawl export.
- `spotify_hybrid_processed.csv`: partner processed CSV with mood labels, tags, normalized
  text fields, and processing metadata.
- `spotify_processing_summary.json`: processing summary from the partner pipeline.
- `spotify_songs.jsonl`: normalized Spotify retrieval input, kept for later data work.
- `mock_songs.jsonl`: hand-written fixture with the full chunk payload schema, used by
  `MOCK_SONG_PATH` during Phase 1/2.

## Normalized JSONL schema

Each line in `spotify_songs.jsonl` is one song object with the retrieval-facing fields:

- `song_id`
- `title`
- `artist`
- `lyric_summary`
- `mood`
- `genres`
- `tags`
- `preview_url`
- `spotify_url`
- `combined_text`

The crawl does not contain real lyric summaries yet. For now, `lyric_summary` is a metadata
summary built from Spotify title, artist, album, mood label, tags, and source playlist/category.
Do not treat it as lyric-derived content until the lyric summarization pipeline is added.
