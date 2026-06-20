import json
from collections import Counter
from pathlib import Path

from scripts.build_song_jsonl import (
    CANONICAL_MOODS,
    build_metadata_summary,
    convert_csv,
    select_balanced_mock,
)


SOURCE_CSV = Path("spocrawl/data/processed/spotify_audio_training.csv")
PAYLOAD_FIELDS = {
    "chunk_id",
    "song_id",
    "title",
    "artist",
    "album",
    "metadata_summary",
    "lyrics_summary",
    "mood",
    "genres",
    "tags",
    "preview_url",
    "spotify_url",
    "payload_version",
}


def test_convert_audio_training_csv_to_lean_song_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "songs.jsonl"

    records = convert_csv(SOURCE_CSV, output)

    assert len(records) == 2261
    assert output.exists()
    assert all(set(record) == PAYLOAD_FIELDS for record in records)
    assert all(record["preview_url"] for record in records)
    assert all(set(record["mood"]).issubset(CANONICAL_MOODS) for record in records)
    first_line = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first_line == records[0]


def test_select_balanced_mock_uses_five_songs_per_canonical_mood(tmp_path: Path) -> None:
    records = convert_csv(SOURCE_CSV, tmp_path / "songs.jsonl")

    selected = select_balanced_mock(records, per_mood=5)

    counts = Counter(record["mood"][0] for record in selected)
    assert len(selected) == 30
    assert counts == Counter({mood: 5 for mood in CANONICAL_MOODS})


def test_metadata_summary_excludes_artist_from_semantic_text() -> None:
    summary = build_metadata_summary(
        {
            "track_name": "Quiet Light",
            "primary_artist": "Unique Artist Name",
            "album": "Still Rooms",
            "release_year": "2024",
            "mood_label": "calm",
            "tags_normalized": "calm,relaxing",
        }
    )

    assert "Unique Artist Name" not in summary
    assert "Quiet Light" in summary
    assert "Mood: calm" in summary
