"""Convert the processed Spotify audio snapshot into lean retrieval JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

CANONICAL_MOODS = ("happy", "sad", "calm", "energetic", "romantic", "stressed")
PAYLOAD_VERSION = "v1"


def clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.casefold() == "nan" else text


def split_values(value: object) -> list[str]:
    values = []
    for item in clean(value).replace("|", ",").split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    return values


def normalize_year(value: object) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def build_metadata_summary(row: Mapping[str, object]) -> str:
    title = clean(row.get("track_name"))
    album = clean(row.get("album"))
    year = normalize_year(row.get("release_year"))
    mood = clean(row.get("mood_label")).casefold()
    tags = split_values(row.get("tags_normalized") or row.get("tags"))

    parts = [f"Title: {title}."]
    if album:
        parts.append(f"Album: {album}.")
    if year:
        parts.append(f"Release year: {year}.")
    parts.append(f"Mood: {mood}.")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}.")
    return " ".join(parts)


def build_song_record(row: Mapping[str, object]) -> dict[str, object]:
    song_id = clean(row.get("track_id"))
    title = clean(row.get("track_name"))
    artist = clean(row.get("primary_artist")) or clean(row.get("artists")).split(",")[0].strip()
    mood = clean(row.get("mood_label")).casefold()
    if not song_id or not title or not artist:
        raise ValueError("track_id, track_name, and primary_artist/artists are required")
    if mood not in CANONICAL_MOODS:
        raise ValueError(f"unsupported mood_label for {song_id}: {mood!r}")

    return {
        "chunk_id": f"spotify_track:{song_id}",
        "song_id": song_id,
        "title": title,
        "artist": artist,
        "album": clean(row.get("album")) or None,
        "metadata_summary": build_metadata_summary(row),
        "lyrics_summary": None,
        "mood": [mood],
        "genres": split_values(row.get("genres")),
        "tags": split_values(row.get("tags_normalized") or row.get("tags")),
        "preview_url": clean(row.get("deezer_preview_url") or row.get("preview_url")) or None,
        "spotify_url": clean(row.get("spotify_url")) or None,
        "payload_version": PAYLOAD_VERSION,
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def convert_csv(input_path: Path, output_path: Path) -> list[dict[str, object]]:
    with input_path.open(newline="", encoding="utf-8") as file:
        records = [build_song_record(row) for row in csv.DictReader(file)]
    write_jsonl(output_path, records)
    return records


def select_balanced_mock(
    records: Sequence[dict[str, object]],
    *,
    per_mood: int = 5,
    seed: int = 42,
) -> list[dict[str, object]]:
    by_mood: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        moods = record.get("mood") or []
        if moods and moods[0] in CANONICAL_MOODS:
            by_mood[str(moods[0])].append(record)

    randomizer = random.Random(seed)
    selected = []
    for mood in CANONICAL_MOODS:
        candidates = by_mood[mood]
        if len(candidates) < per_mood:
            raise ValueError(f"not enough {mood!r} records: {len(candidates)} < {per_mood}")
        selected.extend(randomizer.sample(candidates, per_mood))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("spocrawl/data/processed/spotify_audio_training.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/spotify_songs.jsonl"))
    parser.add_argument("--mock-output", type=Path, default=Path("data/mock_songs.jsonl"))
    parser.add_argument("--mock-per-mood", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = convert_csv(args.input, args.output)
    mock_records = select_balanced_mock(
        records,
        per_mood=args.mock_per_mood,
        seed=args.seed,
    )
    write_jsonl(args.mock_output, mock_records)
    print(f"Wrote {len(records)} songs to {args.output}")
    print(f"Wrote {len(mock_records)} balanced mock songs to {args.mock_output}")


if __name__ == "__main__":
    main()
