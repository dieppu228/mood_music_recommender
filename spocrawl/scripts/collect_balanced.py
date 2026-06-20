"""Bổ sung crawl search theo mood thiếu để cân bằng class."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.data_collection.auth import SpotifyAuth
from src.data_collection.client import SpotifyClient
from src.data_collection.crawler import SpotifyCrawler
from src.data_collection.exceptions import SpotifyRateLimitError
from src.data_collection.queries import MOOD_SEARCH_QUERIES, MOOD_SUPPLEMENT_QUERIES

HYBRID_RAW = "spotify_hybrid_tracks.csv"
SUPPLEMENT_CACHE = "spotify_balanced_supplement.csv"
DEFAULT_MOODS = ("stressed", "romantic", "energetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl bổ sung mood thiếu, merge vào hybrid raw")
    parser.add_argument("--per-mood-target", type=int, default=1500, help="Số track search tối thiểu/mood")
    parser.add_argument(
        "--moods",
        type=str,
        default=",".join(DEFAULT_MOODS),
        help="Mood cần bổ sung (comma-separated)",
    )
    parser.add_argument("--limit-per-query", type=int, default=80, help="Max track mỗi query")
    parser.add_argument("--gentle", action="store_true", help="Chậm hơn, tránh rate limit")
    parser.add_argument("--wait-if-limited", action="store_true", help="Chờ Retry-After khi bị 429")
    parser.add_argument("--skip-features", action="store_true", help="Bỏ audio features")
    parser.add_argument("--skip-genres", action="store_true", help="Bỏ genres")
    return parser.parse_args()


def load_existing() -> pd.DataFrame:
    path = settings.raw_data_dir / HYBRID_RAW
    if not path.exists():
        print(f"Error: không tìm thấy {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded existing hybrid: {len(df)} tracks")
    return df


def mood_gaps(df: pd.DataFrame, moods: list[str], target: int) -> dict[str, int]:
    if "data_origin" in df.columns:
        search = df[df["data_origin"] == "search"]
    else:
        search = df

    counts = search["mood_label"].value_counts().to_dict() if "mood_label" in search.columns else {}
    gaps = {}
    for mood in moods:
        current = int(counts.get(mood, 0))
        if current < target:
            gaps[mood] = target - current
    return gaps


def build_round_robin_queries(moods: list[str]) -> list[tuple[str, str]]:
    queries_by_mood: dict[str, list[str]] = {}
    for mood in moods:
        base = list(MOOD_SEARCH_QUERIES.get(mood, []))
        extra = list(MOOD_SUPPLEMENT_QUERIES.get(mood, []))
        seen: set[str] = set()
        merged: list[str] = []
        for q in base + extra:
            if q not in seen:
                seen.add(q)
                merged.append(q)
        queries_by_mood[mood] = merged

    pairs: list[tuple[str, str]] = []
    max_len = max((len(v) for v in queries_by_mood.values()), default=0)
    for i in range(max_len):
        for mood in moods:
            qs = queries_by_mood.get(mood, [])
            if i < len(qs):
                pairs.append((mood, qs[i]))
    return pairs


def crawl_balanced(
    crawler: SpotifyCrawler,
    client: SpotifyClient,
    pairs: list[tuple[str, str]],
    gaps: dict[str, int],
    exclude_ids: set[str],
    limit_per_query: int,
    checkpoint_path: Path,
) -> pd.DataFrame:
    records: list[dict] = []
    seen_ids = set(exclude_ids)
    mood_added = {mood: 0 for mood in gaps}
    total = len(pairs)

    if checkpoint_path.exists():
        cached = pd.read_csv(checkpoint_path)
        if not cached.empty:
            records = cached.to_dict(orient="records")
            seen_ids.update(cached["track_id"].dropna().astype(str))
            if "mood_label" in cached.columns:
                for mood in gaps:
                    mood_added[mood] = int((cached["mood_label"] == mood).sum())
            print(f"  Resumed supplement cache: {len(records)} tracks")

    def gaps_remaining() -> bool:
        return any(mood_added[m] < gaps[m] for m in gaps)

    for index, (mood, query) in enumerate(pairs, start=1):
        if not gaps_remaining():
            print("  Đã đủ quota cho các mood mục tiêu")
            break
        if mood_added[mood] >= gaps[mood]:
            continue

        print(f"  [{index}/{total}] mood={mood!r} need={gaps[mood] - mood_added[mood]} query={query!r}")
        try:
            tracks = client.search_all_tracks(query, max_results=limit_per_query)
        except SpotifyRateLimitError as exc:
            print(f"  Rate limit: {exc}")
            crawler.rate_limited = True
            break

        added = 0
        for track in tracks:
            if mood_added[mood] >= gaps[mood]:
                break
            track_id = track.get("id")
            if not track_id or track_id in seen_ids:
                continue
            seen_ids.add(track_id)
            record = crawler._extract_track_record(track)
            record.update(
                {
                    "mood_label": mood,
                    "search_query": query,
                    "tags": f"{mood},{query}",
                    "source_type": "search",
                    "source_id": query,
                    "source_name": query,
                    "data_origin": "search",
                }
            )
            records.append(record)
            mood_added[mood] += 1
            added += 1

        if records:
            pd.DataFrame(records).to_csv(checkpoint_path, index=False)

        print(
            f"       -> +{added} | mood totals: "
            + ", ".join(f"{m}={mood_added[m]}/{gaps[m]}" for m in gaps)
        )

    return pd.DataFrame(records)


def merge_and_save(existing: pd.DataFrame, supplement: pd.DataFrame) -> Path:
    if supplement.empty:
        combined = existing
    else:
        combined = pd.concat([existing, supplement], ignore_index=True)
        combined = combined.drop_duplicates(subset=["track_id"], keep="first")

    path = settings.raw_data_dir / HYBRID_RAW
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    moods = [m.strip() for m in args.moods.split(",") if m.strip()]

    if not settings.spotify_client_id or not settings.spotify_client_secret:
        print("Error: thiếu SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET trong .env")
        sys.exit(1)

    existing = load_existing()
    gaps = mood_gaps(existing, moods, args.per_mood_target)

    if not gaps:
        print("Các mood mục tiêu đã đủ quota — không cần crawl thêm.")
        return

    print("\nCần bổ sung (search tracks):")
    for mood, gap in gaps.items():
        print(f"  {mood}: +{gap}")

    exclude_ids = set(existing["track_id"].dropna().astype(str))
    pairs = build_round_robin_queries(list(gaps.keys()))
    cache_path = settings.raw_data_dir / SUPPLEMENT_CACHE

    page_delay = 0.6 if args.gentle else 0.3
    query_delay = 2.5 if args.gentle else 1.0
    limit = min(args.limit_per_query, 50) if args.gentle else args.limit_per_query

    print("\nConnecting Spotify (client credentials)...")
    auth = SpotifyAuth()
    client = SpotifyClient(
        auth.get_app_client(),
        search_page_delay=page_delay,
        query_delay=query_delay,
    )

    if not client.probe_search(wait=args.wait_if_limited, search_type="track"):
        print("Search API bị chặn — thử lại sau hoặc dùng --wait-if-limited")
        sys.exit(1)

    crawler = SpotifyCrawler(client)
    print(f"\nBalanced crawl ({len(pairs)} queries, gentle={args.gentle})...")
    supplement = crawl_balanced(
        crawler,
        client,
        pairs,
        gaps,
        exclude_ids,
        limit,
        cache_path,
    )

    if not args.skip_features and not supplement.empty:
        supplement = crawler.enrich_with_audio_features(supplement)
    if not args.skip_genres and not supplement.empty:
        supplement = crawler.enrich_with_genres(supplement)

    if not supplement.empty:
        supplement.to_csv(cache_path, index=False)

    out_path = merge_and_save(existing, supplement)
    combined = pd.read_csv(out_path)
    search = combined[combined["data_origin"] == "search"] if "data_origin" in combined.columns else combined

    print(f"\nSaved: {out_path} ({len(combined)} tracks, +{len(supplement)} mới)")
    print("\nSearch mood distribution:")
    print(search["mood_label"].value_counts().to_string())

    if crawler.rate_limited:
        print("\nDừng sớm do rate limit — chạy lại script sau để resume từ checkpoint.")
    print("\nChạy lại notebook process_and_visualize.ipynb để xử lý & vẽ biểu đồ.")


if __name__ == "__main__":
    main()
