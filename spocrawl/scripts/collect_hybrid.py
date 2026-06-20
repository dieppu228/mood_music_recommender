"""Combine profile library data with mood-based playlist crawl to reach 10k+ tracks."""

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
from src.data_collection.merger import merge_datasets
from src.preprocessing.pipeline import PreprocessingPipeline

PROFILE_RAW = "spotify_profile_tracks.csv"
PROFILE_PROCESSED = "spotify_profile_tracks.csv"
HYBRID_RAW = "spotify_hybrid_tracks.csv"
HYBRID_PROCESSED = "spotify_hybrid_tracks.csv"
PLAYLIST_CACHE = "spotify_playlist_supplement.csv"
SEARCH_CACHE = "spotify_search_supplement.csv"
LEGACY_SEARCH = "spotify_tracks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge profile tracks with mood playlist crawl for 10k+ dataset"
    )
    parser.add_argument("--target", type=int, default=10_000, help="Total unique tracks")
    parser.add_argument(
        "--playlists-per-query",
        type=int,
        default=5,
        help="Playlists to fetch per mood search query (default: 5)",
    )
    parser.add_argument(
        "--limit-per-query",
        type=int,
        default=200,
        help="[track mode] Max tracks per search query",
    )
    parser.add_argument(
        "--mode",
        choices=["playlist", "track"],
        default="playlist",
        help="Crawl mode: search playlists (default) or individual tracks",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Only merge existing profile + supplement cache files",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip audio features enrichment",
    )
    parser.add_argument(
        "--skip-genres",
        action="store_true",
        help="Skip artist genre enrichment",
    )
    parser.add_argument(
        "--gentle",
        action="store_true",
        help="Slower requests to avoid rate limits",
    )
    parser.add_argument(
        "--wait-if-limited",
        action="store_true",
        help="Wait for Retry-After (up to 2h) when search is rate-limited",
    )
    return parser.parse_args()


def load_profile_df() -> pd.DataFrame:
    processed_path = settings.processed_data_dir / PROFILE_PROCESSED
    raw_path = settings.raw_data_dir / PROFILE_RAW

    if processed_path.exists():
        print(f"Loading profile data: {processed_path}")
        return pd.read_csv(processed_path)
    if raw_path.exists():
        print(f"Loading profile data: {raw_path}")
        return pd.read_csv(raw_path)

    print("Error: no profile dataset found. Run scripts/collect_profile.py first.")
    sys.exit(1)


def load_supplement_df(profile_ids: set[str], mode: str) -> pd.DataFrame:
    playlist_cache = settings.raw_data_dir / PLAYLIST_CACHE
    search_cache = settings.raw_data_dir / SEARCH_CACHE
    legacy_path = settings.raw_data_dir / LEGACY_SEARCH

    if mode == "playlist" and playlist_cache.exists():
        print(f"Loading cached playlist supplement: {playlist_cache}")
        return pd.read_csv(playlist_cache)
    if search_cache.exists():
        print(f"Loading cached search supplement: {search_cache}")
        return pd.read_csv(search_cache)
    if legacy_path.exists():
        print(f"Loading legacy mood search data: {legacy_path}")
        df = pd.read_csv(legacy_path)
        return df[~df["track_id"].astype(str).isin(profile_ids)]
    return pd.DataFrame()


def main() -> None:
    args = parse_args()

    if not settings.spotify_client_id or not settings.spotify_client_secret:
        print("Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env")
        sys.exit(1)

    profile_df = load_profile_df()
    profile_ids = set(profile_df["track_id"].dropna().astype(str))
    print(f"Profile tracks: {len(profile_df)} unique")

    gap = max(0, args.target - len(profile_ids))
    print(f"Target: {args.target} | Need {gap} more tracks")
    print(f"Crawl mode: {args.mode}")

    supplement_df = pd.DataFrame()
    cache_path = (
        settings.raw_data_dir / PLAYLIST_CACHE
        if args.mode == "playlist"
        else settings.raw_data_dir / SEARCH_CACHE
    )

    page_delay = 0.5 if args.gentle else 0.3
    query_delay = 2.0 if args.gentle else 1.0
    if args.gentle:
        print("Gentle mode: slower requests")

    if gap > 0 and not args.skip_search:
        auth = SpotifyAuth()
        if args.mode == "playlist":
            print("\nConnecting to Spotify (user OAuth — needed to read playlist tracks)...")
            spotify = auth.get_user_client()
        else:
            print("\nConnecting to Spotify (client credentials)...")
            spotify = auth.get_app_client()
        client = SpotifyClient(
            spotify,
            search_page_delay=page_delay,
            query_delay=query_delay,
        )

        print("Checking search API quota...")
        if not client.probe_search(wait=args.wait_if_limited, search_type="playlist"):
            print("Search blocked — merging cached data only. Use --wait-if-limited.")
            args.skip_search = True
        else:
            print("Search API available.")

    if gap > 0 and not args.skip_search:
        crawler = SpotifyCrawler(client)

        if args.mode == "playlist":
            print(
                f"Crawling via playlists "
                f"(up to {gap} new tracks, {args.playlists_per_query} playlists/query)..."
            )
            supplement_df = crawler.crawl_by_mood_playlists(
                playlists_per_query=args.playlists_per_query,
                exclude_ids=profile_ids,
                target_new_tracks=gap,
                checkpoint_path=cache_path,
            )
        else:
            limit = min(args.limit_per_query, 80) if args.gentle else args.limit_per_query
            print(f"Crawling individual tracks (up to {gap}, {limit}/query)...")
            supplement_df = crawler.crawl_by_mood_queries(
                limit_per_query=limit,
                exclude_ids=profile_ids,
                target_new_tracks=gap,
                checkpoint_path=cache_path,
            )

        if crawler.rate_limited:
            print("Crawl stopped early due to rate limit — merging partial results")

        if supplement_df.empty:
            print("Warning: crawl returned no new tracks")
        else:
            if not args.skip_features:
                supplement_df = crawler.enrich_with_audio_features(supplement_df)
            if not args.skip_genres:
                supplement_df = crawler.enrich_with_genres(supplement_df)
            supplement_df.to_csv(cache_path, index=False)
            print(f"Saved supplement: {cache_path} ({len(supplement_df)} tracks)")
    else:
        supplement_df = load_supplement_df(profile_ids, args.mode)

    print("\nMerging profile + supplement datasets...")
    hybrid_df = merge_datasets(profile_df, supplement_df)
    if len(hybrid_df) > args.target:
        hybrid_df = hybrid_df.head(args.target)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = settings.raw_data_dir / HYBRID_RAW
    hybrid_df.to_csv(raw_path, index=False)
    print(f"Saved raw hybrid data: {raw_path} ({len(hybrid_df)} tracks)")

    pipeline = PreprocessingPipeline()
    processed_df = pipeline.run(hybrid_df)
    processed_path = pipeline.save(processed_df, HYBRID_PROCESSED)
    print(f"Saved processed hybrid data: {processed_path} ({len(processed_df)} tracks)")

    if "data_origin" in processed_df.columns:
        print("\nTracks by origin:")
        print(processed_df["data_origin"].value_counts().to_string())

    if "mood_label" in processed_df.columns:
        search_part = processed_df[processed_df["data_origin"] == "search"]
        if not search_part.empty:
            print("\nSupplement tracks per mood:")
            print(search_part["mood_label"].value_counts().to_string())

    if len(processed_df) >= args.target:
        print(f"\nTarget reached: {len(processed_df)} >= {args.target}")
    else:
        print(f"\nBelow target: {len(processed_df)} < {args.target}")

    print("\nDone.")


if __name__ == "__main__":
    main()
