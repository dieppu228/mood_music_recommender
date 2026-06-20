import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings
from src.data_collection.client import SpotifyClient
from src.data_collection.exceptions import SpotifyRateLimitError
from spotipy.exceptions import SpotifyException

from src.data_collection.queries import MOOD_PLAYLIST_QUERIES, MOOD_SEARCH_QUERIES
from src.data_collection.rate_limit import is_forbidden

AUDIO_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


def _safe_print(message: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))


class SpotifyCrawler:
    def __init__(self, client: SpotifyClient) -> None:
        self._client = client
        self.rate_limited = False

    def crawl_by_mood_playlists(
        self,
        playlists_per_query: int = 5,
        mood_queries: dict[str, list[str]] | None = None,
        exclude_ids: set[str] | None = None,
        target_new_tracks: int | None = None,
        checkpoint_path: Path | None = None,
        crawled_playlist_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        mood_queries = mood_queries or MOOD_PLAYLIST_QUERIES
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set(exclude_ids or [])
        seen_playlists: set[str] = set(crawled_playlist_ids or [])

        if checkpoint_path and checkpoint_path.exists():
            cached = pd.read_csv(checkpoint_path)
            if not cached.empty:
                records = cached.to_dict(orient="records")
                seen_ids.update(cached["track_id"].dropna().astype(str))
                if "playlist_id" in cached.columns:
                    seen_playlists.update(
                        cached["playlist_id"].dropna().astype(str).unique()
                    )
                print(f"  Resumed playlist cache with {len(records)} tracks")

        pairs = [
            (mood, query)
            for mood, queries in mood_queries.items()
            for query in queries
        ]
        total = len(pairs)

        for index, (mood, query) in enumerate(pairs, start=1):
            if target_new_tracks is not None and len(records) >= target_new_tracks:
                print(f"  Reached target of {target_new_tracks} new tracks")
                break

            print(f"  [{index}/{total}] mood={mood!r} search playlists: {query!r}")
            try:
                playlists = self._client.search_all_playlists(
                    query, max_playlists=playlists_per_query
                )
            except SpotifyRateLimitError as exc:
                print(f"  Rate limit on playlist search {query!r}: {exc}")
                self.rate_limited = True
                break

            query_added = 0
            for playlist in playlists:
                if not playlist:
                    continue
                if target_new_tracks is not None and len(records) >= target_new_tracks:
                    break

                playlist_id = playlist.get("id")
                if not playlist_id or playlist_id in seen_playlists:
                    continue
                seen_playlists.add(playlist_id)

                playlist_name = playlist.get("name", "Unknown")
                owner = playlist.get("owner", {}).get("display_name", "")
                track_total = playlist.get("tracks", {}).get("total", "?")
                _safe_print(
                    f"       playlist {playlist_name!r} "
                    f"({track_total} tracks, owner={owner})"
                )

                playlist_added = 0
                try:
                    for item in self._client.iter_playlist_items(playlist_id):
                        if target_new_tracks is not None and len(records) >= target_new_tracks:
                            break
                        record = self._playlist_item_to_record(
                            item,
                            mood=mood,
                            search_query=query,
                            playlist_id=playlist_id,
                            playlist_name=playlist_name,
                        )
                        if record and record["track_id"] not in seen_ids:
                            seen_ids.add(record["track_id"])
                            records.append(record)
                            playlist_added += 1
                            query_added += 1
                except SpotifyRateLimitError as exc:
                    print(f"  Rate limit on playlist {playlist_name!r}: {exc}")
                    self.rate_limited = True
                    break
                except SpotifyException as exc:
                    if is_forbidden(exc) or exc.http_status in (401, 403):
                        print(f"       -> skipped (no access, {exc.http_status})")
                        continue
                    raise

                print(f"       -> +{playlist_added} tracks from playlist")

                if checkpoint_path and records:
                    pd.DataFrame(records).to_csv(checkpoint_path, index=False)

                if self.rate_limited:
                    break

            print(
                f"       query total: +{query_added} new "
                f"(dataset: {len(records)}, unique ids: {len(seen_ids)})"
            )

        return pd.DataFrame(records)

    def crawl_by_mood_queries(
        self,
        limit_per_query: int = 50,
        mood_queries: dict[str, list[str]] | None = None,
        exclude_ids: set[str] | None = None,
        target_new_tracks: int | None = None,
        checkpoint_path: Path | None = None,
        start_query_index: int = 0,
    ) -> pd.DataFrame:
        mood_queries = mood_queries or MOOD_SEARCH_QUERIES
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set(exclude_ids or [])

        if checkpoint_path and checkpoint_path.exists():
            cached = pd.read_csv(checkpoint_path)
            if not cached.empty:
                records = cached.to_dict(orient="records")
                seen_ids.update(cached["track_id"].dropna().astype(str))
                print(f"  Resumed search cache with {len(records)} tracks")

        pairs = [
            (mood, query)
            for mood, queries in mood_queries.items()
            for query in queries
        ]
        total = len(pairs)

        for index, (mood, query) in enumerate(pairs, start=1):
            if index <= start_query_index:
                continue
            if target_new_tracks is not None and len(records) >= target_new_tracks:
                print(f"  Reached target of {target_new_tracks} new search tracks")
                break

            print(f"  [{index}/{total}] mood={mood!r} query={query!r}")
            try:
                tracks = self._client.search_all_tracks(query, max_results=limit_per_query)
            except SpotifyRateLimitError as exc:
                print(f"  Rate limit hit on query {query!r}: {exc}")
                self.rate_limited = True
                break

            added = 0
            for track in tracks:
                if target_new_tracks is not None and len(records) >= target_new_tracks:
                    break
                track_id = track.get("id")
                if not track_id or track_id in seen_ids:
                    continue
                seen_ids.add(track_id)
                record = self._extract_track_record(track)
                record["mood_label"] = mood
                record["search_query"] = query
                record["tags"] = f"{mood},{query}"
                record["source_type"] = "search"
                record["source_id"] = query
                record["source_name"] = query
                records.append(record)
                added += 1

            if checkpoint_path and records:
                pd.DataFrame(records).to_csv(checkpoint_path, index=False)

            print(
                f"       -> {added} new search tracks "
                f"(search total: {len(records)}, combined unique: {len(seen_ids)})"
            )

        return pd.DataFrame(records)

    def crawl_by_queries(
        self,
        queries: list[str],
        limit_per_query: int = 50,
    ) -> pd.DataFrame:
        mood_queries = {"unknown": queries}
        df = self.crawl_by_mood_queries(limit_per_query, mood_queries)
        if "mood_label" in df.columns:
            df = df.drop(columns=["mood_label"])
        return df

    def enrich_with_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        track_ids = df["track_id"].dropna().astype(str).unique().tolist()
        if not track_ids:
            return df

        print(f"  Fetching audio features for {len(track_ids)} tracks...")
        try:
            feature_rows = self._client.get_audio_features_batch(track_ids)
        except Exception as exc:
            print(f"  Warning: audio features unavailable ({exc})")
            return df

        if not feature_rows:
            print("  Warning: no audio features returned (API may restrict new apps)")
            return df

        features_df = pd.DataFrame(feature_rows)
        keep_cols = ["id"] + [c for c in AUDIO_FEATURE_COLUMNS if c in features_df.columns]
        features_df = features_df[keep_cols]
        merged = df.merge(features_df, left_on="track_id", right_on="id", how="left")
        if "id" in merged.columns:
            merged = merged.drop(columns=["id"])
        print(f"  Audio features attached for {merged[AUDIO_FEATURE_COLUMNS[0]].notna().sum()} tracks")
        return merged

    def enrich_with_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        if "artist_ids" not in df.columns:
            return df

        artist_ids: list[str] = []
        for value in df["artist_ids"].dropna():
            ids = [part.strip() for part in str(value).split("|") if part.strip()]
            if ids:
                artist_ids.append(ids[0])

        unique_ids = list(dict.fromkeys(artist_ids))
        if not unique_ids:
            return df

        print(f"  Fetching genres for {len(unique_ids)} artists...")
        try:
            artists = self._client.get_artists_batch(unique_ids)
        except Exception as exc:
            print(f"  Warning: artist genres unavailable ({exc})")
            return df

        genre_map = {
            artist["id"]: ", ".join(artist.get("genres", []))
            for artist in artists
            if artist.get("id")
        }

        def primary_genres(artist_id_str: str) -> str:
            if not isinstance(artist_id_str, str):
                return ""
            first_id = artist_id_str.split("|")[0].strip()
            return genre_map.get(first_id, "")

        enriched = df.copy()
        enriched["genres"] = enriched["artist_ids"].apply(primary_genres)
        with_genres = enriched["genres"].astype(bool).sum()
        print(f"  Genres attached for {with_genres} tracks")
        return enriched

    def save_raw(self, df: pd.DataFrame, filename: str) -> Path:
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = settings.raw_data_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    def load_raw(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(settings.raw_data_dir / filename)

    def _playlist_item_to_record(
        self,
        item: dict[str, Any],
        mood: str,
        search_query: str,
        playlist_id: str,
        playlist_name: str,
    ) -> dict[str, Any] | None:
        track = item.get("track") or item.get("item")
        if not track or track.get("is_local") or track.get("type") != "track":
            return None

        record = self._extract_track_record(track)
        album = track.get("album", {})
        external_ids = track.get("external_ids", {}) or {}
        record.update(
            {
                "album_id": album.get("id"),
                "album_type": album.get("album_type"),
                "explicit": track.get("explicit"),
                "disc_number": track.get("disc_number"),
                "track_number": track.get("track_number"),
                "isrc": external_ids.get("isrc"),
                "added_at": item.get("added_at"),
                "mood_label": mood,
                "search_query": search_query,
                "tags": f"{mood},{search_query},{playlist_name}",
                "playlist_id": playlist_id,
                "playlist_name": playlist_name,
                "source_type": "discovered_playlist",
                "source_id": playlist_id,
                "source_name": playlist_name,
            }
        )
        return record

    @staticmethod
    def _extract_track_record(track: dict[str, Any]) -> dict[str, Any]:
        artist_objs = track.get("artists", [])
        artists = ", ".join(a["name"] for a in artist_objs)
        artist_ids = "|".join(a["id"] for a in artist_objs if a.get("id"))
        album = track.get("album", {})
        return {
            "track_id": track.get("id"),
            "track_name": track.get("name"),
            "artists": artists,
            "artist_ids": artist_ids,
            "album": album.get("name"),
            "release_date": album.get("release_date"),
            "popularity": track.get("popularity"),
            "duration_ms": track.get("duration_ms"),
            "preview_url": track.get("preview_url"),
            "spotify_url": track.get("external_urls", {}).get("spotify"),
        }
