import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings
from src.data_collection.crawler import SpotifyCrawler
from spotipy.exceptions import SpotifyException

from src.data_collection.exceptions import SpotifyRateLimitError
from src.data_collection.profile_client import SpotifyProfileClient
from src.data_collection.rate_limit import is_forbidden


def _safe_print(message: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe)


class SpotifyProfileCrawler:
    def __init__(self, profile_client: SpotifyProfileClient) -> None:
        self._client = profile_client
        self._rate_limited = False

    @property
    def rate_limited(self) -> bool:
        return self._rate_limited

    def crawl_profile(
        self,
        target_tracks: int = 10_000,
        checkpoint_every: int = 500,
        output_filename: str = "spotify_profile_tracks.csv",
    ) -> pd.DataFrame:
        user = self._client.get_current_user()
        display_name = user.get("display_name") or user.get("id", "unknown")
        _safe_print(f"Logged in as: {display_name}")

        output_path = settings.raw_data_dir / output_filename
        state_path = settings.raw_data_dir / "spotify_profile_state.json"
        records, seen_ids = self._load_checkpoint(output_path)
        state = self._load_state(state_path)
        print(f"Starting with {len(records)} tracks from checkpoint")

        if not state.get("liked_done"):
            self._crawl_liked_songs(
                records, seen_ids, target_tracks, checkpoint_every, output_path, state_path
            )
            state["liked_done"] = True
            self._save_state(state_path, state)
        else:
            print("Skipping liked songs (already crawled in checkpoint)")

        if not self._rate_limited and len(seen_ids) < target_tracks:
            self._crawl_playlists(
                records,
                seen_ids,
                target_tracks,
                checkpoint_every,
                output_path,
                state_path,
                state,
                user.get("id"),
            )

        if not self._rate_limited and len(seen_ids) < target_tracks:
            self._crawl_saved_albums(
                records, seen_ids, target_tracks, checkpoint_every, output_path
            )

        if not self._rate_limited and len(seen_ids) < target_tracks:
            self._crawl_top_tracks(records, seen_ids, target_tracks, output_path)

        if not self._rate_limited and len(seen_ids) < target_tracks:
            self._crawl_recently_played(
                records, seen_ids, target_tracks, checkpoint_every, output_path
            )

        if not self._rate_limited and len(seen_ids) < target_tracks:
            self._crawl_followed_artists(
                records, seen_ids, target_tracks, checkpoint_every, output_path
            )

        df = pd.DataFrame(records)
        if not df.empty:
            self._save(df, output_path)
        return df

    def _crawl_liked_songs(
        self,
        records: list[dict[str, Any]],
        seen_ids: set[str],
        target_tracks: int,
        checkpoint_every: int,
        output_path: Path,
        state_path: Path,
    ) -> None:
        print("\n[1/2] Crawling liked songs...")
        added = 0
        try:
            for item in self._client.iter_saved_tracks():
                if len(seen_ids) >= target_tracks:
                    print(f"  Reached target of {target_tracks} tracks")
                    return

                record = self._item_to_record(
                    item,
                    source_type="liked",
                    source_id="liked",
                    source_name="Liked Songs",
                )
                if record and record["track_id"] not in seen_ids:
                    seen_ids.add(record["track_id"])
                    records.append(record)
                    added += 1
                    if added % checkpoint_every == 0:
                        self._checkpoint(records, output_path, "liked songs")
        except SpotifyRateLimitError as exc:
            self._rate_limited = True
            print(f"  Rate limit hit during liked songs crawl: {exc}")

        print(f"  Liked songs done: +{added} new (total unique: {len(seen_ids)})")

    def _crawl_playlists(
        self,
        records: list[dict[str, Any]],
        seen_ids: set[str],
        target_tracks: int,
        checkpoint_every: int,
        output_path: Path,
        state_path: Path,
        state: dict[str, Any],
        user_id: str | None,
    ) -> None:
        print("\n[2/2] Crawling playlists...")
        playlists = []
        try:
            playlists = list(self._client.iter_playlists())
        except SpotifyRateLimitError as exc:
            self._rate_limited = True
            print(f"  Rate limit hit while listing playlists: {exc}")
            return

        print(f"  Found {len(playlists)} playlists")
        added_total = 0
        start_index = int(state.get("last_playlist_index", 0))

        for index, playlist in enumerate(playlists, start=1):
            if index <= start_index:
                continue
            if self._rate_limited or len(seen_ids) >= target_tracks:
                break

            playlist_id = playlist.get("id")
            playlist_name = playlist.get("name", "Unknown")
            owner_id = playlist.get("owner", {}).get("id")
            is_owner = owner_id == user_id
            track_count = playlist.get("tracks", {}).get("total", "?")
            _safe_print(
                f"  [{index}/{len(playlists)}] {playlist_name!r} "
                f"({track_count} tracks, owner={'me' if is_owner else owner_id})"
            )

            added = 0
            try:
                for item in self._client.iter_playlist_tracks(playlist_id):
                    if len(seen_ids) >= target_tracks:
                        print(f"  Reached target of {target_tracks} tracks")
                        return

                    record = self._item_to_record(
                        item,
                        source_type="playlist",
                        source_id=playlist_id,
                        source_name=playlist_name,
                    )
                    if record and record["track_id"] not in seen_ids:
                        seen_ids.add(record["track_id"])
                        records.append(record)
                        added += 1
                        added_total += 1
                        if added_total % checkpoint_every == 0:
                            self._checkpoint(records, output_path, f"playlist: {playlist_name}")
            except SpotifyRateLimitError as exc:
                self._rate_limited = True
                _safe_print(f"  Rate limit hit on playlist {playlist_name!r}: {exc}")
                break
            except SpotifyException as exc:
                if is_forbidden(exc):
                    _safe_print(f"       -> skipped (no access to this playlist)")
                    state["last_playlist_index"] = index
                    self._save_state(state_path, state)
                    continue
                raise

            print(f"       -> +{added} new (total unique: {len(seen_ids)})")
            state["last_playlist_index"] = index
            self._save_state(state_path, state)

        print(f"  Playlists done: +{added_total} new from this phase")

    def _crawl_saved_albums(
        self,
        records: list[dict[str, Any]],
        seen_ids: set[str],
        target_tracks: int,
        checkpoint_every: int,
        output_path: Path,
    ) -> None:
        print("\n[3/4] Crawling saved albums...")
        added_total = 0
        try:
            for item in self._client.iter_saved_albums():
                if self._rate_limited or len(seen_ids) >= target_tracks:
                    break
                album = item.get("album", {})
                album_id = album.get("id")
                album_name = album.get("name", "Unknown")
                try:
                    for track in self._client.iter_album_tracks(album_id):
                        if len(seen_ids) >= target_tracks:
                            return
                        record = self._track_to_record(
                            track,
                            album=album,
                            source_type="saved_album",
                            source_id=album_id,
                            source_name=album_name,
                            added_at=item.get("added_at"),
                        )
                        if record and record["track_id"] not in seen_ids:
                            seen_ids.add(record["track_id"])
                            records.append(record)
                            added_total += 1
                            if added_total % checkpoint_every == 0:
                                self._checkpoint(records, output_path, f"album: {album_name}")
                except SpotifyRateLimitError as exc:
                    self._rate_limited = True
                    print(f"  Rate limit hit on album {album_name!r}: {exc}")
                    break
                except SpotifyException as exc:
                    if is_forbidden(exc):
                        continue
                    raise
        except SpotifyRateLimitError as exc:
            self._rate_limited = True
            print(f"  Rate limit hit listing saved albums: {exc}")

        print(f"  Saved albums done: +{added_total} new (total unique: {len(seen_ids)})")

    def _crawl_top_tracks(
        self,
        records: list[dict[str, Any]],
        seen_ids: set[str],
        target_tracks: int,
        output_path: Path,
    ) -> None:
        print("\n[4/4] Crawling top tracks...")
        added_total = 0
        for time_range in ("short_term", "medium_term", "long_term"):
            if self._rate_limited or len(seen_ids) >= target_tracks:
                break
            try:
                tracks = self._client.get_top_tracks(time_range)
            except SpotifyRateLimitError as exc:
                self._rate_limited = True
                print(f"  Rate limit hit on top tracks ({time_range}): {exc}")
                break
            except SpotifyException as exc:
                if is_forbidden(exc):
                    print(f"  Top tracks unavailable for {time_range} (403)")
                    continue
                raise

            for track in tracks:
                record = self._track_to_record(
                    track,
                    source_type="top_tracks",
                    source_id=time_range,
                    source_name=f"Top Tracks ({time_range})",
                )
                if record and record["track_id"] not in seen_ids:
                    seen_ids.add(record["track_id"])
                    records.append(record)
                    added_total += 1

        print(f"  Top tracks done: +{added_total} new (total unique: {len(seen_ids)})")

    def _crawl_recently_played(
        self,
        records: list[dict[str, Any]],
        seen_ids: set[str],
        target_tracks: int,
        checkpoint_every: int,
        output_path: Path,
    ) -> None:
        print("\n[5/6] Crawling recently played...")
        added_total = 0
        try:
            for item in self._client.iter_recently_played(max_items=2000):
                if self._rate_limited or len(seen_ids) >= target_tracks:
                    break
                track = item.get("track")
                if not track:
                    continue
                record = self._track_to_record(
                    track,
                    source_type="recently_played",
                    source_id="recent",
                    source_name="Recently Played",
                    added_at=item.get("played_at"),
                )
                if record and record["track_id"] not in seen_ids:
                    seen_ids.add(record["track_id"])
                    records.append(record)
                    added_total += 1
                    if added_total % checkpoint_every == 0:
                        self._checkpoint(records, output_path, "recently played")
        except SpotifyRateLimitError as exc:
            self._rate_limited = True
            print(f"  Rate limit hit on recently played: {exc}")

        print(f"  Recently played done: +{added_total} new (total unique: {len(seen_ids)})")

    def _crawl_followed_artists(
        self,
        records: list[dict[str, Any]],
        seen_ids: set[str],
        target_tracks: int,
        checkpoint_every: int,
        output_path: Path,
    ) -> None:
        print("\n[6/6] Crawling followed artists top tracks...")
        added_total = 0
        artist_count = 0
        try:
            for artist in self._client.iter_followed_artists():
                if self._rate_limited or len(seen_ids) >= target_tracks:
                    break
                artist_id = artist.get("id")
                artist_name = artist.get("name", "Unknown")
                artist_count += 1
                if artist_count % 25 == 0:
                    print(f"  Artists processed: {artist_count} (total unique tracks: {len(seen_ids)})")

                try:
                    tracks = self._client.get_artist_top_tracks(artist_id)
                except SpotifyRateLimitError as exc:
                    self._rate_limited = True
                    print(f"  Rate limit hit on artist {artist_name!r}: {exc}")
                    break
                except SpotifyException as exc:
                    if is_forbidden(exc):
                        continue
                    raise

                for track in tracks:
                    if len(seen_ids) >= target_tracks:
                        return
                    record = self._track_to_record(
                        track,
                        source_type="followed_artist",
                        source_id=artist_id,
                        source_name=artist_name,
                    )
                    if record and record["track_id"] not in seen_ids:
                        seen_ids.add(record["track_id"])
                        records.append(record)
                        added_total += 1
                        if added_total % checkpoint_every == 0:
                            self._checkpoint(records, output_path, f"artist: {artist_name}")
        except SpotifyRateLimitError as exc:
            self._rate_limited = True
            print(f"  Rate limit hit listing followed artists: {exc}")

        print(
            f"  Followed artists done: +{added_total} new from {artist_count} artists "
            f"(total unique: {len(seen_ids)})"
        )

    def _track_to_record(
        self,
        track: dict[str, Any],
        source_type: str,
        source_id: str | None,
        source_name: str,
        album: dict[str, Any] | None = None,
        added_at: str | None = None,
    ) -> dict[str, Any] | None:
        if not track or track.get("is_local") or track.get("type") != "track":
            return None
        album = album or track.get("album", {})
        merged_track = dict(track)
        if album and not merged_track.get("album"):
            merged_track["album"] = album
        record = SpotifyCrawler._extract_track_record(merged_track)
        external_ids = track.get("external_ids", {}) or {}
        record.update(
            {
                "album_id": album.get("id"),
                "album_type": album.get("album_type"),
                "explicit": track.get("explicit"),
                "disc_number": track.get("disc_number"),
                "track_number": track.get("track_number"),
                "isrc": external_ids.get("isrc"),
                "added_at": added_at,
                "source_type": source_type,
                "source_id": source_id,
                "source_name": source_name,
            }
        )
        return record

    def _item_to_record(
        self,
        item: dict[str, Any],
        source_type: str,
        source_id: str | None,
        source_name: str,
    ) -> dict[str, Any] | None:
        track = item.get("track") or item.get("item")
        if not track or track.get("is_local") or track.get("type") != "track":
            return None

        return self._track_to_record(
            track,
            source_type=source_type,
            source_id=source_id,
            source_name=source_name,
            added_at=item.get("added_at"),
        )

    def _checkpoint(self, records: list[dict[str, Any]], output_path: Path, phase: str) -> None:
        df = pd.DataFrame(records)
        self._save(df, output_path)
        _safe_print(f"  Checkpoint saved ({len(df)} tracks) - {phase}")

    @staticmethod
    def _save(df: pd.DataFrame, output_path: Path) -> None:
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    @staticmethod
    def _load_state(state_path: Path) -> dict[str, Any]:
        if not state_path.exists():
            return {}
        return json.loads(state_path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_state(state_path: Path, state: dict[str, Any]) -> None:
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @staticmethod
    def _load_checkpoint(output_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
        if not output_path.exists():
            return [], set()

        df = pd.read_csv(output_path)
        if df.empty:
            return [], set()

        records = df.to_dict(orient="records")
        seen_ids = set(df["track_id"].dropna().astype(str))
        return records, seen_ids
