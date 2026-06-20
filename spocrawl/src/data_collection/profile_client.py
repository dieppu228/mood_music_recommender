import time
from datetime import datetime, timezone
from typing import Any

import spotipy

from src.data_collection.rate_limit import spotify_call_with_one_retry


class SpotifyProfileClient:
    SAVED_TRACKS_PAGE = 50
    PLAYLISTS_PAGE = 50
    PLAYLIST_ITEMS_PAGE = 100
    REQUEST_DELAY = 0.01

    def __init__(self, spotify: spotipy.Spotify) -> None:
        self._client = spotify

    def get_current_user(self) -> dict[str, Any]:
        return spotify_call_with_one_retry(self._client.current_user)

    def iter_saved_tracks(self):
        offset = 0
        while True:
            page = spotify_call_with_one_retry(
                self._client.current_user_saved_tracks,
                limit=self.SAVED_TRACKS_PAGE,
                offset=offset,
            )
            items = page.get("items", [])
            if not items:
                break
            yield from items
            offset += len(items)
            if page.get("next") is None:
                break
            time.sleep(self.REQUEST_DELAY)

    def iter_playlists(self):
        offset = 0
        while True:
            page = spotify_call_with_one_retry(
                self._client.current_user_playlists,
                limit=self.PLAYLISTS_PAGE,
                offset=offset,
            )
            items = page.get("items", [])
            if not items:
                break
            yield from items
            offset += len(items)
            if page.get("next") is None:
                break
            time.sleep(self.REQUEST_DELAY)

    def iter_playlist_tracks(self, playlist_id: str):
        offset = 0
        while True:
            page = spotify_call_with_one_retry(
                self._client.playlist_items,
                playlist_id,
                limit=self.PLAYLIST_ITEMS_PAGE,
                offset=offset,
                additional_types=("track",),
            )
            items = page.get("items", [])
            if not items:
                break
            yield from items
            offset += len(items)
            if page.get("next") is None:
                break
            time.sleep(self.REQUEST_DELAY)

    def iter_saved_albums(self):
        offset = 0
        while True:
            page = spotify_call_with_one_retry(
                self._client.current_user_saved_albums,
                limit=50,
                offset=offset,
            )
            items = page.get("items", [])
            if not items:
                break
            yield from items
            offset += len(items)
            if page.get("next") is None:
                break
            time.sleep(self.REQUEST_DELAY)

    def iter_album_tracks(self, album_id: str):
        offset = 0
        while True:
            page = spotify_call_with_one_retry(
                self._client.album_tracks,
                album_id,
                limit=50,
                offset=offset,
            )
            items = page.get("items", [])
            if not items:
                break
            yield from items
            offset += len(items)
            if page.get("next") is None:
                break
            time.sleep(self.REQUEST_DELAY)

    def get_top_tracks(self, time_range: str = "medium_term") -> list[dict[str, Any]]:
        page = spotify_call_with_one_retry(
            self._client.current_user_top_tracks,
            limit=50,
            time_range=time_range,
        )
        return page.get("items", [])

    def iter_followed_artists(self):
        after = None
        while True:
            page = spotify_call_with_one_retry(
                self._client.current_user_followed_artists,
                limit=50,
                after=after,
            )
            artists = page.get("artists", {})
            items = artists.get("items", [])
            if not items:
                break
            yield from items
            after = artists.get("cursors", {}).get("after")
            if not after:
                break
            time.sleep(self.REQUEST_DELAY)

    def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> list[dict[str, Any]]:
        page = spotify_call_with_one_retry(
            self._client.artist_top_tracks,
            artist_id,
            country=market,
        )
        return page.get("tracks", [])

    def iter_recently_played(self, max_items: int = 1000):
        collected = 0
        before = None
        while collected < max_items:
            kwargs: dict[str, Any] = {"limit": 50}
            if before:
                kwargs["before"] = before
            page = spotify_call_with_one_retry(
                self._client.current_user_recently_played,
                **kwargs,
            )
            items = page.get("items", [])
            if not items:
                break
            for item in items:
                yield item
                collected += 1
                if collected >= max_items:
                    return
            played_at = items[-1]["played_at"]
            played_dt = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
            before = int(played_dt.astimezone(timezone.utc).timestamp() * 1000)
            time.sleep(self.REQUEST_DELAY)
