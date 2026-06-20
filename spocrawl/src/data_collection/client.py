import time
from typing import Any

import requests
import spotipy
from spotipy.exceptions import SpotifyException

from src.data_collection.exceptions import SpotifyRateLimitError
from src.data_collection.rate_limit import _extract_retry_after


class SpotifyClient:
    SEARCH_PAGE_SIZE = 10
    BATCH_SIZE = 50

    def __init__(
        self,
        spotify: spotipy.Spotify,
        search_page_delay: float = 0.3,
        query_delay: float = 1.0,
    ) -> None:
        self._client = spotify
        self.search_page_delay = search_page_delay
        self.query_delay = query_delay

    def search_tracks(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        try:
            result = self._client.search(
                q=query,
                type="track",
                limit=min(limit, self.SEARCH_PAGE_SIZE),
                offset=offset,
            )
        except SpotifyException as exc:
            if exc.http_status == 429:
                raise SpotifyRateLimitError(_extract_retry_after(exc)) from exc
            raise
        tracks = result.get("tracks", {})
        return tracks.get("items", []), tracks.get("total", 0)

    def search_all_tracks(self, query: str, max_results: int = 100) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        offset = 0
        total = None

        while len(collected) < max_results:
            page_limit = min(self.SEARCH_PAGE_SIZE, max_results - len(collected))
            items, total = self.search_tracks(query, limit=page_limit, offset=offset)
            if not items:
                break
            collected.extend(p for p in items if p)
            offset += len(items)
            if total is not None and offset >= total:
                break
            if offset >= 1000:
                break
            time.sleep(self.search_page_delay)

        time.sleep(self.query_delay)
        return collected

    def search_playlists(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        try:
            result = self._client.search(
                q=query,
                type="playlist",
                limit=min(limit, self.SEARCH_PAGE_SIZE),
                offset=offset,
            )
        except SpotifyException as exc:
            if exc.http_status == 429:
                raise SpotifyRateLimitError(_extract_retry_after(exc)) from exc
            raise
        playlists = result.get("playlists", {})
        return playlists.get("items", []), playlists.get("total", 0)

    def search_all_playlists(
        self,
        query: str,
        max_playlists: int = 10,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        offset = 0
        total = None

        while len(collected) < max_playlists:
            page_limit = min(self.SEARCH_PAGE_SIZE, max_playlists - len(collected))
            items, total = self.search_playlists(query, limit=page_limit, offset=offset)
            if not items:
                break
            collected.extend(p for p in items if p)
            offset += len(items)
            if total is not None and offset >= total:
                break
            if offset >= 1000:
                break
            time.sleep(self.search_page_delay)

        time.sleep(self.query_delay)
        return collected

    def iter_playlist_items(self, playlist_id: str):
        offset = 0
        page_size = 100
        while True:
            try:
                page = self._client.playlist_items(
                    playlist_id,
                    limit=page_size,
                    offset=offset,
                    additional_types=("track",),
                )
            except SpotifyException as exc:
                if exc.http_status == 429:
                    raise SpotifyRateLimitError(_extract_retry_after(exc)) from exc
                raise
            items = page.get("items", [])
            if not items:
                break
            yield from items
            offset += len(items)
            if page.get("next") is None:
                break
            time.sleep(self.search_page_delay)

    def probe_search(self, wait: bool = False, search_type: str = "playlist") -> bool:
        token = self._client.auth_manager.get_access_token(as_dict=False)
        response = requests.get(
            "https://api.spotify.com/v1/search",
            params={"q": "chill playlist", "type": search_type, "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if response.status_code == 200:
            return True
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 0))
            if retry_after >= 3600:
                print(f"Search API rate-limited — retry after ~{retry_after / 3600:.1f}h")
            elif retry_after > 0:
                print(f"Search API rate-limited — retry after ~{retry_after / 60:.0f}min")
            else:
                print("Search API rate-limited")
            if wait and 0 < retry_after <= 7200:
                print(f"Waiting {retry_after + 15}s for quota reset...")
                time.sleep(retry_after + 15)
                return self.probe_search(wait=False)
            return False
        response.raise_for_status()
        return False

    def get_track(self, track_id: str) -> dict[str, Any]:
        return self._client.track(track_id)

    def get_audio_features_batch(self, track_ids: list[str]) -> list[dict[str, Any]]:
        if not track_ids:
            return []

        features: list[dict[str, Any]] = []
        for i in range(0, len(track_ids), self.BATCH_SIZE):
            chunk = track_ids[i : i + self.BATCH_SIZE]
            batch = self._client.audio_features(chunk)
            features.extend(f for f in batch if f)
            time.sleep(0.1)
        return features

    def get_artists_batch(self, artist_ids: list[str]) -> list[dict[str, Any]]:
        if not artist_ids:
            return []

        artists: list[dict[str, Any]] = []
        for artist_id in artist_ids:
            artists.append(self.get_artist(artist_id))
            time.sleep(0.05)
        return artists

    def get_artist(self, artist_id: str) -> dict[str, Any]:
        return self._client.artist(artist_id)

    def get_recommendations(
        self,
        seed_tracks: list[str] | None = None,
        seed_genres: list[str] | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        result = self._client.recommendations(
            seed_tracks=seed_tracks or [],
            seed_genres=seed_genres or [],
            limit=limit,
            **kwargs,
        )
        return result.get("tracks", [])
