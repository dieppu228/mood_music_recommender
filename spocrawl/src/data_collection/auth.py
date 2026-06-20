import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from config.settings import settings

TOKEN_CACHE = settings.root_dir / ".spotify_token"


class SpotifyAuth:
    def __init__(self) -> None:
        self._user_scope = (
            "user-library-read "
            "user-top-read "
            "user-follow-read "
            "user-read-recently-played "
            "user-read-private "
            "user-read-email "
            "playlist-read-private "
            "playlist-read-collaborative"
        )

    def get_app_client(self) -> spotipy.Spotify:
        """Client-credentials flow for public catalog crawling (no browser login)."""
        auth_manager = SpotifyClientCredentials(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
        )
        return spotipy.Spotify(auth_manager=auth_manager, retries=0, status_retries=0)

    def get_oauth_manager(self) -> SpotifyOAuth:
        return SpotifyOAuth(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            redirect_uri=settings.spotify_redirect_uri,
            scope=self._user_scope,
            cache_path=str(TOKEN_CACHE),
            open_browser=True,
        )

    def get_user_client(self) -> spotipy.Spotify:
        """Authorization-code flow for user-specific endpoints."""
        return spotipy.Spotify(auth_manager=self.get_oauth_manager(), retries=0, status_retries=0)
