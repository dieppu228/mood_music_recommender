class SpotifyRateLimitError(Exception):
    def __init__(self, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        message = "Spotify API rate limit reached"
        if retry_after is not None:
            message = f"{message} (retry after {retry_after}s)"
        super().__init__(message)
