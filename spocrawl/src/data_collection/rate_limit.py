import time
from collections.abc import Callable
from typing import Any, TypeVar

import spotipy
from spotipy.exceptions import SpotifyException

from src.data_collection.exceptions import SpotifyRateLimitError

T = TypeVar("T")


def spotify_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        return func(*args, **kwargs)
    except SpotifyException as exc:
        if exc.http_status == 429:
            retry_after = _extract_retry_after(exc)
            raise SpotifyRateLimitError(retry_after) from exc
        raise


def is_forbidden(exc: Exception) -> bool:
    return isinstance(exc, SpotifyException) and exc.http_status == 403


MAX_RETRY_WAIT_SECONDS = 120


def spotify_call_with_one_retry(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        return spotify_call(func, *args, **kwargs)
    except SpotifyRateLimitError as exc:
        if exc.retry_after and exc.retry_after <= MAX_RETRY_WAIT_SECONDS:
            print(f"  Rate limited, waiting {exc.retry_after}s before one retry...")
            time.sleep(exc.retry_after + 1)
            return spotify_call(func, *args, **kwargs)
        raise


def _extract_retry_after(exc: SpotifyException) -> int | None:
    headers = getattr(exc, "headers", None)
    if headers and "Retry-After" in headers:
        try:
            return int(headers["Retry-After"])
        except (TypeError, ValueError):
            return None
    return None
