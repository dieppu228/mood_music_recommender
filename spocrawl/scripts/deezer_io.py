"""Deezer search + preview helpers (shared by probe/fetch)."""

from __future__ import annotations

import re
import time

import pandas as pd
import requests

API_SEARCH = "https://api.deezer.com/search"
VN_CHARS = re.compile(r"[\u00c0-\u1ef9]")
SEARCH_RETRIES = 5


def normalize_title(title: str) -> str:
    s = str(title or "").strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"(?i)\b(feat\.?|ft\.?|featuring)\b.*$", "", s)
    s = re.sub(
        r"(?i)\s*-\s*(remaster(ed)?(\s+\d{4})?|remix(ed)?|mixed|edit(ed)?|live|radio|bonus|deluxe|version).*$",
        "",
        s,
    )
    s = re.sub(r"(?i)\s+-\s+(remixed|mixed|edited).*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_artist(artist: str) -> str:
    s = str(artist or "").strip().split(",")[0]
    s = re.sub(r"(?i)\b(feat\.?|ft\.?|featuring)\b.*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out


def norm(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def lang_bucket(row: pd.Series) -> str:
    for col in ("track_name", "primary_artist", "artists"):
        if VN_CHARS.search(str(row.get(col, "") or "")):
            return "likely_vn"
    combined = " ".join(str(row.get(c, "") or "") for c in ("track_name", "primary_artist"))
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", combined):
        return "likely_asia_non_vn"
    return "likely_western_other"


def is_western(row: pd.Series) -> bool:
    return lang_bucket(row) == "likely_western_other"


def _get(params: dict, retries: int = SEARCH_RETRIES):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return requests.get(API_SEARCH, params=params, timeout=25)
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_exc


def _fetch_search_items(q: str) -> tuple[list[dict], str]:
    try:
        r = _get({"q": q, "limit": 5})
    except requests.RequestException as exc:
        return [], f"error:{type(exc).__name__}"
    if r.status_code != 200:
        return [], f"error:http_{r.status_code}"
    data = r.json()
    if "error" in data:
        return [], f"error:dz_{data['error'].get('type', '?')}"
    return data.get("data", []), ""


def _pick_match(items: list[dict], artist: str, title: str) -> tuple[dict | None, str]:
    if not items:
        return None, "not_found"
    title_n = norm(title)
    clean_title_n = norm(normalize_title(title))
    artist_n = norm(artist)
    clean_artist_n = norm(normalize_artist(artist))
    for it in items:
        it_title = norm(it.get("title", ""))
        it_artist = norm(it.get("artist", {}).get("name", ""))
        title_hit = it_title in (title_n, clean_title_n)
        artist_hit = (
            not artist_n
            or artist_n in it_artist
            or clean_artist_n in it_artist
            or it_artist in artist_n
        )
        if title_hit and artist_hit:
            return it, "matched"
    return items[0], "matched_fuzzy"


def search_deezer(artist: str, title: str) -> tuple[dict | None, str]:
    clean_artist = normalize_artist(artist)
    clean_title = normalize_title(title)
    queries = _dedupe_queries(
        [
            f'artist:"{clean_artist}" track:"{clean_title}"' if clean_artist and clean_title else "",
            f'artist:"{artist}" track:"{title}"' if artist.strip() and title.strip() else "",
            f"{clean_artist} {clean_title}" if clean_artist and clean_title else "",
            f"{artist} {title}" if artist.strip() and title.strip() else "",
        ]
    )

    last_error = "not_found"
    for q in queries:
        items, err = _fetch_search_items(q)
        if err:
            last_error = err
            if err.startswith("error:"):
                return None, err
            continue
        if items:
            return _pick_match(items, artist, title)
    return None, last_error


def build_western_sample(
    df: pd.DataFrame,
    per_mood: int,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    pool = df[df.apply(is_western, axis=1)].copy()
    if per_mood > 0:
        parts = []
        for mood in sorted(pool["mood_label"].dropna().unique()):
            sub = pool[pool["mood_label"] == mood]
            parts.append(sub.sample(n=min(per_mood, len(sub)), random_state=seed))
        sample = pd.concat(parts, ignore_index=True).drop_duplicates("track_id")
    else:
        sample = pool
    if limit > 0 and len(sample) > limit:
        sample = sample.sample(n=limit, random_state=seed)
    return sample.reset_index(drop=True)
