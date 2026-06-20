"""Probe lyrics sources on western sample — research only."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import settings
from deezer_io import is_western

VN_CHARS = re.compile(r"[\u00c0-\u1ef9]")


def artist_name(row) -> str:
    return str(row.get("primary_artist") or row.get("artists", "")).split(",")[0].strip()


def has_lyrics(text: str | None, min_len: int = 80) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    return len(t) >= min_len and not t.lower().startswith("instrumental")


def probe_lrclib(artist: str, title: str) -> tuple[str, int]:
    try:
        r = requests.get(
            "https://lrclib.net/api/search",
            params={"track_name": title, "artist_name": artist},
            timeout=20,
            headers={"User-Agent": "ds-mood-probe/1.0"},
        )
        if r.status_code == 429:
            return "rate_limit", 0
        if r.status_code != 200:
            return f"http_{r.status_code}", 0
        data = r.json()
        if not data:
            return "not_found", 0
        for hit in data[:3]:
            plain = hit.get("plainLyrics") or ""
            synced = hit.get("syncedLyrics") or ""
            text = plain if len(plain) >= len(synced) else synced
            if has_lyrics(text):
                return "ok", len(text)
        return "no_lyrics_field", 0
    except requests.RequestException as exc:
        return f"error:{type(exc).__name__}", 0


def probe_lyrics_ovh(artist: str, title: str) -> tuple[str, int]:
    url = f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(title)}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 404:
            return "not_found", 0
        if r.status_code == 429:
            return "rate_limit", 0
        if r.status_code != 200:
            return f"http_{r.status_code}", 0
        lyrics = r.json().get("lyrics", "")
        if has_lyrics(lyrics):
            return "ok", len(lyrics)
        return "empty", 0
    except requests.RequestException as exc:
        return f"error:{type(exc).__name__}", 0


def probe_genius_search(token: str, artist: str, title: str) -> tuple[str, int]:
    if not token:
        return "no_token", 0
    try:
        r = requests.get(
            "https://api.genius.com/search",
            params={"q": f"{title} {artist}"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if r.status_code != 200:
            return f"http_{r.status_code}", 0
        hits = r.json().get("response", {}).get("hits", [])
        if not hits:
            return "not_found", 0
        return "search_hit", 0
    except requests.RequestException as exc:
        return f"error:{type(exc).__name__}", 0


def probe_genius_scrape(artist: str, title: str) -> tuple[str, int]:
    try:
        r = requests.get(
            "https://genius.com/api/search/multi",
            params={"q": f"{title} {artist}"},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code == 403:
            return "scrape_403", 0
        if r.status_code != 200:
            return f"http_{r.status_code}", 0
        sections = r.json().get("response", {}).get("sections", [])
        for sec in sections:
            for hit in sec.get("hits", []):
                if hit.get("type") == "song":
                    url = hit.get("result", {}).get("url")
                    if url:
                        page = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                        if page.status_code == 403:
                            return "scrape_403", 0
                        if page.status_code == 200 and "Lyrics" in page.text:
                            return "page_ok_no_parse", len(page.text)
        return "not_found", 0
    except requests.RequestException as exc:
        return f"error:{type(exc).__name__}", 0


def probe_deezer_lyrics(artist: str, title: str) -> tuple[str, int]:
    from deezer_io import search_deezer

    item, st = search_deezer(artist, title)
    if not item:
        return st, 0
    track_id = item.get("id")
    try:
        r = requests.get(f"https://api.deezer.com/track/{track_id}", timeout=20)
        if r.status_code != 200:
            return f"http_{r.status_code}", 0
        data = r.json()
        # Deezer sometimes exposes lyrics via separate endpoint in some regions
        lr = requests.get(f"https://api.deezer.com/track/{track_id}/lyrics", timeout=20)
        if lr.status_code == 200:
            body = lr.json()
            text = body.get("lyrics", "") or body.get("LYRICS_TEXT", "") or ""
            if has_lyrics(str(text)):
                return "ok", len(str(text))
        return "no_lyrics_endpoint", 0
    except requests.RequestException as exc:
        return f"error:{type(exc).__name__}", 0


def probe_chartlyrics(artist: str, title: str) -> tuple[str, int]:
    try:
        r = requests.get(
            "http://api.chartlyrics.com/apiv1.asmx/SearchLyric",
            params={"artist": artist, "song": title},
            timeout=20,
        )
        if r.status_code != 200:
            return f"http_{r.status_code}", 0
        if "LyricId" in r.text and "not found" not in r.text.lower():
            return "maybe_found_xml", len(r.text)
        return "not_found", 0
    except requests.RequestException as exc:
        return f"error:{type(exc).__name__}", 0


def summarize(series: pd.Series) -> dict:
    counts = series.value_counts().to_dict()
    ok = int((series == "ok").sum())
    n = len(series)
    return {"n": n, "ok": ok, "ok_pct": round(ok / n * 100, 1) if n else 0, "status": counts}


def main():
    import os
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    genius_token = os.getenv("GENIUS_ACCESS_TOKEN", "")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35

    df = pd.read_csv(settings.processed_data_dir / "spotify_hybrid_processed.csv")
    pool = df[df.apply(is_western, axis=1)]
    sample = pool.sample(n=min(n, len(pool)), random_state=42)

    sources = {
        "lrclib": probe_lrclib,
        "lyrics_ovh": probe_lyrics_ovh,
        "genius_search": lambda a, t: probe_genius_search(genius_token, a, t),
        "genius_scrape": probe_genius_scrape,
        "deezer_lyrics": probe_deezer_lyrics,
        "chartlyrics": probe_chartlyrics,
    }

    records = []
    print(f"Probing {len(sample)} western tracks, delay={delay}s", flush=True)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        a, t = artist_name(row), str(row["track_name"]).strip()
        rec = {"track_id": row["track_id"], "artist": a, "title": t}
        for name, fn in sources.items():
            status, length = fn(a, t)
            rec[f"{name}_status"] = status
            rec[f"{name}_len"] = length
            time.sleep(delay if name != "genius_search" else 0.1)
        records.append(rec)
        if i % 10 == 0:
            print(f"  [{i}/{len(sample)}]", flush=True)

    out = pd.DataFrame(records)
    summary = {src: summarize(out[f"{src}_status"]) for src in sources}
    out_path = settings.processed_data_dir / "lyrics_sources_probe.csv"
    sum_path = settings.processed_data_dir / "lyrics_sources_probe_summary.json"
    out.to_csv(out_path, index=False)
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===", flush=True)
    for src, s in summary.items():
        print(f"{src}: {s['ok']}/{s['n']} ({s['ok_pct']}%) | {s['status']}", flush=True)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
