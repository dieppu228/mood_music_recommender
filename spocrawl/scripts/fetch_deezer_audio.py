"""Fetch Deezer preview + librosa audio features cho subset English.

Default: ~3k bài Tây (500/mood x 6), checkpoint/resume, ghi audio_features.csv + merge processed.

Chạy:
  pip install librosa soundfile
  python scripts/fetch_deezer_audio.py
  python scripts/fetch_deezer_audio.py --per-mood 500 --delay 0.4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import settings
from deezer_io import build_western_sample, search_deezer

CHECKPOINT_FILE = "deezer_audio_checkpoint.csv"
SUMMARY_FILE = "deezer_audio_summary.json"
FEATURES_FILE = "audio_features.csv"
SUBSET_FILE = "spotify_audio_subset.csv"
WESTERN_SAMPLE_FILE = "spotify_western_sample.csv"
TRAINING_FILE = "spotify_audio_training.csv"
MFCC_N = 13
CHROMA_N = 12
CONTRAST_N = 7
FEATURE_SCHEMA = "v2"

_SPECTRAL = ("audio_rms", "audio_centroid", "audio_rolloff", "audio_zcr")
FEATURE_COLS = (
    ["audio_tempo"]
    + list(_SPECTRAL)
    + [f"{c}_std" for c in _SPECTRAL]
    + [f"audio_mfcc_{i:02d}" for i in range(1, MFCC_N + 1)]
    + [f"audio_mfcc_{i:02d}_std" for i in range(1, MFCC_N + 1)]
    + [f"audio_chroma_{i:02d}" for i in range(1, CHROMA_N + 1)]
    + [f"audio_contrast_{i:02d}" for i in range(1, CONTRAST_N + 1)]
)
CHECKPOINT_META_COLS = (
    "deezer_match_status",
    "deezer_id",
    "deezer_preview_url",
    "audio_status",
    "audio_error",
    "feature_schema",
)
N_FFT = 2048
MIN_AUDIO_SAMPLES = N_FFT
SILENCE_RMS = 1e-4
DOWNLOAD_RETRIES = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Deezer preview audio features")
    p.add_argument("--input", default="spotify_hybrid_processed.csv")
    p.add_argument("--per-mood", type=int, default=500, help="Bài/mood trong pool Tây (~3k)")
    p.add_argument("--limit", type=int, default=0, help="Cap tổng (0 = chỉ per-mood)")
    p.add_argument("--delay", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true")
    p.add_argument(
        "--retry-status",
        default="",
        help="Comma-separated statuses to drop and re-queue (e.g. connection_error,feature_failed,download_failed)",
    )
    return p.parse_args()


def parse_retry_status(raw: str) -> set[str]:
    return {s.strip().lower().replace("-", "_") for s in raw.split(",") if s.strip()}


def _is_connection_error(status: str) -> bool:
    s = str(status or "").lower()
    return "connectionerror" in s.replace("_", "")


def should_drop_for_retry(rec: dict, retry_status: set[str]) -> bool:
    if not retry_status:
        return False
    audio_status = str(rec.get("audio_status", "")).lower().replace("-", "_")
    match_status = str(rec.get("deezer_match_status", "")).lower().replace("-", "_")
    if "connection_error" in retry_status and (
        _is_connection_error(audio_status) or _is_connection_error(match_status)
    ):
        return True
    if "feature_failed" in retry_status and audio_status == "feature_failed":
        return True
    if "download_failed" in retry_status and audio_status == "download_failed":
        return True
    if "not_found" in retry_status and audio_status == "not_found":
        return True
    return False


def _has_v2_features(rec: dict) -> bool:
    if rec.get("feature_schema") == FEATURE_SCHEMA:
        return True
    chroma = rec.get("audio_chroma_01")
    return chroma is not None and pd.notna(chroma)


def needs_reextract(rec: dict) -> bool:
    if rec.get("audio_status") != "ok":
        return False
    if not _has_v2_features(rec):
        return True
    return any(col not in rec or pd.isna(rec.get(col)) for col in FEATURE_COLS)


def load_checkpoint(path: Path, retry_status: set[str] | None = None) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(path, error_bad_lines=False)
    out: dict[str, dict] = {}
    stale = 0
    retried = 0
    retry_status = retry_status or set()
    for _, row in df.iterrows():
        tid = str(row["track_id"])
        rec = {c: row.get(c) for c in df.columns if c != "track_id"}
        if needs_reextract(rec):
            stale += 1
            continue
        if should_drop_for_retry(rec, retry_status):
            retried += 1
            continue
        out[tid] = rec
    if stale:
        print(f"Checkpoint: dropped {stale} stale schema rows (will re-fetch preview)", flush=True)
    if retried:
        labels = ",".join(sorted(retry_status))
        print(f"Checkpoint: dropped {retried} rows for --retry-status ({labels})", flush=True)
    return out


def save_checkpoint(path: Path, cache: dict[str, dict]) -> None:
    rows = [{"track_id": tid, **rec} for tid, rec in cache.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def download_preview(url: str) -> tuple[bytes | None, str]:
    if not url:
        return None, "empty_url"
    last_exc = ""
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                return r.content, ""
            last_exc = f"http_{r.status_code}"
        except requests.RequestException as exc:
            last_exc = f"{type(exc).__name__}: {exc}"
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(1.5 * (attempt + 1))
    return None, last_exc


def _frame_stats(matrix: np.ndarray, prefix: str, with_std: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(matrix.shape[0]):
        idx = i + 1
        col = matrix[i]
        out[f"{prefix}_{idx:02d}"] = float(np.mean(col))
        if with_std:
            out[f"{prefix}_{idx:02d}_std"] = float(np.std(col))
    return out


def extract_features(audio_bytes: bytes) -> tuple[dict[str, float] | None, str, str]:
    """Return (features, status, error_message). status: ok | clip_too_short | feature_failed."""
    import librosa

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=30)
    except Exception as exc:
        return None, "feature_failed", f"load:{type(exc).__name__}: {exc}"
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    if len(y) < MIN_AUDIO_SAMPLES:
        return None, "clip_too_short", f"len(y)={len(y)} < n_fft={MIN_AUDIO_SAMPLES}"

    rms_level = float(np.sqrt(np.mean(np.square(y))))
    if rms_level < SILENCE_RMS:
        return None, "clip_too_short", f"near_silent:rms={rms_level:.2e}"

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0

        rms_f = librosa.feature.rms(y=y)
        centroid_f = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff_f = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr_f = librosa.feature.zero_crossing_rate(y)
        mfcc_f = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N)
        chroma_f = librosa.feature.chroma_cens(y=y, sr=sr)
        contrast_f = librosa.feature.spectral_contrast(y=y, sr=sr)
    except Exception as exc:
        return None, "feature_failed", f"extract:{type(exc).__name__}: {exc}"

    out: dict[str, float] = {
        "audio_tempo": tempo_val if np.isfinite(tempo_val) else 0.0,
        "feature_schema": FEATURE_SCHEMA,
    }
    for name, mat in (
        ("audio_rms", rms_f),
        ("audio_centroid", centroid_f),
        ("audio_rolloff", rolloff_f),
        ("audio_zcr", zcr_f),
    ):
        out[name] = float(np.mean(mat))
        out[f"{name}_std"] = float(np.std(mat))
    out.update(_frame_stats(mfcc_f, "audio_mfcc", with_std=True))
    out.update(_frame_stats(chroma_f, "audio_chroma", with_std=False))
    out.update(_frame_stats(contrast_f, "audio_contrast", with_std=False))
    return out, "ok", ""


def fetch_one(artist: str, title: str) -> dict:
    item, match_status = search_deezer(artist, title)
    base: dict = {
        "deezer_match_status": match_status,
        "deezer_id": item.get("id") if item else None,
        "deezer_preview_url": item.get("preview", "") if item else "",
        "audio_status": "not_found",
        "audio_error": "",
    }
    if match_status.startswith("error:"):
        base["audio_status"] = match_status
        base["audio_error"] = match_status
        return base
    if not item:
        base["audio_status"] = "not_found"
        base["audio_error"] = "not_found"
        return base

    preview = base["deezer_preview_url"]
    if not preview:
        base["audio_status"] = "no_preview"
        base["audio_error"] = "no_preview"
        return base

    audio_bytes, dl_err = download_preview(preview)
    if not audio_bytes:
        if _is_connection_error(dl_err):
            base["audio_status"] = "error:ConnectionError"
            base["audio_error"] = dl_err
        else:
            base["audio_status"] = "download_failed"
            base["audio_error"] = dl_err or "download_failed"
        return base

    feats, feat_status, feat_err = extract_features(audio_bytes)
    if feat_status == "clip_too_short":
        base["audio_status"] = "clip_too_short"
        base["audio_error"] = feat_err
        return base
    if feat_status != "ok" or not feats:
        base["audio_status"] = "feature_failed"
        base["audio_error"] = feat_err or "unknown"
        return base

    base.update(feats)
    base["audio_status"] = "ok"
    return base


def main() -> None:
    args = parse_args()
    input_path = settings.processed_data_dir / args.input
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    try:
        import librosa  # noqa: F401
    except ImportError:
        print("Error: pip install librosa soundfile")
        sys.exit(1)

    df = pd.read_csv(input_path)
    sample = build_western_sample(df, args.per_mood, args.limit, args.seed)
    print(f"Western subset: {len(sample)} tracks", flush=True)
    print(sample["mood_label"].value_counts().to_string(), flush=True)

    out_dir = settings.processed_data_dir
    pin_path = out_dir / WESTERN_SAMPLE_FILE
    sample.assign(track_id=sample["track_id"].astype(str))[["track_id", "mood_label"]].to_csv(
        pin_path, index=False
    )
    print(f"Pinned western sample: {pin_path}", flush=True)

    ckpt_path = out_dir / CHECKPOINT_FILE
    retry_status = parse_retry_status(args.retry_status)
    cache: dict[str, dict] = {} if args.fresh else load_checkpoint(ckpt_path, retry_status)
    pending = sample[~sample["track_id"].astype(str).isin(cache.keys())]
    started = time.time()

    print(f"Checkpoint: {len(cache)} | Pending: {len(pending)}", flush=True)

    for i, (_, row) in enumerate(pending.iterrows(), start=1):
        tid = str(row["track_id"])
        artist = str(row.get("primary_artist") or row.get("artists", "")).split(",")[0].strip()
        title = str(row["track_name"]).strip()
        rec = fetch_one(artist, title)
        cache[tid] = rec

        if i % 10 == 0 or i == len(pending):
            save_checkpoint(ckpt_path, cache)
            ok = sum(1 for v in cache.values() if v.get("audio_status") == "ok")
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"  [{i}/{len(pending)}] audio_ok={ok}/{len(cache)} "
                f"rate={rate:.2f}/s last={title[:30]!r} -> {rec['audio_status']}"
                + (f" ({rec['audio_error'][:60]})" if rec.get("audio_error") else ""),
                flush=True,
            )
        time.sleep(args.delay)

    save_checkpoint(ckpt_path, cache)

    feat_rows = []
    for _, row in sample.iterrows():
        tid = str(row["track_id"])
        rec = cache.get(tid, {})
        feat_rows.append({"track_id": tid, **rec})
    feat_df = pd.DataFrame(feat_rows)

    subset = sample.merge(feat_df, on="track_id", how="left")
    training = subset[subset["audio_status"] == "ok"].copy()
    out_dir = settings.processed_data_dir
    feat_df.to_csv(out_dir / FEATURES_FILE, index=False)
    subset.to_csv(out_dir / SUBSET_FILE, index=False)
    training.to_csv(out_dir / TRAINING_FILE, index=False)

    n = len(feat_df)
    ok = int((feat_df["audio_status"] == "ok").sum())
    summary = {
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": len(FEATURE_COLS),
        "western_subset": len(sample),
        "processed_total": len(df),
        "audio_ok": ok,
        "audio_ok_pct": round(ok / n * 100, 2) if n else 0,
        "status_counts": feat_df["audio_status"].value_counts().to_dict(),
        "by_mood": (
            subset.groupby("mood_label")["audio_status"]
            .apply(lambda s: int((s == "ok").sum()))
            .to_dict()
        ),
        "training_rows": len(training),
        "files": {
            "checkpoint": str(ckpt_path),
            "features": str(out_dir / FEATURES_FILE),
            "subset": str(out_dir / SUBSET_FILE),
            "training": str(out_dir / TRAINING_FILE),
        },
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAudio OK: {ok}/{n} ({summary['audio_ok_pct']}%)", flush=True)
    print(f"Training (audio only): {len(training)} -> {out_dir / TRAINING_FILE}", flush=True)
    print(f"Saved: {out_dir / FEATURES_FILE}", flush=True)
    print(f"Summary: {out_dir / SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
