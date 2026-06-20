"""Pilot Deezer preview coverage — dùng trước khi fetch full."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import settings
from deezer_io import build_western_sample, lang_bucket, search_deezer

SUMMARY_FILE = "deezer_probe_summary.json"
CHECKPOINT_FILE = "deezer_probe_checkpoint.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="spotify_hybrid_processed.csv")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--per-mood", type=int, default=0)
    p.add_argument("--western-only", action="store_true", default=True)
    p.add_argument("--delay", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    path = settings.processed_data_dir / args.input
    df = pd.read_csv(path)
    df["lang_bucket"] = df.apply(lang_bucket, axis=1)
    if args.western_only:
        sample = build_western_sample(df, args.per_mood, args.limit, args.seed)
    elif args.per_mood > 0:
        sample = build_western_sample(df, args.per_mood, 0, args.seed)
        if args.limit:
            sample = sample.head(args.limit)
    else:
        sample = df.sample(n=min(args.limit, len(df)), random_state=args.seed)

    print(f"Sample: {len(sample)}", flush=True)
    records = []
    t0 = time.time()
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        artist = str(row.get("primary_artist") or row.get("artists", "")).split(",")[0].strip()
        title = str(row["track_name"]).strip()
        item, status = search_deezer(artist, title)
        preview = item.get("preview", "") if item else ""
        records.append(
            {
                "track_id": row["track_id"],
                "status": status,
                "has_preview": bool(preview),
            }
        )
        if i % 25 == 0 or i == len(sample):
            prev = sum(r["has_preview"] for r in records)
            print(f"  [{i}/{len(sample)}] preview={prev} rate={i/(time.time()-t0):.2f}/s", flush=True)
        time.sleep(args.delay)

    res = pd.DataFrame(records)
    n = len(res)
    prev = int(res["has_preview"].sum())
    summary = {"sample": n, "with_preview": prev, "preview_pct": round(prev / n * 100, 1)}
    out = settings.processed_data_dir
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    res.to_csv(out / CHECKPOINT_FILE, index=False)
    print(f"Preview: {prev}/{n} ({summary['preview_pct']}%)", flush=True)


if __name__ == "__main__":
    main()
