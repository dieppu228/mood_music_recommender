"""Dem nhanh ti le ngon ngu (heuristic) tu processed CSV."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings

VN_CHARS = re.compile(r"[\u00c0-\u1ef9]")
VN_WORDS = re.compile(r"\b(viet|vietnam|vpop|bolero|remix viet|nhac viet)\b", re.I)


def lang_bucket(row: pd.Series) -> str:
    for col in ("track_name", "primary_artist", "artists", "search_query"):
        text = str(row.get(col, "") or "")
        if VN_CHARS.search(text) or VN_WORDS.search(text):
            return "likely_vn"
    combined = " ".join(
        str(row.get(c, "") or "") for c in ("track_name", "primary_artist", "artists")
    )
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", combined):
        return "likely_asia_non_vn"
    return "likely_western_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="spotify_hybrid_processed.csv")
    args = parser.parse_args()

    path = settings.processed_data_dir / args.input
    df = pd.read_csv(path)
    df["lang_bucket"] = df.apply(lang_bucket, axis=1)

    print(f"File: {path} ({len(df)} tracks)\n")
    counts = df["lang_bucket"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    for k in counts.index:
        print(f"  {k}: {counts[k]} ({pct[k]}%)")

    if "data_origin" in df.columns:
        print("\nBy data_origin:")
        print(pd.crosstab(df["data_origin"], df["lang_bucket"], margins=True).to_string())

    if "mood_label" in df.columns:
        print("\nBy mood_label:")
        print(pd.crosstab(df["mood_label"], df["lang_bucket"]).to_string())


if __name__ == "__main__":
    main()
