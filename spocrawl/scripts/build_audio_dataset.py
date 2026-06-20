"""Xuất dataset training chỉ gồm bài có audio preview (audio_status=ok).

Dùng sau khi crawl xong, hoặc từ checkpoint đang chạy dở:
  python scripts/build_audio_dataset.py
  python scripts/build_audio_dataset.py --checkpoint deezer_audio_checkpoint.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import settings
from fetch_deezer_audio import CHECKPOINT_META_COLS, FEATURE_COLS

AUDIO_COLS = list(CHECKPOINT_META_COLS) + list(FEATURE_COLS)

OUTPUT_NAME = "spotify_audio_training.csv"
SUMMARY_NAME = "audio_training_summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build audio-only training CSV")
    p.add_argument("--input", default="spotify_hybrid_processed.csv")
    p.add_argument("--checkpoint", default="deezer_audio_checkpoint.csv")
    p.add_argument("--features", default="audio_features.csv", help="Ưu tiên nếu đã có")
    p.add_argument("--output", default=OUTPUT_NAME)
    return p.parse_args()


def load_audio_table(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    feat_path = out_dir / args.features
    ckpt_path = out_dir / args.checkpoint
    if feat_path.exists():
        return pd.read_csv(feat_path)
    if ckpt_path.exists():
        return pd.read_csv(ckpt_path)
    print(f"Error: không tìm thấy {feat_path} hoặc {ckpt_path}")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    out_dir = settings.processed_data_dir
    input_path = out_dir / args.input
    if not input_path.exists():
        print(f"Error: {input_path} không tồn tại")
        sys.exit(1)

    audio = load_audio_table(args, out_dir)
    ok = audio[audio["audio_status"] == "ok"].copy()
    if ok.empty:
        print("Error: chưa có bài nào audio_status=ok")
        sys.exit(1)

    meta = pd.read_csv(input_path)
    drop = [c for c in AUDIO_COLS if c in meta.columns]
    meta = meta.drop(columns=drop, errors="ignore")
    audio_cols = ["track_id"] + [c for c in AUDIO_COLS if c in ok.columns]
    merged = meta.merge(ok[audio_cols], on="track_id", how="inner")

    output_path = out_dir / args.output
    merged.to_csv(output_path, index=False)

    summary = {
        "source_processed": str(input_path),
        "audio_rows_total": len(audio),
        "audio_ok": len(ok),
        "training_rows": len(merged),
        "mood_distribution": merged["mood_label"].value_counts().to_dict(),
        "label_source_distribution": merged["label_source"].value_counts().to_dict()
        if "label_source" in merged.columns
        else {},
        "output": str(output_path),
        "note": "Chỉ bài có preview Deezer + librosa features; dùng file này cho train multimodal.",
    }
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Audio OK: {len(ok)}/{len(audio)}")
    print(f"Training set: {len(merged)} rows -> {output_path}")
    print(merged["mood_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
