"""
leakage_probes.py — Đo leakage trên dataset mood Spotify (post-patch)

Chạy:  python idk.py data/processed/spotify_hybrid_processed.csv

Sau khi re-run Bước 7–8, CSV không còn bản leaky (combined_text = text_content).
Số TRƯỚC (0.993, recovery 0.811, …) chỉ lưu ARCHIVED — không tái tạo từ CSV hiện tại.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

MOOD_KEYWORDS = {
    "happy": ["happy", "joy", "excited", "fun", "cheerful", "vui", "hạnh phúc"],
    "sad": ["sad", "lonely", "depressed", "cry", "buồn", "cô đơn"],
    "calm": ["calm", "relax", "peaceful", "chill", "thư giãn", "bình yên"],
    "energetic": ["energy", "workout", "pump", "hype", "năng lượng", "tập trung"],
    "romantic": ["love", "romantic", "date", "yêu", "lãng mạn"],
    "stressed": ["stress", "anxious", "tired", "overwhelmed", "căng thẳng", "mệt"],
}
ALL_KW = {k for kws in MOOD_KEYWORDS.values() for k in kws}
STOP = set(ENGLISH_STOP_WORDS)
GENERIC = {"liked", "songs", "song", "playlist", "nan", "unknown"}
N_SPLITS = 5
SEED = 42

# Probe đầu tiên trên CSV TRƯỚC khi vá Bước 7 (đã bị ghi đè — chỉ dùng làm bằng chứng báo cáo)
ARCHIVED_PRE_PATCH = {
    "[A] recovery combined_text (leaky)": 0.811,
    "[B] model tokens (leaky)": 0.993,
    "[E] query-only": 1.000,
    "[E] tokens leaky GROUPED query": 0.906,
}


def clf():
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000))


def cv_acc(X, y, groups=None):
    X, y = list(X), list(y)
    if groups is not None:
        n_groups = pd.Series(groups).nunique()
        k = min(N_SPLITS, n_groups)
        if k < 2:
            return float("nan")
        return cross_val_score(
            clf(), X, y, cv=GroupKFold(k), groups=list(groups), scoring="accuracy"
        ).mean()
    k = min(N_SPLITS, pd.Series(y).value_counts().min())
    if k < 2:
        return float("nan")
    return cross_val_score(
        clf(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=SEED), scoring="accuracy"
    ).mean()


def recover_label(text):
    text = str(text).lower()
    scores = {m: sum(1 for kw in kws if kw in text) for m, kws in MOOD_KEYWORDS.items()}
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score > 0 else None


def strip_keywords(tokens_str):
    return " ".join(t for t in str(tokens_str).split() if t not in ALL_KW)


def safe_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def dedupe_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def tokenize_text(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return ""
    tokens = dedupe_tokens(
        [t for t in text.split() if t not in STOP and t not in GENERIC and len(t) > 1]
    )
    return " ".join(tokens)


def build_text_content_row(row) -> str:
    parts = [
        safe_str(row.get("track_name_clean") or row.get("track_name")),
        safe_str(row.get("artists_clean") or row.get("artists")),
        safe_str(row.get("album")),
    ]
    return " ".join(p for p in parts if p)


def build_text_crawl_context_row(row) -> str:
    if row.get("data_origin") != "search":
        return ""
    parts = [safe_str(row.get("search_query")), safe_str(row.get("tags_normalized"))]
    return " ".join(dict.fromkeys(p for p in parts if p))


def assign_label_source_row(row) -> str:
    if row.get("data_origin") == "search":
        return "search_query"
    if row.get("mood_inferred"):
        return "inferred"
    return "default_calm"


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    if "text_content" not in df.columns:
        df["text_content"] = df.apply(build_text_content_row, axis=1)
    if "text_crawl_context" not in df.columns:
        df["text_crawl_context"] = df.apply(build_text_crawl_context_row, axis=1)
    df["text_content"] = df["text_content"].fillna("").astype(str)
    df["text_crawl_context"] = df["text_crawl_context"].fillna("").astype(str)
    if "label_source" not in df.columns:
        df["label_source"] = df.apply(assign_label_source_row, axis=1)
    # tokens từ text_content — ưu tiên cột tokens trong CSV nếu đã re-run notebook
    if "tokens" in df.columns:
        df["tokens_content"] = df["tokens"].fillna("").astype(str).apply(tokenize_text)
        # nếu tokens CSV rỗng, fallback tokenize text_content
        empty = df["tokens_content"].str.strip() == ""
        df.loc[empty, "tokens_content"] = df.loc[empty, "text_content"].apply(tokenize_text)
    else:
        df["tokens_content"] = df["text_content"].apply(tokenize_text)
    df["tokens_context"] = df["text_crawl_context"].apply(tokenize_text)
    return df


def is_post_patch(df: pd.DataFrame) -> bool:
    """CSV sau Bước 7 mới: combined_text == text_content, không còn query trong feature chính."""
    if "text_crawl_context" not in df.columns:
        return False
    if "combined_text" not in df.columns or "text_content" not in df.columns:
        return True
    sample = df.head(min(1000, len(df)))
    eq = (sample["combined_text"].fillna("") == sample["text_content"].fillna("")).mean()
    recovery = (sample["text_content"].apply(recover_label) == sample["mood_label"]).mean()
    return eq > 0.95 or recovery < 0.35


def fmt(v):
    return f"{v:.3f}" if not np.isnan(v) else "n/a"


def main(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} cols")
    if "mood_label" not in df.columns:
        sys.exit("Thiếu cột mood_label")

    df = df.dropna(subset=["mood_label"]).copy()
    df = ensure_features(df)
    post_patch = is_post_patch(df)

    baseline = df["mood_label"].value_counts(normalize=True).max()
    print(f"Số lớp mood: {df['mood_label'].nunique()} | baseline (lớp đa số): {baseline:.3f}")
    print(f"CSV mode: {'post-patch (feature sạch)' if post_patch else 'legacy / mixed'}\n")

    if "label_source" in df.columns:
        print("Nguồn gán nhãn:")
        print(df["label_source"].value_counts().to_string(), "\n")

    probes: dict[str, float] = {}

    probes["[A] recovery text_content"] = (
        df["text_content"].apply(recover_label) == df["mood_label"]
    ).mean()
    probes["[B] model tokens (text_content)"] = cv_acc(df["tokens_content"], df["mood_label"])
    df["tokens_ablated"] = df["tokens_content"].apply(strip_keywords)
    probes["[C] model SAU ablation keyword"] = cv_acc(df["tokens_ablated"], df["mood_label"])

    if {"track_name_clean", "artists_clean"}.issubset(df.columns):
        tc = (
            df["track_name_clean"].fillna("") + " " + df["artists_clean"].fillna("")
        ).str.strip()
        probes["[D] model chỉ track+artist"] = cv_acc(tc, df["mood_label"])

    if "artists_clean" in df.columns:
        probes["[G] model artist-only"] = cv_acc(df["artists_clean"].fillna(""), df["mood_label"])

    ctx = df[df["tokens_context"].str.strip() != ""]
    if len(ctx) > 50 and ctx["mood_label"].nunique() > 1:
        probes["[E'] crawl_context (search, random)"] = cv_acc(ctx["tokens_context"], ctx["mood_label"])

    if "data_origin" in df.columns and "search_query" in df.columns:
        s = df[df["data_origin"] == "search"].copy()
        s["search_query"] = s["search_query"].fillna("").astype(str)
        s = s[s["search_query"].str.strip() != ""]
        if len(s) > 50 and s["mood_label"].nunique() > 1:
            probes["[E] query-only (search, random)"] = cv_acc(s["search_query"], s["mood_label"])

    print("=" * 62)
    if post_patch:
        print("ARCHIVED — CSV leaky đã bị ghi đè, KHÔNG tái tạo từ file hiện tại:")
        print("-" * 62)
        for k, v in ARCHIVED_PRE_PATCH.items():
            print(f"  {k:<42} {fmt(v)}")
        print()

    print("PROBE HIỆN TẠI (text_content = track+artist+album)")
    print("-" * 62)
    for k, v in probes.items():
        print(f"  {k:<42} {fmt(v)}")
    print("=" * 62)

    a_rec = probes.get("[A] recovery text_content")
    b_mod = probes.get("[B] model tokens (text_content)")
    c_abl = probes.get("[C] model SAU ablation keyword")
    d_ta = probes.get("[D] model chỉ track+artist")
    g_art = probes.get("[G] model artist-only")
    e_q = probes.get("[E] query-only (search, random)")
    e_ctx = probes.get("[E'] crawl_context (search, random)")

    print("\nĐỌC KẾT QUẢ (báo cáo):")
    if post_patch:
        print(
            f"  - Trước vá (archived): recovery {ARCHIVED_PRE_PATCH['[A] recovery combined_text (leaky)']:.3f}"
            f" -> Sau: [A] {fmt(a_rec)} (< baseline {baseline:.3f} = keyword đã rời feature)"
        )
        print(
            f"  - Trước vá (archived): [B] {ARCHIVED_PRE_PATCH['[B] model tokens (leaky)']:.3f}"
            f" -> Sau: [B] {fmt(b_mod)}"
        )
    if a_rec is not None:
        print(f"  - [A] recovery thấp ({fmt(a_rec)}) = xác nhận query/tags không còn trong feature")
    if b_mod is not None and d_ta is not None:
        print(f"  - [B] ≈ [D]: {fmt(b_mod)} vs {fmt(d_ta)} — không rò query")
    if c_abl is not None and b_mod is not None:
        print(f"  - [C] ablation {fmt(b_mod)} -> {fmt(c_abl)}: phần lớn signal là metadata, không đọc nhãn")
    if e_q is not None and e_ctx is not None:
        print(f"  - [E]/[E'] ≈ 1.0: CẤM train trên crawl_context/query — đó là leakage, không thành tích")
    if g_art is not None:
        print(f"  - [G] artist-only {fmt(g_art)} = cận dưới phòng thủ (baseline {baseline:.3f})")

    print("\nTrain: ưu tiên label_source=search_query; cân nhắc bỏ 1134 default_calm.")
    print("Các số trên là BIÊN ĐO LEAKAGE / weak supervision — không phải 'mood ground truth'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python idk.py <đường_dẫn_processed.csv>")
        sys.exit(1)
    main(sys.argv[1])
