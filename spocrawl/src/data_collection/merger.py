import pandas as pd


PROFILE_PRIORITY_COLUMNS = [
    "album_id",
    "album_type",
    "explicit",
    "disc_number",
    "track_number",
    "isrc",
    "added_at",
    "source_type",
    "source_id",
    "source_name",
]

SEARCH_PRIORITY_COLUMNS = [
    "mood_label",
    "search_query",
    "tags",
    "genres",
]


def merge_datasets(
    profile_df: pd.DataFrame,
    search_df: pd.DataFrame,
) -> pd.DataFrame:
    profile = profile_df.copy()
    search = search_df.copy()

    profile["data_origin"] = "profile"
    search["data_origin"] = "search"

    if "source_type" not in search.columns:
        search["source_type"] = "search"
    if "source_name" not in search.columns:
        search["source_name"] = search.get("search_query", pd.Series(dtype=str))

    combined = pd.concat([profile, search], ignore_index=True)
    combined = combined.drop_duplicates(subset=["track_id"], keep="first")

    for col in PROFILE_PRIORITY_COLUMNS + SEARCH_PRIORITY_COLUMNS + ["data_origin"]:
        if col not in combined.columns:
            combined[col] = pd.NA

    return combined
