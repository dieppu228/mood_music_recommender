MOOD_PLAYLIST_QUERIES: dict[str, list[str]] = {
    "happy": [
        "happy playlist",
        "feel good playlist",
        "upbeat dance playlist",
        "positive vibes playlist",
        "summer party playlist",
    ],
    "sad": [
        "sad playlist",
        "heartbreak playlist",
        "melancholy indie playlist",
        "lonely songs playlist",
        "crying playlist",
    ],
    "calm": [
        "chill playlist",
        "lofi beats playlist",
        "peaceful acoustic playlist",
        "sleep music playlist",
        "ambient relax playlist",
    ],
    "energetic": [
        "workout playlist",
        "gym motivation playlist",
        "high energy edm playlist",
        "running playlist",
        "cardio hits playlist",
    ],
    "romantic": [
        "romantic playlist",
        "love songs playlist",
        "date night playlist",
        "love ballads playlist",
        "r&b love playlist",
    ],
    "stressed": [
        "stress relief playlist",
        "meditation playlist",
        "calming instrumental playlist",
        "anxiety relief playlist",
        "spa music playlist",
    ],
}

MOOD_EXTRA_TRACK_QUERIES: dict[str, list[str]] = {
    "happy": [
        "year:2024 genre:pop",
        "year:2023 genre:dance",
        "feel good morning",
        "upbeat indie pop",
        "party hits",
        "genre:pop happy",
        "summer hits",
        "dance pop 2024",
    ],
    "sad": [
        "year:2024 genre:indie sad",
        "heartbreak acoustic",
        "lonely night songs",
        "genre:indie melancholy",
        "sad piano",
        "breakup songs 2024",
        "emotional ballad",
        "genre:r-n-b sad",
    ],
    "calm": [
        "year:2024 genre:ambient",
        "genre:chillhop",
        "soft piano instrumental",
        "genre:acoustic calm",
        "rainy day chill",
        "study focus music",
        "genre:jazz chill",
        "sleep ambient",
    ],
    "energetic": [
        "year:2024 genre:edm",
        "genre:rock workout",
        "genre:hip-hop energy",
        "running motivation",
        "genre:electronic dance",
        "pump up hits",
        "genre:metal workout",
        "cardio edm",
    ],
    "romantic": [
        "year:2024 genre:r-n-b love",
        "slow love songs",
        "genre:soul romantic",
        "wedding songs",
        "genre:indie love",
        "acoustic love",
        "date night rnb",
        "genre:pop love songs",
    ],
    "stressed": [
        "year:2024 genre:ambient",
        "genre:classical relax",
        "nature sounds sleep",
        "genre:new-age meditation",
        "deep focus instrumental",
        "genre:lo-fi study",
        "wind down music",
        "genre:instrumental calm",
    ],
}

MOOD_SUPPLEMENT_QUERIES: dict[str, list[str]] = {
    "stressed": [
        "anxiety relief music",
        "burnout recovery playlist",
        "nhạc thư giãn căng thẳng",
        "overwhelmed calm songs",
        "stress relief piano",
        "tension release instrumental",
        "mental health chill",
        "genre:ambient anxiety",
    ],
    "romantic": [
        "slow dance love songs",
        "anniversary playlist",
        "nhạc tình yêu lãng mạn",
        "first dance wedding",
        "genre:r-n-b romance",
        "acoustic love ballads",
        "late night love",
        "valentine playlist",
    ],
    "energetic": [
        "pre workout hype",
        "hiit cardio playlist",
        "nhạc tập gym năng lượng",
        "boxing training music",
        "genre:drum and bass workout",
        "morning energy boost",
        "powerlifting motivation",
        "dance workout hits",
    ],
}

# Per-track search queries (playlist-style + genre/year queries)
MOOD_SEARCH_QUERIES: dict[str, list[str]] = {
    mood: (
        [f"{q} playlist" if "playlist" not in q else q for q in MOOD_PLAYLIST_QUERIES[mood]]
        + MOOD_EXTRA_TRACK_QUERIES.get(mood, [])
    )
    for mood in MOOD_PLAYLIST_QUERIES
}


def flatten_mood_queries() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for mood, queries in MOOD_PLAYLIST_QUERIES.items():
        for query in queries:
            pairs.append((mood, query))
    return pairs
