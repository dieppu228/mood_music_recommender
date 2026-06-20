import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str
    root_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    models_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        client_id = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("client_id", "")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET") or os.getenv("client_secret", "")
        redirect_uri = os.getenv(
            "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"
        )

        data_dir = ROOT_DIR / "data"
        return cls(
            spotify_client_id=client_id,
            spotify_client_secret=client_secret,
            spotify_redirect_uri=redirect_uri,
            root_dir=ROOT_DIR,
            data_dir=data_dir,
            raw_data_dir=data_dir / "raw",
            processed_data_dir=data_dir / "processed",
            models_dir=data_dir / "models",
        )


settings = Settings.from_env()
