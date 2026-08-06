"""
Disk cache for perception results, keyed by media_id.
Avoids re-calling Gemini for files already analyzed in a previous run —
important on a free-tier quota with a small, fixed set of media files.
"""
import json
from pathlib import Path
from perception.schemas import PerceptionResult

CACHE_DIR = Path("src/.cache/perception")


def _cache_path(media_id: str) -> Path:
    return CACHE_DIR / f"{media_id}.json"


def get_cached(media_id: str) -> PerceptionResult | None:
    path = _cache_path(media_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return PerceptionResult.from_dict(json.load(f))


def set_cached(result: PerceptionResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(result.media_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)