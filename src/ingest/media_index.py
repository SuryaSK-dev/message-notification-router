"""
Resolves a media_id (image or voice note) to an actual file path on disk.
"""
from pathlib import Path
from ingest.loaders import Dataset

DATASET_DIR = Path("dataset")


def resolve_media_path(dataset: Dataset, media_type, media_id) -> Path | None:
    """
    media_type: 'image' or 'voice' (matches messages.csv media_type values)
    media_id: e.g. 'img_007' or 'vn_003'; may be NaN/float for text-only messages
    Returns a Path relative to the project root, or None if missing/not found.
    """
    if not isinstance(media_id, str) or not media_id:
        return None

    if media_type == "image":
        row = dataset.images.loc[dataset.images["image_id"] == media_id]
        if row.empty:
            return None
        return DATASET_DIR / row.iloc[0]["file_path"]

    if media_type == "voice":
        row = dataset.voice_notes.loc[dataset.voice_notes["voice_note_id"] == media_id]
        if row.empty:
            return None
        return DATASET_DIR / row.iloc[0]["file_path"]

    return None