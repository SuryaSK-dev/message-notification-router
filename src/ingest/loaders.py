"""
Loads all dataset CSVs into pandas DataFrames.
Single source of truth for dataset paths and raw loading.
"""
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("dataset")


def _read(filename: str) -> pd.DataFrame:
    path = DATASET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {path}")
    return pd.read_csv(path)


class Dataset:
    """Holds all raw tables. Loaded once, reused across the pipeline."""

    def __init__(self):
        self.messages = _read("messages.csv")
        self.users = _read("users.csv")
        self.groups = _read("groups.csv")
        self.group_members = _read("group_members.csv")
        self.business_accounts = _read("business_accounts.csv")
        self.user_business_history = _read("user_business_history.csv")
        self.message_history = _read("message_history.csv")
        self.message_events = _read("message_events.csv")
        self.images = _read("images.csv")
        self.voice_notes = _read("voice_notes.csv")
        self.daily_notification_summary = _read("daily_notification_summary.csv")

    def summary(self) -> str:
        names = [
            "messages", "users", "groups", "group_members", "business_accounts",
            "user_business_history", "message_history", "message_events",
            "images", "voice_notes", "daily_notification_summary",
        ]
        return "\n".join(f"{n}: {len(getattr(self, n))} rows" for n in names)


def load_dataset() -> Dataset:
    return Dataset()