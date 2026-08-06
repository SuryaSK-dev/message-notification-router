"""
Phase 1 smoke tests — run with: python -m pytest code/tests/test_phase1.py -v
Must be run from the project root so relative dataset paths resolve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add code/ to path

from ingest.loaders import load_dataset
from ingest.media_index import resolve_media_path
from context.context_builder import build_context


def test_dataset_loads_expected_row_counts():
    ds = load_dataset()
    assert len(ds.messages) == 110
    assert len(ds.users) == 54


def test_media_resolves_for_image_message():
    ds = load_dataset()
    img_rows = ds.messages[ds.messages["media_type"] == "image"]
    assert not img_rows.empty
    row = img_rows.iloc[0]
    path = resolve_media_path(ds, row["media_type"], row["media_id"])
    assert path is not None
    assert path.exists()


def test_media_resolves_for_voice_message():
    ds = load_dataset()
    voice_rows = ds.messages[ds.messages["media_type"] == "voice"]
    assert not voice_rows.empty
    row = voice_rows.iloc[0]
    path = resolve_media_path(ds, row["media_type"], row["media_id"])
    assert path is not None
    assert path.exists()


def test_build_context_for_business_message():
    ds = load_dataset()
    biz_msg = ds.messages[ds.messages["conversation_type"] == "business"].iloc[0]
    ctx = build_context(ds, biz_msg["message_id"])
    assert ctx.business is not None
    assert ctx.user  # user dict is not empty


def test_build_context_for_group_message():
    ds = load_dataset()
    group_msg = ds.messages[ds.messages["conversation_type"] == "group"].iloc[0]
    ctx = build_context(ds, group_msg["message_id"])
    assert ctx.group is not None

# add to src/tests/test_phase1.py
def test_build_context_for_personal_message():
    ds = load_dataset()
    personal_msg = ds.messages[ds.messages["conversation_type"] == "personal"].iloc[0]
    ctx = build_context(ds, personal_msg["message_id"])
    assert ctx.sender_user_id is not None