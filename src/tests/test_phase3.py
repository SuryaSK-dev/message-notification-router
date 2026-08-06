import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.loaders import load_dataset
from context.context_builder import build_context
from reasoning.signal_bundle import build_signal_bundle


def test_bundle_for_business_message_with_no_media():
    ds = load_dataset()
    msg = ds.messages[ds.messages["conversation_type"] == "business"].iloc[0]
    ctx = build_context(ds, msg["message_id"])
    bundle = build_signal_bundle(ds, ctx)
    assert bundle.business_verified is not None
    assert bundle.text_for_reasoning  # has real message_text


def test_bundle_for_image_message_pulls_perception():
    ds = load_dataset()
    img_msg = ds.messages[ds.messages["media_type"] == "image"].iloc[0]
    ctx = build_context(ds, img_msg["message_id"])
    bundle = build_signal_bundle(ds, ctx)
    assert bundle.media_description  # non-empty, came from perception
    assert bundle.text_for_reasoning  # extracted_text or description, not empty


def test_bundle_includes_evidence_ids():
    ds = load_dataset()
    msg = ds.messages.iloc[0]
    ctx = build_context(ds, msg["message_id"])
    bundle = build_signal_bundle(ds, ctx)
    assert isinstance(bundle.evidence_message_ids, list)