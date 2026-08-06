import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.loaders import load_dataset
from context.context_builder import build_context
from reasoning.signal_bundle import build_signal_bundle
from reasoning.router import route_message, VALID_ACTIONS


def test_known_scam_message_routed_to_mute_via_safety_layer():
    """msg_085 should be muted WITHOUT hitting the LLM — safety rules catch it first."""
    ds = load_dataset()
    ctx = build_context(ds, "msg_085")
    bundle = build_signal_bundle(ds, ctx)
    decision = route_message(bundle, use_cache=False)
    assert decision.action == "mute"
    assert decision.message_type == "scam"
    assert decision.confidence >= 0.9


def test_decision_has_valid_action_for_arbitrary_message():
    ds = load_dataset()
    msg_id = ds.messages.iloc[5]["message_id"]
    ctx = build_context(ds, msg_id)
    bundle = build_signal_bundle(ds, ctx)
    decision = route_message(bundle)
    assert decision.action in VALID_ACTIONS
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reason  # non-empty


def test_decision_caching_avoids_repeat_llm_call():
    ds = load_dataset()
    msg_id = ds.messages.iloc[5]["message_id"]
    ctx = build_context(ds, msg_id)
    bundle = build_signal_bundle(ds, ctx)
    first = route_message(bundle)
    second = route_message(bundle)
    assert first.action == second.action
    assert first.reason == second.reason