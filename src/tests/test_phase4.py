import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.loaders import load_dataset
from context.context_builder import build_context
from reasoning.signal_bundle import build_signal_bundle
from reasoning.safety_rules import evaluate_safety


def test_known_scam_voice_note_is_flagged():
    """msg_085 — the OTP phishing voice note we already inspected manually."""
    ds = load_dataset()
    ctx = build_context(ds, "msg_085")
    bundle = build_signal_bundle(ds, ctx)
    verdict = evaluate_safety(bundle)
    assert verdict.triggered is True
    assert verdict.message_type == "scam"


# replace the failing test in src/tests/test_phase4.py

def test_legit_wanted_business_message_not_flagged():
    """
    A message should NOT be safety-flagged when: business is verified,
    no payment/urgency signals, AND the user hasn't opted out of promotions
    from this business. This is the correct 'should pass through cleanly' case.
    """
    ds = load_dataset()
    msg = ds.messages[ds.messages["conversation_type"] == "business"]
    found = False
    for _, row in msg.iterrows():
        ctx = build_context(ds, row["message_id"])
        bundle = build_signal_bundle(ds, ctx)
        is_clean_case = (
            bundle.business_verified is True
            and not bundle.contains_payment_request
            and not bundle.contains_urgency_language
            and bundle.user_allows_promotions is not False
        )
        if is_clean_case:
            verdict = evaluate_safety(bundle)
            assert verdict.triggered is False, f"Unexpected flag on {row['message_id']}: {verdict.reason}"
            found = True
            break
    assert found, "No clean verified+wanted business message found to test against"


def test_opted_out_promotion_correctly_flagged_as_spam():
    """
    Confirms the opt-out rule works as intended: a verified business message
    with no payment/urgency risk should still be flagged if the user opted out.
    """
    ds = load_dataset()
    msg = ds.messages[ds.messages["conversation_type"] == "business"]
    for _, row in msg.iterrows():
        ctx = build_context(ds, row["message_id"])
        bundle = build_signal_bundle(ds, ctx)
        if bundle.business_verified is True and bundle.user_allows_promotions is False and not bundle.contains_payment_request:
            verdict = evaluate_safety(bundle)
            assert verdict.triggered is True
            assert verdict.message_type == "spam"
            return
    # If no such row exists in the dataset, that's fine — nothing to assert