"""
Deterministic safety overrides. These run BEFORE the LLM call and can force
a `mute` decision regardless of what the LLM would otherwise decide.
Each rule is independently testable and returns a reason string if triggered.
"""
from dataclasses import dataclass
from typing import Optional
from reasoning.signal_bundle import SignalBundle


@dataclass
class SafetyVerdict:
    triggered: bool
    reason: str = ""
    message_type: str = ""  # 'scam' or 'spam', suggested if triggered


def _rule_new_unverified_business_payment_urgency(b: SignalBundle) -> Optional[SafetyVerdict]:
    """Classic phishing pattern: new/unverified business + payment ask + urgency pressure."""
    if b.conversation_type != "business":
        return None
    if b.business_verified is False and b.contains_payment_request and b.contains_urgency_language:
        return SafetyVerdict(
            True,
            "Unverified business account requesting payment/OTP with urgency pressure — classic phishing pattern.",
            "scam",
        )
    return None


def _rule_high_report_count(b: SignalBundle) -> Optional[SafetyVerdict]:
    """Business heavily reported by other users in the last 30 days."""
    if b.business_user_reports_30d is not None and b.business_user_reports_30d >= 15:
        return SafetyVerdict(
            True,
            f"Business has {b.business_user_reports_30d} user reports in the last 30 days — high-risk sender.",
            "scam",
        )
    return None


def _rule_domain_mismatch_with_payment(b: SignalBundle) -> Optional[SafetyVerdict]:
    """Sender domain doesn't match the business's official domain, and the message asks for payment."""
    if b.business_domain_matches_official is False and b.contains_payment_request:
        return SafetyVerdict(
            True,
            "Sending domain does not match the business's official domain, combined with a payment request.",
            "scam",
        )
    return None


def _rule_sensitive_document_from_unknown_sender(b: SignalBundle) -> Optional[SafetyVerdict]:
    """A sensitive document (bank statement, ID, etc.) forwarded with no established relationship."""
    if b.contains_sensitive_document and b.conversation_type == "business" and not b.user_business_relationship:
        return SafetyVerdict(
            True,
            "Sensitive personal/financial document shared with no established user-business relationship.",
            "scam",
        )
    return None


def _rule_new_business_new_promotion_opted_out(b: SignalBundle) -> Optional[SafetyVerdict]:
    """User explicitly opted out of promotions from this business — repeat contact anyway is spam."""
    if b.user_allows_promotions is False and not (b.contains_payment_request or b.contains_urgency_language):
        return SafetyVerdict(
            True,
            "User has opted out of promotions from this business.",
            "spam",
        )
    return None


# Ordered list — first match wins, but we evaluate all and take the highest-severity one.
ALL_RULES = [
    _rule_new_unverified_business_payment_urgency,
    _rule_domain_mismatch_with_payment,
    _rule_high_report_count,
    _rule_sensitive_document_from_unknown_sender,
    _rule_new_business_new_promotion_opted_out,
]


def evaluate_safety(bundle: SignalBundle) -> SafetyVerdict:
    """
    Runs all rules. Returns the first triggered verdict with message_type 'scam'
    (highest severity) if any scam rule fires; otherwise the first 'spam' rule;
    otherwise a non-triggered verdict.
    """
    triggered = [r(bundle) for r in ALL_RULES]
    triggered = [v for v in triggered if v is not None]

    if not triggered:
        return SafetyVerdict(False)

    scam_hits = [v for v in triggered if v.message_type == "scam"]
    if scam_hits:
        return scam_hits[0]

    return triggered[0]