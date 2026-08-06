"""
The single unified object that Phase 5's LLM router will consume.
Merges MessageContext (Phase 1) with PerceptionResult (Phase 2) into
one flat, complete picture of a message ready for a routing decision.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

from context.context_builder import MessageContext
from perception.schemas import PerceptionResult
from perception.analyzer import analyze
from ingest.media_index import resolve_media_path
from ingest.loaders import Dataset


@dataclass
class SignalBundle:
    message_id: str
    user_id: str
    conversation_type: str
    created_at: str
    forwarded_count: int

    # unified content — message_text OR perception's extracted_text, whichever applies
    text_for_reasoning: str
    media_type: Optional[str]
    media_description: str  # empty if no media

    # risk flags (False/default if no media)
    contains_payment_request: bool
    contains_urgency_language: bool
    contains_sensitive_document: bool
    media_brand_mentioned: str

    # user behavior profile
    user_do_not_disturb_window: Optional[str]
    user_messages_opened_30d: Optional[int]
    user_messages_replied_30d: Optional[int]
    user_notifications_dismissed_30d: Optional[int]
    user_messages_reported_30d: Optional[int]

    # sender trust signals — populated depending on conversation_type
    group_type: Optional[str] = None
    group_muted_by_user: Optional[bool] = None
    group_member_role: Optional[str] = None

    business_verified: Optional[bool] = None
    business_account_age_days: Optional[int] = None
    business_user_reports_30d: Optional[int] = None
    business_domain_matches_official: Optional[bool] = None
    user_business_relationship: Optional[str] = None  # why_user_knows_account
    user_allows_promotions: Optional[bool] = None

    sender_user_id: Optional[str] = None

    # evidence for output.csv — real historical message IDs + their outcomes
    evidence_message_ids: list = field(default_factory=list)
    evidence_summary: list = field(default_factory=list)  # human-readable outcomes

    def to_dict(self) -> dict:
        return asdict(self)


def build_signal_bundle(dataset: Dataset, ctx: MessageContext) -> SignalBundle:
    # --- perception merge ---
    media_description = ""
    contains_payment_request = False
    contains_urgency_language = False
    contains_sensitive_document = False
    media_brand_mentioned = ""
    text_for_reasoning = ctx.message_text

    if ctx.media_type in ("image", "voice") and ctx.media_id:
        path = resolve_media_path(dataset, ctx.media_type, ctx.media_id)
        result: PerceptionResult = analyze(ctx.media_id, ctx.media_type, path)
        media_description = result.raw_description
        contains_payment_request = result.contains_payment_request
        contains_urgency_language = result.contains_urgency_language
        contains_sensitive_document = result.contains_sensitive_document
        media_brand_mentioned = result.brand_or_sender_mentioned
        # for image/voice messages, message_text is empty — use extracted content instead
        text_for_reasoning = result.extracted_text or result.raw_description

    # --- group trust signals ---
    group_type = ctx.group["group_type"] if ctx.group else None
    group_muted_by_user = bool(ctx.group_member["group_muted_by_user"]) if ctx.group_member else None
    group_member_role = ctx.group_member["role"] if ctx.group_member else None

    # --- business trust signals ---
    business_verified = bool(ctx.business["verified"]) if ctx.business else None
    business_account_age_days = ctx.business["account_age_days"] if ctx.business else None
    business_user_reports_30d = ctx.business["user_reports_30d"] if ctx.business else None
    business_domain_matches_official = None
    if ctx.business:
        business_domain_matches_official = (
            ctx.business.get("official_domain") == ctx.business.get("domain_used_by_sender")
        )
    user_business_relationship = ctx.business_history["why_user_knows_account"] if ctx.business_history else None
    user_allows_promotions = bool(ctx.business_history["allows_promotions"]) if ctx.business_history else None

    # --- evidence from history ---
    evidence_ids = [h["message"]["message_id"] for h in ctx.history] if ctx.history else []
    evidence_summary = []
    for h in ctx.history:
        ev = h.get("event")
        if ev:
            evidence_summary.append(
                f"{h['message']['message_id']}: opened={bool(ev['message_opened'])}, "
                f"replied={bool(ev['message_replied'])}, "
                f"dismissed={bool(ev['notification_dismissed'])}, "
                f"reported={bool(ev['message_reported'])}"
            )

    return SignalBundle(
        message_id=ctx.message_id,
        user_id=ctx.user_id,
        conversation_type=ctx.conversation_type,
        created_at=ctx.created_at,
        forwarded_count=ctx.forwarded_count,
        text_for_reasoning=text_for_reasoning,
        media_type=ctx.media_type,
        media_description=media_description,
        contains_payment_request=contains_payment_request,
        contains_urgency_language=contains_urgency_language,
        contains_sensitive_document=contains_sensitive_document,
        media_brand_mentioned=media_brand_mentioned,
        user_do_not_disturb_window=ctx.user.get("do_not_disturb_window"),
        user_messages_opened_30d=ctx.user.get("messages_opened_30d"),
        user_messages_replied_30d=ctx.user.get("messages_replied_30d"),
        user_notifications_dismissed_30d=ctx.user.get("notifications_dismissed_30d"),
        user_messages_reported_30d=ctx.user.get("messages_reported_30d"),
        group_type=group_type,
        group_muted_by_user=group_muted_by_user,
        group_member_role=group_member_role,
        business_verified=business_verified,
        business_account_age_days=business_account_age_days,
        business_user_reports_30d=business_user_reports_30d,
        business_domain_matches_official=business_domain_matches_official,
        user_business_relationship=user_business_relationship,
        user_allows_promotions=user_allows_promotions,
        sender_user_id=ctx.sender_user_id,
        evidence_message_ids=evidence_ids,
        evidence_summary=evidence_summary,
    )