"""
Builds a single, unified context object for one incoming message,
joining across user, group, business, and historical message data.
This is the object the perception + reasoning layers consume.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from ingest.loaders import Dataset
from ingest.media_index import resolve_media_path

MAX_HISTORY_ITEMS = 10


@dataclass
class MessageContext:
    message_id: str
    user_id: str
    conversation_type: str
    message_text: str
    media_type: Optional[str]
    media_id: Optional[str]
    media_path: Optional[str]
    forwarded_count: int
    created_at: str

    user: dict = field(default_factory=dict)
    group: Optional[dict] = None
    group_member: Optional[dict] = None
    business: Optional[dict] = None
    business_history: Optional[dict] = None
    sender_user_id: Optional[str] = None

    # historical evidence: list of dicts, each a past message + its event outcome
    history: list = field(default_factory=list)


def _row_to_dict(row: pd.Series) -> dict:
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


def _get_history_for_message(dataset: Dataset, msg: pd.Series, limit: int = MAX_HISTORY_ITEMS) -> list:
    """
    Finds relevant historical messages for this user, scoped to the same
    sender/group/business, most recent first. Attaches engagement outcome
    from message_events.csv when available.
    """
    mh = dataset.message_history
    user_id = msg["user_id"]

    candidates = mh[mh["user_id"] == user_id]

    if msg["conversation_type"] == "group" and pd.notna(msg.get("group_id")):
        candidates = candidates[candidates["group_id"] == msg["group_id"]]
    elif msg["conversation_type"] == "business" and pd.notna(msg.get("business_id")):
        candidates = candidates[candidates["business_id"] == msg["business_id"]]
    elif msg["conversation_type"] == "personal" and pd.notna(msg.get("sender_user_id")):
        candidates = candidates[candidates["sender_user_id"] == msg["sender_user_id"]]
    else:
        return []

    candidates = candidates.sort_values("created_at", ascending=False).head(limit)

    events = dataset.message_events
    results = []
    for _, hrow in candidates.iterrows():
        event_match = events[
            (events["user_id"] == user_id) & (events["message_id"] == hrow["message_id"])
        ]
        event = _row_to_dict(event_match.iloc[0]) if not event_match.empty else None
        results.append({"message": _row_to_dict(hrow), "event": event})

    return results


def build_context(dataset: Dataset, message_id: str) -> MessageContext:
    msg_rows = dataset.messages[dataset.messages["message_id"] == message_id]
    if msg_rows.empty:
        raise ValueError(f"message_id not found: {message_id}")
    return build_context_from_row(dataset, msg_rows.iloc[0])

def build_context_from_row(dataset: Dataset, msg: pd.Series) -> MessageContext:
    """
    Same logic as build_context, but takes a message row directly instead of
    looking it up by message_id in dataset.messages. Used for evaluating
    against sample_messages.csv, which has its own separate message_id space.
    """
    user_id = msg["user_id"]
    user_rows = dataset.users[dataset.users["user_id"] == user_id]
    user = _row_to_dict(user_rows.iloc[0]) if not user_rows.empty else {}

    group = None
    group_member = None
    if msg["conversation_type"] == "group" and pd.notna(msg.get("group_id")):
        g_rows = dataset.groups[dataset.groups["group_id"] == msg["group_id"]]
        group = _row_to_dict(g_rows.iloc[0]) if not g_rows.empty else None

        gm_rows = dataset.group_members[
            (dataset.group_members["group_id"] == msg["group_id"])
            & (dataset.group_members["user_id"] == user_id)
        ]
        group_member = _row_to_dict(gm_rows.iloc[0]) if not gm_rows.empty else None

    business = None
    business_history = None
    if msg["conversation_type"] == "business" and pd.notna(msg.get("business_id")):
        b_rows = dataset.business_accounts[dataset.business_accounts["business_id"] == msg["business_id"]]
        business = _row_to_dict(b_rows.iloc[0]) if not b_rows.empty else None

        bh_rows = dataset.user_business_history[
            (dataset.user_business_history["user_id"] == user_id)
            & (dataset.user_business_history["business_id"] == msg["business_id"])
        ]
        business_history = _row_to_dict(bh_rows.iloc[0]) if not bh_rows.empty else None

    media_path = resolve_media_path(dataset, msg.get("media_type"), msg.get("media_id"))
    history = _get_history_for_message(dataset, msg)

    return MessageContext(
        message_id=msg["message_id"],
        user_id=user_id,
        conversation_type=msg["conversation_type"],
        message_text="" if pd.isna(msg.get("message_text")) else msg["message_text"],
        media_type=None if pd.isna(msg.get("media_type")) else msg["media_type"],
        media_id=None if pd.isna(msg.get("media_id")) else msg["media_id"],
        media_path=str(media_path) if media_path else None,
        forwarded_count=int(msg["forwarded_count"]) if pd.notna(msg.get("forwarded_count")) else 0,
        created_at=str(msg.get("created_at", "")),
        user=user,
        group=group,
        group_member=group_member,
        business=business,
        business_history=business_history,
        sender_user_id=None if pd.isna(msg.get("sender_user_id")) else msg["sender_user_id"],
        history=history,
    )