"""
Structured output shape for perception results — same shape for
image and audio, so the reasoning layer doesn't need to branch on media type.
"""
from dataclasses import dataclass, asdict


@dataclass
class PerceptionResult:
    media_id: str
    media_type: str  # 'image' or 'voice'
    raw_description: str
    extracted_text: str
    contains_payment_request: bool
    contains_urgency_language: bool
    contains_sensitive_document: bool
    brand_or_sender_mentioned: str  # empty string if none
    model_confidence_note: str      # model's own caveat, if any; empty string otherwise

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PerceptionResult":
        return PerceptionResult(**d)