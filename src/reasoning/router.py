"""
Phase 5: the actual routing decision. Combines the deterministic safety
verdict (Phase 4) with an LLM reasoning call (Gemini) over the signal
bundle (Phase 3) to produce the final notify/digest/mute decision.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from google.genai import types
from perception.gemini_client import get_client
from config.settings import TEXT_MODEL
from reasoning.signal_bundle import SignalBundle
from reasoning.safety_rules import SafetyVerdict, evaluate_safety

VALID_ACTIONS = {"notify", "digest", "mute"}
CACHE_DIR = Path("src/.cache/routing")


@dataclass
class RoutingDecision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list

    def to_dict(self) -> dict:
        return asdict(self)


ROUTING_PROMPT_TEMPLATE = """You are a notification routing system for a WhatsApp-style messaging platform.
Decide how ONE incoming message should be handled for the SPECIFIC receiving user described below.

Respond with ONLY a JSON object, no markdown fences, no commentary, matching exactly this schema:
{{
  "action": "notify" | "digest" | "mute",
  "message_type": one short label, e.g. "urgent", "personal", "promotion", "event", "reminder", "scam", "spam", "informational",
  "reason": "one or two sentences explaining the decision, grounded in the specific signals below",
  "confidence": a number between 0 and 1,
  "relevant_evidence_message_ids": a list of message_ids from the evidence provided below that most directly support this decision, or an empty list if none are directly relevant
}}

DECISION RULES:
- notify: important enough to interrupt the user right now (urgent, time-sensitive, direct personal relevance, or a mention in an otherwise-muted context that still matters).
- digest: useful but not urgent — can wait and be shown later (routine updates, non-urgent promotions the user has engaged with before, general group activity).
- mute: low-value, repetitive, unwanted, suspicious, or unsafe. Also use mute if the group is muted by the user AND this message does not contain a direct, urgent, personal reason to override that mute.
- Personalize using the user's actual behavior (open/reply/dismiss/report rates) and their history with this specific sender — the same content can warrant different actions for different users.
- A muted group should stay muted UNLESS there is a clear urgent/personal signal that justifies overriding the mute (e.g. direct mention, emergency).
- Weigh business trust signals (verified status, account age, domain match, report count) — but you do not need to re-derive scam judgments already caught by upstream safety rules; focus on legitimate-but-varying-relevance cases.

MESSAGE AND USER CONTEXT:
{context_json}

Respond with the JSON object only."""


def _cache_path(message_id: str) -> Path:
    return CACHE_DIR / f"{message_id}.json"


def _get_cached(message_id: str) -> RoutingDecision | None:
    path = _cache_path(message_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return RoutingDecision(**json.load(f))


def _set_cached(decision: RoutingDecision) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(decision.message_id), "w", encoding="utf-8") as f:
        json.dump(decision.to_dict(), f, indent=2)


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def _call_llm_router(bundle: SignalBundle, max_retries: int = 3) -> RoutingDecision:
    client = get_client()
    context_json = json.dumps(bundle.to_dict(), indent=2, default=str)
    prompt = ROUTING_PROMPT_TEMPLATE.format(context_json=context_json)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[types.Part.from_text(text=prompt)],
            )
            raw = _parse_json_response(response.text)
            break
        except Exception as e:
            last_error = e
            is_retryable = "429" in str(e) or "503" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_retryable and attempt < max_retries:
                wait = 20 * attempt  # rate limits need longer waits than server errors
                print(f"  rate limited (attempt {attempt}/{max_retries}), waiting {wait}s")
                time.sleep(wait)
                continue
            raise last_error
    else:
        raise last_error

    action = raw.get("action", "digest")
    if action not in VALID_ACTIONS:
        action = "digest"

    confidence = float(raw.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return RoutingDecision(
        message_id=bundle.message_id,
        action=action,
        message_type=raw.get("message_type", "informational"),
        reason=raw.get("reason", ""),
        confidence=confidence,
        evidence_message_ids=raw.get("relevant_evidence_message_ids", []),
    )

def route_message(bundle: SignalBundle, use_cache: bool = True) -> RoutingDecision:
    """
    Main entry point. Runs safety rules first (deterministic override);
    only calls the LLM if no safety rule triggered.
    """
    if use_cache:
        cached = _get_cached(bundle.message_id)
        if cached is not None:
            return cached

    safety_verdict: SafetyVerdict = evaluate_safety(bundle)

    if safety_verdict.triggered:
        decision = RoutingDecision(
            message_id=bundle.message_id,
            action="mute",
            message_type=safety_verdict.message_type,
            reason=safety_verdict.reason,
            confidence=0.95,
            evidence_message_ids=[],
        )
    else:
        try:
            decision = _call_llm_router(bundle)
        except Exception as e:
            # never crash the pipeline on one bad LLM call — safe fallback
            decision = RoutingDecision(
                message_id=bundle.message_id,
                action="digest",
                message_type="informational",
                reason=f"LLM routing failed, defaulted to digest: {e}",
                confidence=0.3,
                evidence_message_ids=[],
            )

    _set_cached(decision)
    return decision