"""
Top-level entry point for the perception layer: given a media_type + path + media_id,
returns a PerceptionResult, using the cache first and falling back to a safe default on failure.
"""
from pathlib import Path
from perception.schemas import PerceptionResult
from perception.cache import get_cached, set_cached
from perception.gemini_client import analyze_media_file


def _safe_default(media_id: str, media_type: str, error_note: str) -> PerceptionResult:
    """Used when analysis fails — never crash the pipeline on one bad media file."""
    return PerceptionResult(
        media_id=media_id,
        media_type=media_type,
        raw_description="",
        extracted_text="",
        contains_payment_request=False,
        contains_urgency_language=False,
        contains_sensitive_document=False,
        brand_or_sender_mentioned="",
        model_confidence_note=f"ANALYSIS_FAILED: {error_note}",
    )


def analyze(media_id: str, media_type: str, path: Path, use_cache: bool = True) -> PerceptionResult:
    if use_cache:
        cached = get_cached(media_id)
        if cached is not None:
            return cached

    if path is None or not path.exists():
        result = _safe_default(media_id, media_type, "file not found")
        set_cached(result)
        return result

    try:
        raw = analyze_media_file(path)
        result = PerceptionResult(
            media_id=media_id,
            media_type=media_type,
            raw_description=raw.get("raw_description", ""),
            extracted_text=raw.get("extracted_text", ""),
            contains_payment_request=bool(raw.get("contains_payment_request", False)),
            contains_urgency_language=bool(raw.get("contains_urgency_language", False)),
            contains_sensitive_document=bool(raw.get("contains_sensitive_document", False)),
            brand_or_sender_mentioned=raw.get("brand_or_sender_mentioned", ""),
            model_confidence_note=raw.get("model_confidence_note", ""),
        )
    except Exception as e:
        result = _safe_default(media_id, media_type, str(e))

    set_cached(result)
    return result
