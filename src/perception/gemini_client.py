"""
Thin wrapper around the Gemini API for multimodal perception calls.
Handles client construction, file bytes loading, and JSON response parsing.
"""
import json
import time
import mimetypes
from pathlib import Path
from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, VISION_MODEL

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


PERCEPTION_PROMPT = """You are analyzing a message attachment (image or voice note) sent on a
WhatsApp-style messaging platform. Respond with ONLY a JSON object, no markdown fences, no
commentary, matching exactly this schema:

{
  "raw_description": "one or two sentence factual description of what this is",
  "extracted_text": "all readable/spoken text, verbatim, empty string if none",
  "contains_payment_request": true or false,
  "contains_urgency_language": true or false,
  "contains_sensitive_document": true or false,
  "brand_or_sender_mentioned": "brand/company/bank name if visible or mentioned, else empty string",
  "model_confidence_note": "brief note if the content is ambiguous or hard to read, else empty string"
}

Guidance:
- contains_payment_request: true if it asks the recipient to pay, send money, share OTP/PIN, or scan a payment QR.
- contains_urgency_language: true if it pressures immediate action ("act now", "expires today", "verify immediately").
- contains_sensitive_document: true if this is a bank statement, ID card, passport, or similarly sensitive personal document.
Be factual and literal. Do not speculate beyond what is visibly/audibly present."""


def _read_bytes_and_mime(path: Path) -> tuple[bytes, str]:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(path, "rb") as f:
        return f.read(), mime_type


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def analyze_media_file(path: Path, max_retries: int = 3) -> dict:
    """
    Sends the file (image or audio) to Gemini with the perception prompt,
    returns the parsed JSON dict. Retries on transient server errors (503).
    Raises on final failure or on client-side errors — caller decides fallback.
    """
    client = get_client()
    file_bytes, mime_type = _read_bytes_and_mime(path)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    PERCEPTION_PROMPT,
                ],
            )
            return _parse_json_response(response.text)
        except Exception as e:
            last_error = e
            is_retryable = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
            if is_retryable and attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s, 8s
                print(f"  retryable error (attempt {attempt}/{max_retries}), waiting {wait}s: {e}")
                time.sleep(wait)
                continue
            raise last_error

    raise last_error