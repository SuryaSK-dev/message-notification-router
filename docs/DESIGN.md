# System Design: Message Notification Router

## 1. Problem Statement
WhatsApp-style platforms interleave high-signal messages (urgent, personal,
time-sensitive) with low-signal noise (promotions, forwards, repetitive
society notices) and active risk (scams, phishing). Flat notification
policies fail in both directions: real messages get buried, unwanted or
unsafe messages interrupt the user.

## 2. Goals
- Per-message, per-user routing decision: notify / digest / mute
- Multimodal understanding: text, image (poster/screenshot), voice note
- Personalization grounded in real behavioral + relationship signals
- Deterministic safety overrides that can't be reasoned around by the LLM
- Evidence-backed decisions (traceable to historical messages)

## 3. Non-Goals
- Not a spam/scam classifier trained from scratch (uses pretrained LLM/VLM reasoning)
- Not a production-scale system (single-user-batch dataset scope)
- Not optimizing for latency; optimizing for decision quality and explainability

## 4. Architecture
[Ingest] -> [Perception: image/audio -> text via Gemini] -> [Context/Retrieval: user, group, business, history] 
-> [Safety Rules: deterministic overrides] -> [LLM Router: structured decision] -> [Output writer]

## 5. Key Design Decisions
- Gemini (multimodal) chosen over separate OCR/ASR pipeline: single model call
  handles both image and audio understanding, reducing pipeline complexity.
- Safety overrides are deterministic, not LLM-decided: prevents prompt-level
  reasoning from overriding hard scam/risk signals.
- Evidence retrieval is required, not optional: routing decisions must cite
  real historical message IDs, making the system auditable.

## 6. Evaluation Methodology
- Labeled sample set used only for validating output format/logic, never
  hardcoded against.
- Metrics: action accuracy, message_type accuracy, evidence relevance,
  confidence calibration.

## 7. Limitations
- Small dataset (110 messages) — not representative of full production scale
- Free-tier API rate limits constrain iteration speed
- No human-in-the-loop feedback loop implemented (v1 scope)

## 8. Future Work
- Feedback loop: incorporate message_events outcomes to adjust thresholds
- Embedding-based retrieval instead of heuristic matching
- Batch/digest scheduling logic (currently per-message only)
