# src/debug_routing.py
import sys
sys.path.insert(0, "src")
from ingest.loaders import load_dataset
from context.context_builder import build_context
from reasoning.signal_bundle import build_signal_bundle
from reasoning.router import route_message

ds = load_dataset()

# pick a few varied messages: one business, one group, one personal
samples = []
for conv_type in ["business", "group", "personal"]:
    rows = ds.messages[ds.messages["conversation_type"] == conv_type]
    if not rows.empty:
        samples.append(rows.iloc[0]["message_id"])

for msg_id in samples:
    ctx = build_context(ds, msg_id)
    bundle = build_signal_bundle(ds, ctx)
    decision = route_message(bundle, use_cache=False)
    print(f"--- {msg_id} ({ctx.conversation_type}) ---")
    print("text:", ctx.message_text[:100] if ctx.message_text else bundle.media_description[:100])
    print("action:", decision.action)
    print("message_type:", decision.message_type)
    print("reason:", decision.reason)
    print("confidence:", decision.confidence)
    print("evidence:", decision.evidence_message_ids)
    print()