# src/debug_context.py
import sys
sys.path.insert(0, "src")
from ingest.loaders import load_dataset
from context.context_builder import build_context

ds = load_dataset()
sample_id = ds.messages.iloc[0]["message_id"]
ctx = build_context(ds, sample_id)

print("message_id:", ctx.message_id)
print("text:", ctx.message_text[:80])
print("conversation_type:", ctx.conversation_type)
print("user:", ctx.user)
print("group:", ctx.group)
print("business:", ctx.business)
print("history items found:", len(ctx.history))
if ctx.history:
    print("sample history entry:", ctx.history[0])