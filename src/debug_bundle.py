# src/debug_bundle.py
import sys
sys.path.insert(0, "src")
from ingest.loaders import load_dataset
from context.context_builder import build_context
from reasoning.signal_bundle import build_signal_bundle

ds = load_dataset()

# find the message that uses vn_008 (the OTP phishing voice note)
target = ds.messages[ds.messages["media_id"] == "vn_008"]
if target.empty:
    print("vn_008 not referenced by any message row — picking first voice message instead")
    target = ds.messages[ds.messages["media_type"] == "voice"]

msg_id = target.iloc[0]["message_id"]
ctx = build_context(ds, msg_id)
bundle = build_signal_bundle(ds, ctx)

for k, v in bundle.to_dict().items():
    print(f"{k}: {v}")# src/debug_bundle.py
import sys
sys.path.insert(0, "src")
from ingest.loaders import load_dataset
from context.context_builder import build_context
from reasoning.signal_bundle import build_signal_bundle

ds = load_dataset()

# find the message that uses vn_008 (the OTP phishing voice note)
target = ds.messages[ds.messages["media_id"] == "vn_008"]
if target.empty:
    print("vn_008 not referenced by any message row — picking first voice message instead")
    target = ds.messages[ds.messages["media_type"] == "voice"]

msg_id = target.iloc[0]["message_id"]
ctx = build_context(ds, msg_id)
bundle = build_signal_bundle(ds, ctx)

for k, v in bundle.to_dict().items():
    print(f"{k}: {v}")