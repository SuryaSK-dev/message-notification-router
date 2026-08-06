# src/perception_smoke_test.py
"""
Run once to populate the cache and manually eyeball results.
python src/perception_smoke_test.py
"""
import sys
sys.path.insert(0, "src")

from ingest.loaders import load_dataset
from ingest.media_index import resolve_media_path
from perception.analyzer import analyze

ds = load_dataset()

media_rows = ds.messages[ds.messages["media_type"].isin(["image", "voice"])]
print(f"Found {len(media_rows)} messages with media\n")

for _, row in media_rows.iterrows():
    path = resolve_media_path(ds, row["media_type"], row["media_id"])
    result = analyze(row["media_id"], row["media_type"], path)
    print(f"--- {row['media_id']} ({row['media_type']}) ---")
    print("description:", result.raw_description)
    print("extracted_text:", result.extracted_text[:150])
    print("payment_request:", result.contains_payment_request,
          "| urgency:", result.contains_urgency_language,
          "| sensitive_doc:", result.contains_sensitive_document)
    print("brand:", result.brand_or_sender_mentioned)
    if result.model_confidence_note:
        print("note:", result.model_confidence_note)
    print()