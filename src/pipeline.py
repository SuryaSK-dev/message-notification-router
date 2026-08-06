"""
Phase 6: runs the full pipeline over every message in messages.csv
and writes dataset/output.csv in the required schema.

Resumable: routing decisions are cached per message_id (src/.cache/routing/),
so re-running after an interruption or rate limit only processes what's missing.
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, "src")

from ingest.loaders import load_dataset
from context.context_builder import build_context
from reasoning.signal_bundle import build_signal_bundle
from reasoning.router import route_message, _get_cached

OUTPUT_PATH = Path("dataset/output.csv")
PACE_SECONDS = 4  # delay between LLM-bound calls to respect free-tier rate limits


def run_pipeline():
    ds = load_dataset()
    message_ids = ds.messages["message_id"].tolist()
    total = len(message_ids)

    print(f"Routing {total} messages...\n")

    results = []
    newly_called_llm = 0
    failures = []

    for i, message_id in enumerate(message_ids, start=1):
        already_cached = _get_cached(message_id) is not None

        try:
            ctx = build_context(ds, message_id)
            bundle = build_signal_bundle(ds, ctx)
            decision = route_message(bundle, use_cache=True)
            results.append(decision)

            status = "cached" if already_cached else "LIVE"
            print(f"[{i}/{total}] {message_id}: {decision.action} ({decision.message_type}, "
                  f"conf={decision.confidence}) [{status}]")

            if not already_cached:
                newly_called_llm += 1
                time.sleep(PACE_SECONDS)  # only pace when we actually hit the API

        except Exception as e:
            print(f"[{i}/{total}] {message_id}: FAILED — {e}")
            failures.append((message_id, str(e)))

    write_output_csv(results)

    print(f"\nDone. {len(results)}/{total} routed successfully.")
    if failures:
        print(f"{len(failures)} messages failed entirely and are NOT in output.csv:")
        for mid, err in failures:
            print(f"  {mid}: {err}")


def write_output_csv(decisions):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"])
        for d in decisions:
            evidence = ";".join(d.evidence_message_ids) if d.evidence_message_ids else "none"
            writer.writerow([d.message_id, d.action, d.message_type, d.reason, d.confidence, evidence])
    print(f"\nWrote {len(decisions)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()