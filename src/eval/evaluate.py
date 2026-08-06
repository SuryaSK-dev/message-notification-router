"""
Phase 7: scores the routing pipeline against sample_messages.csv (labeled
ground truth), which is used ONLY for validation — never for training or
hardcoding. Uses a separate routing cache namespace to avoid colliding
with the real messages.csv routing cache.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from ingest.loaders import load_dataset, DATASET_DIR
from context.context_builder import build_context_from_row
from reasoning.signal_bundle import build_signal_bundle
from reasoning.router import route_message


def load_sample_messages() -> pd.DataFrame:
    path = DATASET_DIR / "sample_messages.csv"
    return pd.read_csv(path)


def parse_evidence_ids(raw) -> set:
    if pd.isna(raw) or str(raw).strip().lower() == "none":
        return set()
    return set(x.strip() for x in str(raw).split(";") if x.strip())


def evaluate():
    ds = load_dataset()
    samples = load_sample_messages()

    action_correct = 0
    type_correct = 0
    evidence_overlaps = []
    mismatches = []

    for i, row in samples.iterrows():
        ctx = build_context_from_row(ds, row)
        bundle = build_signal_bundle(ds, ctx)
        decision = route_message(bundle, use_cache=True)

        expected_action = str(row["action"]).strip().lower()
        expected_type = str(row["message_type"]).strip().lower()
        expected_evidence = parse_evidence_ids(row.get("evidence_message_ids"))
        predicted_evidence = set(decision.evidence_message_ids)

        action_match = decision.action == expected_action
        type_match = decision.message_type.strip().lower() == expected_type

        if action_match:
            action_correct += 1
        if type_match:
            type_correct += 1

        if expected_evidence or predicted_evidence:
            overlap = len(expected_evidence & predicted_evidence) / max(len(expected_evidence | predicted_evidence), 1)
            evidence_overlaps.append(overlap)

        if not action_match or not type_match:
            mismatches.append({
                "message_id": row["message_id"],
                "expected_action": expected_action,
                "predicted_action": decision.action,
                "expected_type": expected_type,
                "predicted_type": decision.message_type,
                "predicted_reason": decision.reason,
            })

    total = len(samples)
    print(f"=== Evaluation against {total} labeled sample messages ===\n")
    print(f"Action accuracy:       {action_correct}/{total} ({100*action_correct/total:.1f}%)")
    print(f"Message_type accuracy: {type_correct}/{total} ({100*type_correct/total:.1f}%)")
    if evidence_overlaps:
        avg_overlap = sum(evidence_overlaps) / len(evidence_overlaps)
        print(f"Avg evidence overlap:  {100*avg_overlap:.1f}% (Jaccard similarity)")

    if mismatches:
        print(f"\n=== {len(mismatches)} mismatches ===")
        for m in mismatches:
            print(f"\n{m['message_id']}")
            print(f"  action:  expected={m['expected_action']:10s} predicted={m['predicted_action']}")
            print(f"  type:    expected={m['expected_type']:10s} predicted={m['predicted_type']}")
            print(f"  reason:  {m['predicted_reason'][:150]}")


if __name__ == "__main__":
    evaluate()