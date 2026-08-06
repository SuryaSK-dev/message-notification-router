# src/verify_output.py
import sys
sys.path.insert(0, "src")
import pandas as pd

out = pd.read_csv("dataset/output.csv")
print("Total rows:", len(out))
print()
print("Action distribution:")
print(out["action"].value_counts())
print()
print("message_type distribution:")
print(out["message_type"].value_counts())
print()
print("Any nulls in required columns?")
print(out.isnull().sum())
print()
print("Confidence range:", out["confidence"].min(), "-", out["confidence"].max())
print()
print("Any invalid actions?")
print(out[~out["action"].isin(["notify", "digest", "mute"])])