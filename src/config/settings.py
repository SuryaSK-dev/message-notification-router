import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")

# Use the "-latest" alias so we don't hardcode a model version that gets deprecated.
TEXT_MODEL = "gemini-flash-lite-latest"  # higher free-tier quota than full flash
VISION_MODEL = "gemini-flash-latest"   # same model handles images natively
AUDIO_MODEL = "gemini-flash-latest"    # same model handles audio natively

DATASET_DIR = "dataset"
OUTPUT_PATH = "dataset/output.csv"
CACHE_DIR = "code/.cache"
