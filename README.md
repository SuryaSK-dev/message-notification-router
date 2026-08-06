# Message Notification Router

An AI-powered, multimodal notification routing system for WhatsApp-style
messaging platforms. Given an incoming message (text, image, or voice note),
the system decides whether to `notify`, `digest`, or `mute` it — personalized
per user, using sender trust, group context, business relationship history,
and behavioral signals.

## Problem
See [docs/DESIGN.md](docs/DESIGN.md) for the full system design and problem framing.

## Architecture
Ingest -> Perception (VLM/ASR via Gemini) -> Context/Retrieval -> Safety Rules -> LLM Routing -> Output

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r code/requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
```

## Run

```bash
python code/pipeline.py
```

## Evaluate

```bash
python code/eval/evaluate.py
```
