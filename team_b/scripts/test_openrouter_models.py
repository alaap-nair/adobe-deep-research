"""Probe OpenRouter for valid model IDs. Diagnostic only — does not affect state."""
import os
import sys
sys.path.insert(0, "src")
import config  # loads .env via dotenv
import requests

key = os.getenv("OPENROUTER_API_KEY")
print(f"Key set: {bool(key)}")

for model in [
    "qwen/qwen2.5-72b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "microsoft/phi-4",
]:
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://github.com/alaap-nair/adobe-deep-research",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'ok' in one word."}],
                "max_tokens": 16,
            },
            timeout=30,
        )
        print(f"{model}: HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"  body: {r.text[:300]}")
    except Exception as e:
        print(f"{model}: EXCEPTION {e}")
