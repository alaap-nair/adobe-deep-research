"""
extract_triples.py -- Team 2: Schema-free entity/relation extraction via LLM.

Single-shot open extraction: the LLM reads text and extracts any entities
and relationships it finds, creating node/edge labels on the fly.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = float(os.getenv("temperature", "0"))
TOP_P = float(os.getenv("top_p", "1"))

SYSTEM_PROMPT = """You are extracting a knowledge graph from scientific text.

Read the passage and extract:
- Entities (concepts, molecules, processes, locations)
- Relationships between them

Return only structured JSON with:
- head
- relation (short verb phrase)
- tail
- evidence (exact sentence from text)

Rules:
- Relations must be concise (1–3 words)
- Do not invent facts
- Only extract relationships explicitly supported by the text
- If no relationships are found, return {"triples": []}

You must return ONLY valid JSON. Do not include explanations or markdown."""


def call_openrouter(text, max_retries=5):
    """Send text to the LLM via OpenRouter and return the raw API response."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    for attempt in range(max_retries):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/alaap-nair/adobe-deep-research",
            },
            json={
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract a knowledge graph from this text:\n\n{text}"},
                ],
            },
        )
        if response.status_code == 429:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            # If free tier is persistently rate-limited, fall back to paid route
            if ":free" in MODEL_NAME:
                paid_model = MODEL_NAME.replace(":free", "")
                print(f"Free tier exhausted. Falling back to paid model: {paid_model}")
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/alaap-nair/adobe-deep-research",
                    },
                    json={
                        "model": paid_model,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Extract a knowledge graph from this text:\n\n{text}"},
                        ],
                    },
                )
                response.raise_for_status()
                return response.json()
        response.raise_for_status()
        return response.json()


def parse_llm_json(raw_content):
    """Parse JSON from LLM output, stripping markdown fences if present."""
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]  # drop opening ```json line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return json.loads(content)


def extract_triples(text):
    """Extract triplets from text using schema-free LLM extraction.

    Returns a list of dicts with keys: head, relation, tail, evidence.
    """
    result = call_openrouter(text)

    if "choices" not in result:
        print("API error:", result)
        return []

    raw = result["choices"][0]["message"]["content"]

    try:
        data = parse_llm_json(raw)
        return data.get("triples", [])
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print("Raw response:", raw[:500])
        return []


if __name__ == "__main__":
    import sys

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "passage.txt")

    with open(path, "r") as f:
        text = f.read()

    triples = extract_triples(text)
    print("Model used:", MODEL_NAME)
    print("Number of triples:", len(triples))
    for t in triples:
        print(f"  {t['head']} --[{t['relation']}]--> {t['tail']}")
