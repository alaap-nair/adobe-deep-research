import json
import os

import requests
from dotenv import load_dotenv

# Load environment from this module's directory so imports work
# regardless of the current working directory.
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

ALLOWED_RELATIONS = [
    "occurs_in",
    "produces",
    "converts_to",
    "uses",
    "requires",
    "inhibits",
    "activates",
    "transports_to",
    "donates_electrons_to",
    "accepts_electrons_from",
]


def extract_triples(text, entities):
    prompt = f"""
You are building a biological knowledge graph.
Entities: {entities}
Text: {text}
Only use the following relation types: {ALLOWED_RELATIONS}
Return ONLY valid JSON. No explanations. No markdown.
If unsure, return an empty list.
Expected format:
{{
  "triples": [
    {{"head": "...", "relation": "...", "tail": "...", "evidence": "..."}}
  ]
}}
"""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL_NAME, "temperature": 0, "top_p": 1, "messages": [{"role": "user", "content": prompt}]},
    )
    if response.status_code == 429:
        print("RATE LIMIT - skipping")
        return {"triples": []}
    result = response.json()
    if "choices" not in result:
        print("ERROR:", result)
        return {"triples": []}
    content = result["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        content = "\n".join(lines)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("JSON parse failed:", content[:300])
        return {"triples": []}
