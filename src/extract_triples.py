import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

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
    "accepts_electrons_from"
]

def extract_triples(text, entities):

    prompt = f"""
You are building a biological knowledge graph.

Entities:
{entities}

Text:
{text}

Only use the following relation types:
{ALLOWED_RELATIONS}

Return ONLY valid JSON.
Do not include explanations.
Do not include markdown.
If unsure, return an empty list.

Expected format:
{{
  "triples": [
    {{
      "head": "...",
      "relation": "...",
      "tail": "...",
      "evidence": "..."
    }}
  ]
}}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL_NAME,
            "temperature": 0,
            "top_p": 1,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    if response.status_code == 429:
        try:
            err = response.json()
            provider = err.get("error", {}).get("provider_name", "unknown")
            msg = err.get("error", {}).get("message", str(err))
        except Exception:
            msg = response.text
            provider = "unknown"
        print("RATE LIMIT (429): Upstream provider is temporarily limiting requests.")
        return {"triples": []}

    result = response.json()

    if "choices" not in result:
        print("ERROR:", result)
        return {"triples": []}

    content = result["choices"][0]["message"]["content"]

    # Strip markdown code fence if present (e.g. ```json ... ```)
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove trailing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("JSON parsing failed:")
        print(content[:500] + ("..." if len(content) > 500 else ""))
        return {"triples": []}