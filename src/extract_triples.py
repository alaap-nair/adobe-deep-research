from src.schema import ALLOWED_RELATIONS
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print("API KEY LOADED:", OPENROUTER_API_KEY is not None)

MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

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

    for attempt in range(3):  # 🔁 retry logic
        try:
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
                },
                timeout=60
            )

            # 🚨 Handle HTTP errors first
            if response.status_code != 200:
                print(f"HTTP ERROR {response.status_code}: {response.text[:200]}")
                continue

            # 🚨 Safe JSON parsing
            try:
                result = response.json()
            except Exception:
                print("Invalid JSON response from API:")
                print(response.text[:500])
                continue

            # 🚨 Structure validation
            if "choices" not in result or not result["choices"]:
                print("Malformed response:", result)
                continue

            content = result["choices"][0]["message"].get("content")

            if not content:
                print("Empty model response")
                continue

            # clean markdown
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines)

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print("JSON parsing failed:")
                print(content[:500])
                continue

        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            continue

    # If all retries fail
    return {"triples": []}