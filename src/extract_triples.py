from src.schema import ALLOWED_RELATIONS, canonicalize_relation
import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _entity_lookup(entities):
    by_norm = {}
    for entity in entities or []:
        cleaned = _normalize_ws(str(entity))
        if cleaned:
            by_norm[cleaned.lower()] = cleaned
    return by_norm


def _canonicalize_entity(raw: str, entity_by_norm: dict[str, str]) -> str:
    cleaned = _normalize_ws(raw)
    if not cleaned:
        return ""
    return entity_by_norm.get(cleaned.lower(), cleaned)


def _grounded_in_text(text: str, evidence: str, head: str, tail: str) -> bool:
    """
    Require grounding in source text.
    Preferred: evidence span appears in text.
    Fallback: both head and tail appear in text.
    """
    text_norm = _normalize_ws(text).lower()
    ev_norm = _normalize_ws(evidence).lower()
    head_norm = _normalize_ws(head).lower()
    tail_norm = _normalize_ws(tail).lower()
    if ev_norm and ev_norm in text_norm:
        return True
    return bool(head_norm and tail_norm and head_norm in text_norm and tail_norm in text_norm)


def _clean_and_canonicalize_triples(payload: dict, text: str, entities) -> dict:
    triples = payload.get("triples", []) if isinstance(payload, dict) else []
    if not isinstance(triples, list):
        return {"triples": []}
    entity_by_norm = _entity_lookup(entities)
    cleaned = []
    seen = set()
    for item in triples:
        if not isinstance(item, dict):
            continue
        head = _canonicalize_entity(item.get("head", ""), entity_by_norm)
        tail = _canonicalize_entity(item.get("tail", ""), entity_by_norm)
        relation = canonicalize_relation(str(item.get("relation", "")).strip())
        evidence = _normalize_ws(str(item.get("evidence", "")))

        if not head or not tail or relation not in ALLOWED_RELATIONS:
            continue
        if not _grounded_in_text(text, evidence, head, tail):
            continue

        key = (head.lower(), relation, tail.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "evidence": evidence,
            }
        )
    return {"triples": cleaned}


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
Do not invent entity names outside the provided entity list.
Evidence must be an exact quote from the input text.
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
                parsed = json.loads(content)
                return _clean_and_canonicalize_triples(parsed, text, entities)
            except json.JSONDecodeError:
                print("JSON parsing failed:")
                print(content[:500])
                continue

        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            continue

    # If all retries fail
    return {"triples": []}