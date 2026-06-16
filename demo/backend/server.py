"""
server.py -- FastAPI service for the Live Personalized Graph Demo.

Endpoints (all scoped to Maya's group_id):
  GET  /episodes        -> persona + the three episodes + final question
  GET  /subgraph?upto=N -> Maya's subgraph after episode N (drives the animation)
  POST /ask             -> Generalized vs Personalized answer for the final question

What is genuinely live:
  - /ask runs a REAL Graphiti hybrid search over Maya's group_id and synthesizes
    the personalized answer from the facts it retrieves; the generalized answer
    is synthesized with no personal context. This is the demo's live "money" beat.
  - Maya's episodes are really ingested into Neo4j by seed_maya.py before the run.

What is curated (and disclosed):
  - /episodes and /subgraph serve the corpus-grounded snapshot data so the graph
    visual is clean and stable for presentation. The entities are real Ch.8
    entities; seed_maya.py shows the same content flowing through live Graphiti.

Run:  uvicorn demo.backend.server:app --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / "demo" / "web" / "public" / "snapshot.json"

app = FastAPI(title="Personalized Graph Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _snapshot() -> dict[str, Any]:
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/episodes")
def episodes() -> dict[str, Any]:
    snap = _snapshot()
    return {
        "persona": snap["persona"],
        "episodes": snap["episodes"],
        "final_question": snap["meta"]["final_question"],
    }


@app.get("/subgraph")
def subgraph(upto: int = 1) -> dict[str, Any]:
    snap = _snapshot()
    key = str(max(0, min(3, upto)))
    return snap["subgraphs"].get(key, snap["subgraphs"]["0"])


# --- live answer synthesis -------------------------------------------------

SYNTH_SYSTEM = (
    "You are a biology tutor. Answer the student's question clearly and concisely "
    "using only the supplied context. When personal context about the student is "
    "provided, explicitly connect the answer to what they have previously studied."
)


def _synthesize(question: str, personal_facts: list[str]) -> str:
    """Call the configured LLM to synthesize an answer. Provider mirrors Graphiti."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    context = (
        "Personal context retrieved from this student's knowledge graph:\n- "
        + "\n- ".join(personal_facts)
        if personal_facts
        else "No personal context is available; answer generally."
    )
    user = f"{context}\n\nQuestion: {question}"

    if anthropic_key:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model=os.getenv("SYNTH_MODEL", "claude-opus-4-8"),
            max_tokens=600,
            system=SYNTH_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    if openrouter_key:
        import requests

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("SYNTH_MODEL", os.getenv("MODEL_NAME", "openai/gpt-4o-mini")),
                "max_tokens": 600,
                "messages": [
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": user},
                ],
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    raise RuntimeError("No LLM credential for synthesis.")


@app.post("/ask")
async def ask() -> dict[str, Any]:
    snap = _snapshot()
    question = snap["meta"]["final_question"]
    group_id = os.getenv("GRAPHITI_GROUP_ID", "maya")

    # Attempt the genuinely-live path: retrieve Maya's facts from Graphiti and
    # synthesize both answers. On any failure (backend not seeded, no key, etc.)
    # fall back to the curated snapshot answers so the demo never breaks.
    try:
        from demo.backend.graphiti_setup import build_graphiti

        graphiti = build_graphiti()
        try:
            results = await graphiti.search(question, group_ids=[group_id])
            personal_facts = [r.fact for r in results][:6]
        finally:
            await graphiti.close()

        if not personal_facts:
            raise RuntimeError("No personal facts retrieved; is Maya's group seeded?")

        generalized = _synthesize(question, [])
        personalized = _synthesize(question, personal_facts)
        return {
            "question": question,
            "generalized": {
                "mode": "Generalized",
                "subtitle": "Standard KAG pipeline — no personal context",
                "answer": generalized,
                "citations": ["corpus: OpenStax Biology Ch.8 — photosynthesis"],
                "used_context": [],
            },
            "personalized": {
                "mode": "Personalized",
                "subtitle": "Graphiti user subgraph injected — grounded in Maya's own episodes (LIVE)",
                "answer": personalized,
                "citations": [f"graphiti: Maya's subgraph ({group_id})"],
                "used_context": personal_facts,
            },
        }
    except Exception as exc:  # noqa: BLE001 -- demo must always return an answer
        fallback = _snapshot()["ask"]
        fallback["_note"] = f"served from snapshot fallback ({type(exc).__name__})"
        return fallback
