"""
ui_backend.py -- Thin adapter between the chat UI and the dual-store KAG pipeline.

Exposes a single `ask()` that runs the *benchmark-winning* retrieval config
(Assignment 10 Part 2: C1 = ``BAAI/bge-reranker-v2-m3``, gated by ``USE_RERANKER``)
and returns one flat, UI-shaped dict with every section the demo needs:

    answer, citations, reasoning,
    chunks, evidence, nodes, edges,
    mode, personalization_note

The UI never touches QueryEngine / qa_client internals directly -- swap or extend
the pipeline here and the Streamlit layer is unaffected.
"""

import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # team_b/
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Load the repo-root .env (OPENROUTER_API_KEY, MODEL_NAME) explicitly so the UI
# works regardless of the directory Streamlit is launched from.
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(ROOT)
load_dotenv(os.path.join(REPO_ROOT, ".env"))
load_dotenv(os.path.join(ROOT, ".env"))  # team_b/.env overrides if present

import config as app_config
from query_engine import QueryEngine
from qa_client import answer_question
from personalization import inject_user_context


# --- engine lifecycle ---------------------------------------------------------

def build_engine() -> QueryEngine:
    """Construct a QueryEngine (loads the embedding model). Cache this in the UI."""
    return QueryEngine()


def check_resources(engine: QueryEngine) -> dict:
    """Non-fatal health probe used to show status/warnings in the UI.

    Returns {"ok": bool, "neo4j": bool, "qdrant": bool, "warnings": [str]}.
    Never raises -- the UI decides how to surface degraded states.
    """
    warnings: list[str] = []
    neo4j_ok = False
    try:
        engine.neo4j.verify_connectivity()
        neo4j_ok = True
    except Exception:
        warnings.append(
            "Neo4j is unreachable — the graph panel will be empty. "
            "Start it with `docker compose up -d neo4j` and ingest with "
            "`python src/run_all.py data/<file>`."
        )

    qdrant_ok = True
    required = [
        getattr(app_config, "CHUNK_COLLECTION", "chunks"),
        app_config.ENTITY_COLLECTION,
        app_config.EVIDENCE_COLLECTION,
    ]
    for name in required:
        try:
            if not engine.qdrant.collection_exists(name) or engine.qdrant.count(name).count == 0:
                qdrant_ok = False
                warnings.append(
                    f"Qdrant collection '{name}' is missing or empty — run "
                    "`python src/run_all.py data/<file>` to ingest."
                )
        except Exception as exc:  # pragma: no cover - defensive
            qdrant_ok = False
            warnings.append(f"Qdrant check failed for '{name}': {exc}")

    return {
        "ok": qdrant_ok,  # answers can still be generated without Neo4j
        "neo4j": neo4j_ok,
        "qdrant": qdrant_ok,
        "warnings": warnings,
    }


def _model_name() -> str:
    name = getattr(app_config, "QA_MODEL", "") or getattr(app_config, "MODEL_NAME", "")
    if not name:
        raise RuntimeError(
            "No QA model configured. Set QA_MODEL (or MODEL_NAME) in .env."
        )
    return name


# --- the one call the UI makes ------------------------------------------------

def ask(
    question: str,
    engine: QueryEngine,
    personalized: bool = False,
    user_id: str | None = None,
    use_reranker: bool = True,
    as_of: datetime | None = None,
    include_invalid: bool = False,
) -> dict:
    """Run the full retrieve -> (personalize) -> synthesize pipeline.

    Args:
        question: the user's natural-language question.
        engine: a (cached) QueryEngine.
        personalized: if True, route retrieval context through the Graphiti
            personalization seam before synthesis (currently a labeled stub).
        user_id: identifier for the personalized graph (unused by the stub).
        use_reranker: if True, run the A10 Part-2 winning config
            (bge-reranker-v2-m3) by setting USE_RERANKER=1 for this query.
        as_of: if set, restrict the graph trace to facts valid at that instant
            (bi-temporal point-in-time view). Defaults to currently-valid only.
        include_invalid: if True, include facts later retracted/invalidated.

    Returns:
        A flat, JSON-serializable dict shaped for the chat UI.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")

    # Select the benchmark-winning retrieval path (C1). USE_RERANKER is read at
    # query time inside retrieve_context(), so toggling it per-call is safe.
    os.environ["USE_RERANKER"] = "1" if use_reranker else "0"

    context = engine.retrieve_context(
        question, as_of=as_of, include_invalid=include_invalid
    )

    note = None
    if personalized:
        context, note = inject_user_context(context, user_id=user_id)

    result = answer_question(question, context, model=_model_name())

    vector_hits = context.get("vector_hits", {})
    graph_trace = context.get("graph_trace", {})
    query_analysis = context.get("query_analysis", {})

    return {
        "question": question,
        "mode": "personalized" if personalized else "generalized",
        "personalization_note": note,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "reasoning": result.get("reasoning", ""),
        "chunks": vector_hits.get("chunks", []),
        "evidence": vector_hits.get("evidence", []),
        "nodes": graph_trace.get("retrieved_nodes", []),
        "edges": graph_trace.get("traversed_edges", []),
        "resolved_entities": query_analysis.get("resolved_entities", []),
        "keywords": query_analysis.get("keywords", []),
        "retrieval_config": "bge-reranker-v2-m3 (A10 P2 winner)" if use_reranker else "baseline (no rerank)",
        "temporal_view": {
            "as_of": as_of.isoformat() if as_of else None,
            "include_invalid": include_invalid,
        },
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does glycolysis produce?"
    eng = build_engine()
    try:
        health = check_resources(eng)
        print("HEALTH:", health, "\n")
        out = ask(q, eng, use_reranker=("--no-rerank" not in sys.argv))
        import json

        print(json.dumps({k: v for k, v in out.items()}, indent=2, ensure_ascii=False)[:2000])
        print("\n--- counts ---")
        for key in ("chunks", "evidence", "nodes", "edges", "citations"):
            print(f"{key}: {len(out[key])}")
    finally:
        eng.close()
