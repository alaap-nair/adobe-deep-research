"""
ask.py -- CLI entrypoint for hybrid QA over Qdrant and Neo4j.

Usage:
    python3 ask.py "What does glycolisis produce"
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config as app_config
from qa_client import answer_question
from query_engine import QueryEngine

OUTPUT_DIR = os.path.join(ROOT, "outputs", "answers")


def slugify_question(question: str, max_length: int = 80) -> str:
    """Create a stable filename slug from the input question."""
    slug = question.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    if not slug:
        slug = "question"
    return slug[:max_length].rstrip("_")


def ensure_query_resources(engine: QueryEngine):
    """Fail fast if required stores or collections are unavailable."""
    if hasattr(engine, "neo4j") and hasattr(engine.neo4j, "verify_connectivity"):
        try:
            engine.neo4j.verify_connectivity()
        except Exception as exc:
            raise RuntimeError(
                "Neo4j is unavailable. Start Neo4j and ingest data before running ask.py."
            ) from exc

    required_collections = [
        getattr(app_config, "CHUNK_COLLECTION", "chunks"),
        app_config.ENTITY_COLLECTION,
        app_config.EVIDENCE_COLLECTION,
    ]
    if not hasattr(engine, "qdrant"):
        return

    for collection_name in required_collections:
        if not engine.qdrant.collection_exists(collection_name):
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' is missing. "
                "Run python3 src/run_all.py <path> first."
            )
        count = engine.qdrant.count(collection_name).count
        if count == 0:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' is empty. "
                "Run python3 src/run_all.py <path> first."
            )


def write_json(path: str, payload: dict):
    """Persist JSON output to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def generate_answer(question: str, context: dict) -> dict:
    """Bridge ask.py to the OpenRouter-backed QA client."""
    model_name = getattr(app_config, "QA_MODEL", getattr(app_config, "MODEL_NAME", ""))
    if not model_name:
        raise RuntimeError("Set QA_MODEL or MODEL_NAME in .env before running ask.py.")
    return answer_question(question=question, context=context, model=model_name)


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python3 ask.py "What does glycolisis produce"', file=sys.stderr)
        return 1

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Question must not be empty.", file=sys.stderr)
        return 1

    engine = None
    try:
        engine = QueryEngine()
        ensure_query_resources(engine)
        context = engine.retrieve_context(
            question,
            top_k_chunks=getattr(app_config, "QA_TOP_K_CHUNKS", 5),
            top_k_entities=getattr(app_config, "QA_TOP_K_ENTITIES", 3),
            top_k_evidence=getattr(app_config, "QA_TOP_K_EVIDENCE", 5),
            hops=getattr(app_config, "QA_GRAPH_HOPS", 2),
        )
        result = generate_answer(question, context)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.close()

    slug = slugify_question(question)
    answer_path = os.path.join(OUTPUT_DIR, f"{slug}.json")
    trace_path = os.path.join(OUTPUT_DIR, f"{slug}_trace.json")

    trace_payload = {
        "question": question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_context": context,
        "result": result,
    }

    write_json(answer_path, result)
    write_json(trace_path, trace_payload)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
