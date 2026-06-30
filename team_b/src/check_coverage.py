"""
check_coverage.py -- W3B coverage verification.

For each question in data/ground_truth.json, extract candidate entity phrases
(reusing query_engine._extract_keywords), look them up in the live KG via
QueryEngine.resolve_query_entities (keyword + fuzzy + semantic fallback),
and report:

  - Per-question coverage: did at least one ground-truth entity resolve?
  - Aggregate coverage percentage.
  - List of questions with no coverage (likely out-of-corpus or missing nodes).

Usage:
    python src/check_coverage.py data/ground_truth.json
    python src/check_coverage.py data/ground_truth.json --json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from query_engine import QueryEngine


def per_question_coverage(engine: QueryEngine, question: str) -> dict:
    resolved = engine.resolve_query_entities(question, top_k=5)
    return {
        "covered": bool(resolved),
        "matched": [
            {
                "entity_id": r.get("entity_id"),
                "name": r.get("name"),
                "matched_text": r.get("matched_text"),
                "match_type": r.get("match_type"),
                "score": r.get("score"),
            }
            for r in resolved
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", help="Path to data/ground_truth.json")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    with open(args.ground_truth, "r", encoding="utf-8") as handle:
        rows = json.load(handle)

    engine = QueryEngine()
    try:
        results = []
        for row in rows:
            cov = per_question_coverage(engine, row["question"])
            results.append({"id": row["id"], "question": row["question"], **cov})
    finally:
        engine.close()

    covered = sum(1 for r in results if r["covered"])
    total = len(results)
    pct = (covered / total * 100) if total else 0.0

    if args.json:
        print(json.dumps({"coverage_pct": pct, "covered": covered, "total": total, "results": results}, indent=2))
        return 0

    print(f"Coverage: {covered}/{total} questions have >=1 ground-truth entity in graph ({pct:.1f}%)")
    print()
    missing = [r for r in results if not r["covered"]]
    if missing:
        print("Missing coverage:")
        for r in missing:
            print(f"  {r['id']}: {r['question']}")
        print()
    print("Per-question matched entities:")
    for r in results:
        marker = "OK " if r["covered"] else "-- "
        names = ", ".join(m["name"] for m in r["matched"]) or "(none)"
        print(f"  {marker}{r['id']}: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
