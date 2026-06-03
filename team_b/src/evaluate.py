"""
evaluate.py -- Evaluation metrics for extraction quality.

Currently implements evidence grounding: checks whether each triple's
evidence field actually appears in the source text.
"""

import json
import sys
import os
from difflib import SequenceMatcher


def evidence_grounding_score(triples: list[dict], source_text: str, threshold: float = 0.8) -> dict:
    """Check how many triples have evidence grounded in the source text.

    Returns a dict with:
        - score: fraction of triples with grounded evidence
        - total: number of triples checked
        - grounded: list of indices with grounded evidence
        - ungrounded: list of (index, evidence, best_ratio) for failures
    """
    grounded = []
    ungrounded = []

    for i, triple in enumerate(triples):
        evidence = triple.get("evidence", "").strip()
        if not evidence:
            ungrounded.append((i, "(empty)", 0.0))
            continue

        # Try exact substring match first
        if evidence in source_text:
            grounded.append(i)
            continue

        # Fall back to fuzzy matching -- slide a window over source text
        best_ratio = 0.0
        ev_len = len(evidence)
        for start in range(0, len(source_text) - ev_len + 1, ev_len // 4 or 1):
            window = source_text[start : start + ev_len + 20]
            ratio = SequenceMatcher(None, evidence.lower(), window.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
            if ratio >= threshold:
                break

        if best_ratio >= threshold:
            grounded.append(i)
        else:
            ungrounded.append((i, evidence[:80], best_ratio))

    total = len(triples)
    return {
        "score": len(grounded) / total if total > 0 else 0.0,
        "total": total,
        "grounded_count": len(grounded),
        "ungrounded": [
            {"index": idx, "evidence_preview": ev, "best_similarity": round(r, 3)}
            for idx, ev, r in ungrounded
        ],
    }


def evaluate_output(output_path: str, source_path: str):
    """Run all available eval metrics on an extraction output file."""
    with open(output_path, "r") as f:
        output = json.load(f)

    with open(source_path, "r") as f:
        source_text = f.read()

    triples = output.get("triples", [])
    print(f"Evaluating {len(triples)} triples from {output_path}")
    print(f"Source text: {len(source_text)} chars from {source_path}")
    print()

    # Evidence grounding
    result = evidence_grounding_score(triples, source_text)
    print(f"Evidence Grounding: {result['grounded_count']}/{result['total']} "
          f"({result['score']:.1%})")

    if result["ungrounded"]:
        print("\nUngrounded triples:")
        for item in result["ungrounded"]:
            print(f"  Triple {item['index']}: similarity={item['best_similarity']:.3f}")
            print(f"    Evidence: {item['evidence_preview']}...")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/evaluate.py <output.json> <source.txt>")
        print("Example: python src/evaluate.py outputs/triples.json data/passage.txt")
        sys.exit(1)

    evaluate_output(sys.argv[1], sys.argv[2])
