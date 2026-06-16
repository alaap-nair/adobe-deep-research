"""
check_coverage.py -- Verify entity coverage of questions against the knowledge graph.

Checks whether the entities mentioned in the ground truth questions exist
as nodes in the extracted triples. Reports coverage percentage.

Usage:
    python src/check_coverage.py
    python src/check_coverage.py data/ground_truth.json
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_schema import normalize_name, entity_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_graph_entities(triple_files: list[str] | None = None) -> dict[str, set[str]]:
    """Load all entity IDs from triple output files.

    Returns a dict mapping entity_id -> set of original surface forms.
    """
    if triple_files is None:
        import glob
        triple_files = sorted(glob.glob(os.path.join(ROOT, "outputs", "triples_*.json")))

    entities: dict[str, set[str]] = {}
    for path in triple_files:
        with open(path) as f:
            data = json.load(f)
        triples = data.get("triples", data) if isinstance(data, dict) else data
        for t in triples:
            for field in ["head", "tail"]:
                name = t.get(field, "").strip()
                if name:
                    eid = entity_id(name)
                    if eid not in entities:
                        entities[eid] = set()
                    entities[eid].add(name.lower())
    return entities


def extract_key_terms(question: str, ground_truth: str) -> list[str]:
    """Extract likely biological entity terms from a question and its ground truth answer.

    Uses a curated list of biology terms + regex for acronyms. Does NOT
    extract question phrases or sentence fragments.
    """
    combined = f"{question} {ground_truth}"

    # Known biology terms to look for (multi-word first for greedy matching)
    bio_terms = [
        # Multi-word terms
        "gene expression", "protein synthesis", "guide RNA", "amino acid",
        "gel electrophoresis", "restriction enzyme", "electron transport chain",
        "cell membrane", "double helix", "hydroxyl group", "genetic information",
        "DNA replication", "DNA sequencing",
        # Single-word / acronyms
        "DNA", "RNA", "mRNA", "tRNA", "rRNA", "CRISPR", "Cas9", "protein",
        "gene", "genome", "genetics", "codon", "transcription", "translation",
        "replication", "ribosome", "nucleotide", "thymine",
        "uracil", "adenine", "guanine", "cytosine", "helix", "hydroxyl",
        "enzyme", "polymerase", "promoter", "exon", "intron", "plasmid",
        "PCR", "glycolysis", "ATP", "mitochondria", "chromosome", "histone",
        "telomere", "mutation", "splicing", "polypeptide",
    ]

    found = []
    combined_lower = combined.lower()
    for term in bio_terms:
        if term.lower() in combined_lower and term not in found:
            found.append(term)

    return found


def check_coverage(questions_path: str | None = None):
    """Check entity coverage for all questions in the ground truth file."""
    if questions_path is None:
        questions_path = os.path.join(ROOT, "data", "ground_truth.json")

    with open(questions_path) as f:
        questions = json.load(f)

    graph_entities = load_graph_entities()
    all_entity_names = set()
    for names in graph_entities.values():
        all_entity_names.update(names)

    print(f"Graph has {len(graph_entities)} unique entity IDs")
    print(f"Checking coverage for {len(questions)} questions\n")

    total_terms = 0
    covered_terms = 0
    per_question = []

    for q in questions:
        question = q["question"]
        ground_truth = q["ground_truth"]
        key_terms = extract_key_terms(question, ground_truth)

        found = []
        missing = []
        for term in key_terms:
            eid = entity_id(term)
            # Check exact ID match or substring match in entity names
            if eid in graph_entities:
                found.append(term)
            elif any(term.lower() in name for name in all_entity_names):
                found.append(term)
            else:
                missing.append(term)

        total = len(key_terms)
        total_terms += total
        covered_terms += len(found)
        coverage = len(found) / total * 100 if total > 0 else 0

        per_question.append({
            "question": question,
            "key_terms": key_terms,
            "found": found,
            "missing": missing,
            "coverage": coverage,
        })

        status = "OK" if coverage >= 80 else "GAP"
        print(f"[{status}] {question[:60]:<60} {coverage:5.1f}%")
        if missing:
            print(f"       Missing: {', '.join(missing)}")

    overall = covered_terms / total_terms * 100 if total_terms > 0 else 0
    print(f"\n{'='*70}")
    print(f"Overall coverage: {covered_terms}/{total_terms} terms ({overall:.1f}%)")
    print(f"{'='*70}")

    # Save results
    results = {
        "total_terms": total_terms,
        "covered_terms": covered_terms,
        "overall_coverage_pct": round(overall, 2),
        "per_question": per_question,
    }
    out_path = os.path.join(ROOT, "outputs", "coverage_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to {out_path}")

    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    check_coverage(path)
