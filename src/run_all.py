"""
run_all.py -- Pipeline orchestrator for Team 2 schema-free extraction.

Usage: python src/run_all.py [path/to/passage.txt]
"""

import os
import json
import sys

# Ensure sibling imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_triples import extract_triples, MODEL_NAME
from extract_entitites import extract_entities
from schema import ExtractionResult
from graph_schema import build_graph_objects
from visualize_graph import visualize_from_triples

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_passage(path=None):
    if path is None:
        path = os.path.join(ROOT, "data", "passage.txt")
    if path.lower().endswith(".pdf"):
        from parse_pdf import parse_pdf
        return parse_pdf(path)
    with open(path, "r") as f:
        return f.read()


def save_output(triples, entities, path=None):
    if path is None:
        path = os.path.join(ROOT, "outputs", "triples.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    result = ExtractionResult(
        model=MODEL_NAME,
        num_triples=len(triples),
        num_entities=len(entities),
        entities=entities,
        triples=triples,
    )
    result.validate_counts()

    with open(path, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    return path


def main():
    # 1. Load passage (optional CLI arg)
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    text = load_passage(input_path)
    print(f"Loaded passage ({len(text)} chars)")

    # 2. Extract triples (single-shot schema-free LLM call)
    print("Extracting triples via OpenRouter...")
    triples = extract_triples(text)

    # 3. Derive entities from triples
    entities = extract_entities(triples)

    # 4. Required output
    print("Model used:", MODEL_NAME)
    print("Number of triples:", len(triples))

    # 5. Display results
    print(f"Unique entities: {len(entities)}")
    for t in triples:
        print(f"  {t.get('head')} --[{t.get('relation')}]--> {t.get('tail')}")

    # 6. Save to JSON (derive output name from input)
    if input_path:
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = save_output(triples, entities, os.path.join(ROOT, "outputs", f"triples_{base}.json"))
    else:
        out_path = save_output(triples, entities)
    print(f"\nSaved to {out_path}")

    # 7. Build structured graph objects
    entities_structured, relations_structured = build_graph_objects(triples)

    # 8. Build knowledge graph (Neo4j) -- skip if unavailable
    if "--no-graph" not in sys.argv:
        try:
            from build_graph import build_graph as ingest_graph
            ingest_graph(triples)
        except Exception as e:
            print(f"Neo4j ingestion skipped: {e}")

    # 9. Build vector store (Qdrant) -- skip if unavailable
    if "--no-graph" not in sys.argv:
        try:
            from build_vectorstore import build_vectorstore
            build_vectorstore(entities_structured, relations_structured)
        except Exception as e:
            print(f"Qdrant ingestion skipped: {e}")

    # 10. Generate visualization (always runs -- no external service needed)
    viz_path = os.path.join(ROOT, "outputs", "graph_visualization.html")
    visualize_from_triples(entities_structured, relations_structured, viz_path)
    print(f"Visualization saved to {viz_path}")


if __name__ == "__main__":
    main()
