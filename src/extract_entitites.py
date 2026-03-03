"""
extract_entitites.py -- Team 2: Derive unique entities from extracted triplets.

In schema-free extraction, entities are discovered during triple extraction
rather than in a separate pre-processing step. This module post-processes
the triplet list to produce a deduplicated entity inventory.
"""


def extract_entities(triples):
    """Collect unique entity names from triplet heads and tails."""
    entities = set()
    for triple in triples:
        head = triple.get("head", "").strip()
        tail = triple.get("tail", "").strip()
        if head:
            entities.add(head)
        if tail:
            entities.add(tail)
    return sorted(entities)


if __name__ == "__main__":
    sample = [
        {"head": "Glycolysis", "relation": "occurs in", "tail": "cytosol", "evidence": "..."},
        {"head": "Glycolysis", "relation": "produces", "tail": "pyruvate", "evidence": "..."},
    ]
    entities = extract_entities(sample)
    print("Entities:", entities)
