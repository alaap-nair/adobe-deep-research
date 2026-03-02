import json
import csv
import os
from extract_entitites import extract_entities
from extract_triples import extract_triples

MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct:free"

# Project root (parent of src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_text(path=None):
    if path is None:
        path = os.path.join(_ROOT, "data", "passage.txt")
    with open(path, "r") as f:
        return f.read()

def save_triples_to_csv(triples, path=None):
    if path is None:
        path = os.path.join(_ROOT, "triples.csv")
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header row
        writer.writerow(["head", "relation", "tail", "evidence"])

        for triple in triples:
            writer.writerow([
                triple.get("head", ""),
                triple.get("relation", ""),
                triple.get("tail", ""),
                triple.get("evidence", "")
            ])

def main():
    text = load_text()

    print("Extracting entities...")
    entities = extract_entities(text)
    print("Entities found:", entities)

    print("\nExtracting triples...")
    graph_data = extract_triples(text, entities)

    triples = graph_data.get("triples", [])

    print("\nModel used:", MODEL_NAME)
    print("Number of triples:", len(triples))

    # Save CSV
    save_triples_to_csv(triples)

    print("Triples saved to triples.csv")

if __name__ == "__main__":
    main()