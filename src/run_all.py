import json
import csv
import os
import sys
from extract_entitites import extract_entities
from extract_triples import extract_triples
from schema import validate_triples, validate_pipeline_output
from parse_pdf import pdf_to_text

MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

# Project root (parent of src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_text(path=None):
    """Load text from a .txt or .pdf file. Defaults to data/passage.txt or data/passage.pdf."""
    if path is None:
        txt_path = os.path.join(_ROOT, "data", "passage.txt")
        pdf_path = os.path.join(_ROOT, "data", "passage.pdf")
        if os.path.isfile(pdf_path):
            path = pdf_path
        else:
            path = txt_path
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.lower().endswith(".pdf"):
        return pdf_to_text(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
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
    # Optional: python src/run_all.py [path_to_pdf_or_txt]
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    text = load_text(input_path)
    print("Input loaded (chars):", len(text))

    print("Extracting entities...")
    entities = extract_entities(text)
    print("Entities found:", entities)

    print("\nExtracting triples...")
    graph_data = extract_triples(text, entities)
    triples = graph_data.get("triples", [])

    print("\nModel used:", MODEL_NAME)
    print("Number of triples (raw):", len(triples))

    # Schema validation for output quality
    valid_triples, errors = validate_triples(triples)
    if errors:
        print("\nValidation issues:", len(errors))
        for e in errors[:10]:
            print("  -", e)
        if len(errors) > 10:
            print("  ... and", len(errors) - 10, "more")
    print("Valid triples (schema):", len(valid_triples))

    # Full output schema check (entities + triples) for quality verification
    output_ok, output_errors = validate_pipeline_output(entities, valid_triples)
    if not output_ok and output_errors:
        print("Pipeline output schema:", output_errors[0])
    else:
        print("Pipeline output schema: valid")

    # Save valid triples to CSV
    save_triples_to_csv(valid_triples)
    print("Triples saved to triples.csv")

if __name__ == "__main__":
    main()