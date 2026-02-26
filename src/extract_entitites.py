import spacy
from pathlib import Path
import json

INPUT = Path("data/passage.txt")
OUT = Path("outputs/entities.json")

def main():
    text = INPUT.read_text(encoding="utf-8")

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Very simple baseline: use noun chunks + named entities
    entities = set()

    for ent in doc.ents:
        entities.add(ent.text.strip())

    for chunk in doc.noun_chunks:
        s = chunk.text.strip()
        if len(s) >= 3:
            entities.add(s)

    entities = sorted(entities, key=lambda x: (len(x), x))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"entities": entities}, indent=2), encoding="utf-8")
    print(f"Wrote {len(entities)} entities -> {OUT}")

if __name__ == "__main__":
    main()