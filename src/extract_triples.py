from pathlib import Path
import json
import re

INPUT = Path("data/passage.txt")
OUT = Path("outputs/triples.json")

# Minimal rule-based patterns (just to have something working)
PATTERNS = [
    # "X is the process by which Y"
    (re.compile(r"(.+?) is the process by which (.+?)\.", re.IGNORECASE),
     lambda m: [(m.group(1).strip(), "enables", m.group(2).strip())]),

    # "X is located in Y"
    (re.compile(r"(.+?) is located in (.+?)\.", re.IGNORECASE),
     lambda m: [(m.group(1).strip(), "located_in", m.group(2).strip())]),

    # "X occurs in Y"
    (re.compile(r"(.+?) occurs in (.+?)\.", re.IGNORECASE),
     lambda m: [(m.group(1).strip(), "occurs_in", m.group(2).strip())]),

    # "X produces Y"
    (re.compile(r"(.+?) produces (.+?)\.", re.IGNORECASE),
     lambda m: [(m.group(1).strip(), "produces", m.group(2).strip())]),
]

def main():
    text = INPUT.read_text(encoding="utf-8")
    triples = []

    # Split into sentences crudely (good enough for baseline)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    for sent in sentences:
        for pat, fn in PATTERNS:
            m = pat.search(sent)
            if m:
                triples.extend(fn(m))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"triples": triples}, indent=2), encoding="utf-8")
    print(f"Wrote {len(triples)} triples -> {OUT}")

if __name__ == "__main__":
    main()