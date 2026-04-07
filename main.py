from extractor import extract_triples
from ingestor import create_vector_index, hybrid_search, ingest_triple

texts = [
    "Pyruvate kinase converts phosphoenolpyruvate to pyruvate during glycolysis.",
    "ATP synthase uses the proton gradient to produce ATP in the mitochondria.",
    "Hexokinase phosphorylates glucose using ATP in the first step of glycolysis.",
]

entities = [
    "pyruvate",
    "ATP",
    "glycolysis",
    "mitochondria",
    "glucose",
    "hexokinase",
    "pyruvate kinase",
    "ATP synthase",
]

create_vector_index()

for text in texts:
    print(f"\nProcessing: {text[:60]}...")
    result = extract_triples(text, entities)
    for triple in result.get("triples", []):
        print(f"  Ingesting: {triple['head']} --[{triple['relation']}]--> {triple['tail']}")
        ingest_triple(triple, text)

print("\n--- Hybrid Search Results ---")
results = hybrid_search("how is ATP produced?")
for r in results:
    print(f"{r['head']} --[{r['relation']}]--> {r['tail']}  (score: {r['score']:.3f})")
