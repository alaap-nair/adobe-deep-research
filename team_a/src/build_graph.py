import argparse
import csv
import os

from dotenv import load_dotenv

from src.chunking import chunk_text
from src.embeddings import embed_texts_openrouter, embedding_dim
from src.neo4j_engine import Neo4jConfig, Neo4jEngine
from src.run_all import load_text


def load_triples_csv(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "head": row.get("head", ""),
                    "relation": row.get("relation", ""),
                    "tail": row.get("tail", ""),
                    "evidence": row.get("evidence", ""),
                }
            )
    return out


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Build Neo4j graph (entities + triples + chunk embeddings).")
    ap.add_argument("--input", default=None, help="Path to .txt or .pdf to chunk + embed (defaults to run_all default).")
    ap.add_argument("--triples", default="triples.csv", help="Path to triples CSV (default: triples.csv).")
    ap.add_argument("--chunk-chars", type=int, default=1500, help="Max chars per chunk.")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in chars.")
    ap.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    triples_path = args.triples
    if not os.path.isabs(triples_path):
        triples_path = os.path.join(root, triples_path)
    if not os.path.isfile(triples_path):
        raise FileNotFoundError(f"Triples CSV not found: {triples_path}. Run: python src/run_all.py [input] first.")

    text = load_text(args.input)
    chunks = chunk_text(text, max_chars=args.chunk_chars, overlap=args.chunk_overlap)
    if not chunks:
        raise ValueError("No chunks produced from input text.")

    print(f"Chunks: {len(chunks)}")
    embeddings = embed_texts_openrouter(chunks, batch_size=args.batch_size)
    dim = embedding_dim(embeddings)
    print(f"Embedding dim: {dim}")

    cfg = Neo4jConfig.from_env()
    engine = Neo4jEngine(cfg)
    try:
        engine.setup_schema(vector_dimensions=dim)

        triples = load_triples_csv(triples_path)
        entity_names = sorted(
            {t.get("head", "").strip() for t in triples if t.get("head")} | {t.get("tail", "").strip() for t in triples if t.get("tail")}
        )

        engine.upsert_entities(entity_names)
        engine.upsert_triples(triples)

        source = os.path.abspath(os.path.expanduser(args.input)) if args.input else "default_input"
        chunk_rows = []
        chunk_ids_and_text = []
        for i, (c, e) in enumerate(zip(chunks, embeddings)):
            cid = f"{os.path.basename(source)}::chunk_{i}"
            chunk_rows.append(
                {"id": cid, "text": c, "source": source, "chunk_index": i, "embedding": e}
            )
            chunk_ids_and_text.append((cid, c))

        engine.upsert_chunks(chunk_rows)
        engine.link_chunk_mentions(chunk_ids_and_text, entity_names)

        print("Neo4j graph build complete.")
        print("Try this in Neo4j Browser:")
        print("  MATCH (e:Entity)-[r]->(e2:Entity) RETURN e,r,e2 LIMIT 50;")
        print("  MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN c,e LIMIT 50;")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
