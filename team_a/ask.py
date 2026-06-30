import json
import os
import re
import sys

import requests
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.neo4j_engine import Neo4jConfig, Neo4jEngine  # noqa: E402

try:
    from src.embeddings import embed_texts  # type: ignore[attr-defined]  # noqa: E402
except Exception:
    from src.embeddings import embed_texts_openrouter as embed_texts  # noqa: E402


def strip_markdown_fence(content: str) -> str:
    content = (content or "").strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def extract_query_entities(question: str, known_entities: list[str], max_entities: int = 5) -> list[str]:
    q = (question or "").lower()
    hits: list[tuple[int, str]] = []
    for name in known_entities:
        n = (name or "").strip()
        if not n:
            continue
        if re.search(rf"\b{re.escape(n.lower())}\b", q):
            hits.append((len(n), n))
    hits.sort(reverse=True)
    out = []
    seen = set()
    for _, n in hits:
        if n not in seen:
            out.append(n)
            seen.add(n)
        if len(out) >= max_entities:
            break
    if not out:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", q)
        stop = {"what", "does", "is", "the", "a", "an", "in", "of", "to", "and"}
        out = [t for t in tokens if t not in stop][:max_entities]
    return out


def build_context(chunks: list[dict], triples: list[dict]) -> str:
    chunk_lines = []
    for c in chunks:
        cid = c.get("id", "unknown")
        txt = (c.get("text") or "").strip()
        chunk_lines.append(f"[{cid}] {txt}")
    triple_lines = []
    for t in triples:
        head = t.get("head", "")
        rel = t.get("relation", "")
        tail = t.get("tail", "")
        evidence = t.get("evidence", "")
        if evidence:
            triple_lines.append(f"({head}) -[{rel}]-> ({tail}) | evidence: {evidence}")
        else:
            triple_lines.append(f"({head}) -[{rel}]-> ({tail})")

    chunk_block = "\n\n".join(chunk_lines) if chunk_lines else "(none)"
    triple_block = "\n".join(triple_lines) if triple_lines else "(none)"
    return f"Vector Evidence Chunks:\n{chunk_block}\n\nKnowledge Graph Facts:\n{triple_block}\n"


def generate_answer(question: str, context: str) -> tuple[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ("I don't know.", "Missing OPENROUTER_API_KEY.")
    model = os.getenv("QA_MODEL", "meta-llama/llama-3.1-8b-instruct")

    prompt = (
        "You are a biology QA system. Answer ONLY using the context below.\n"
        "If the answer is not present, say you don't know.\n"
        "Use both text evidence and graph relationships.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Return valid JSON with keys: answer, reasoning."
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        response.raise_for_status()
        data = json.loads(response.text.strip())
        content = data["choices"][0]["message"]["content"]
        content = strip_markdown_fence(content)
        parsed = json.loads(content)
        return (parsed.get("answer", "I don't know."), parsed.get("reasoning", "No reasoning provided."))
    except Exception as e:
        return ("I don't know.", f"Answer generation failed: {e}")


def append_output(payload: dict) -> None:
    with open(os.path.join(ROOT, "outputs.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python ask.py "What does glycolysis produce"')
        sys.exit(1)
    question = sys.argv[1].strip()

    cfg = Neo4jConfig.from_env()
    engine = Neo4jEngine(cfg)
    try:
        query_embedding = embed_texts([question])[0]
        chunks = engine.vector_search(query_embedding, k=5)

        entity_names = engine.list_entity_names()
        query_entities = extract_query_entities(question, entity_names)
        traversal = engine.traverse_from_entities(query_entities, hop_min=1, hop_max=2, limit=20)

        triples = []
        kg_nodes = set()
        kg_edges = set()
        for row in traversal:
            # current engine returns flat rows (head/relation/tail/evidence)
            if all(k in row for k in ("head", "relation", "tail")):
                triples.append(
                    {
                        "head": row.get("head", ""),
                        "relation": row.get("relation", ""),
                        "tail": row.get("tail", ""),
                        "evidence": row.get("evidence", ""),
                    }
                )
                kg_nodes.add(row.get("head", ""))
                kg_nodes.add(row.get("tail", ""))
                kg_edges.add(f"{row.get('head','')} -[{row.get('relation','')}]-> {row.get('tail','')}")
                continue

            # compatibility: path-style rows
            p = row.get("p")
            if not p:
                continue
            for n in p.nodes:
                name = n.get("name") or n.get("id")
                if name:
                    kg_nodes.add(str(name))
            for r in p.relationships:
                rel_t = r.type if hasattr(r, "type") else "RELATED_TO"
                s = r.start_node.get("name") or r.start_node.get("id")
                t = r.end_node.get("name") or r.end_node.get("id")
                if s and t:
                    kg_edges.add(f"{s} -[{rel_t}]-> {t}")
                    triples.append({"head": s, "relation": rel_t, "tail": t, "evidence": r.get("evidence", "")})

        context = build_context(chunks, triples)
        answer, reasoning = generate_answer(question, context)

        result = {
            "question": question,
            "answer": answer,
            "citations": [c.get("id") or c.get("source") for c in chunks if c.get("id") or c.get("source")],
            "reasoning": reasoning,
            "kg_trace": {
                "nodes": sorted([n for n in kg_nodes if n]),
                "edges": sorted(kg_edges),
            },
        }
        append_output(result)
        print(json.dumps(result, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()

