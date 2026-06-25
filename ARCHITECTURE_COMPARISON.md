# Team A vs Team B — Architecture Comparison

Both teams built a KG-RAG pipeline over OpenStax Biology with the same goal: replicate large-model "deep research" quality using small LLMs grounded in a knowledge graph, instead of web search.

---

## What's the Same

**Core pipeline shape is identical:**
```
source text → chunk → extract triples (LLM) → build graph (Neo4j) → retrieve → synthesize answer → RAGAS eval
```

- Both call **OpenRouter** with `temperature=0`, single-shot prompt per chunk
- Both extract `(head, relation, tail, evidence)` triples
- Both chunk on **paragraph boundaries** with character overlap
- Both store the graph in **Neo4j** and traverse with Cypher BFS
- Both validate triples with **Pydantic**
- Both evaluate with **RAGAS** (faithfulness, answer relevancy, context recall, context precision)
- Both use a fixed allowed-list of relation types — not open-ended extraction

---

## What's Different

### 1. Entity Extraction

**Team A** runs **GLiNER** (`urchade/gliner_medium-v2.1`) on the text *before* calling the LLM. The LLM receives a pre-built entity list and is told not to invent names outside it. This anchors extraction but requires an extra ML model.

**Team B** skips NER entirely. Entities are derived *after* the fact — collect all heads and tails from the extracted triples. Cleaner pipeline, but noisier entity names (which is why they added canonicalization later).

---

### 2. Relation Schema

**Team A** — 10 simple verbs:
`occurs_in`, `produces`, `converts_to`, `uses`, `requires`, `inhibits`, `activates`, `transports_to`, `donates_electrons_to`, `accepts_electrons_from`

Plus a `RELATION_ALIASES` dict that maps ~30 free-form phrasings back to canonical labels (e.g. "generates" → `produces`), giving the LLM wiggle room.

**Team B** — 23 richer typed relations:
`CATALYZES`, `PRODUCES`, `CONSUMES`, `OCCURS_IN`, `PART_OF`, `ENCODES`, `TRANSCRIBES_TO`, `TRANSLATES_TO`, `REPLICATES`, `BINDS`, `PAIRS_WITH`, `REGULATES`, `INHIBITS`, `DECOMPOSES`, `CONVERTS_TO`, `LOCATED_IN`, `BELONGS_TO`, `ABSORBS`, `RELEASES`, `CONSISTS_OF`, `MUTUALISTIC_WITH`, `PARASITIC_ON`

No alias mapping — the LLM must emit the exact canonical string or the triple is dropped.

---

### 3. Storage — The Biggest Divergence

**Team A: one store (Neo4j only)**
- Chunks stored as `(:Chunk)` nodes with `embedding` as a node property
- Embeddings generated via **OpenRouter API** (`openai/text-embedding-3-small`)
- Vector search via **Neo4j's native vector index** (`db.index.vector.queryNodes`)
- Everything in one place, one service to run

**Team B: two stores (Neo4j + Qdrant)**
- Neo4j holds the graph (`Entity` nodes + `RELATES_TO` edges)
- Qdrant holds **3 separate collections**: entities, evidence (triple text), chunks
- Embeddings generated locally via **`sentence-transformers`** (`BAAI/bge-large-en-v1.5`, 1024-dim)
- Both stores share **deterministic IDs** (`ent:<name>`, `triple:<sha256_16>`, `uuid5`) so a Qdrant hit jumps straight to a Neo4j node with no lookup table
- Qdrant can run as a local file store (`./qdrant_data`) — no server required by default

---

### 4. Chunking

| | Team A | Team B |
|---|---|---|
| Max chunk size | 1500 chars | 700 chars |
| Overlap | 200 chars (raw tail of prev chunk) | 100 chars (cleaned to not start mid-word) |
| Long paragraphs | Added as-is even if oversized | Sentence-fallback → word-fallback, always stays under limit |

---

### 5. Retrieval

**Team A** baseline is **TF-IDF-style term scoring** — counts how many question tokens appear in each chunk, ranks by overlap. No dense vectors at query time. No reranking or entity resolution beyond keyword matching.

**Team B** baseline is **Qdrant dense vector search** (cosine similarity on BGE embeddings). Four configurable strategies on top:

| Mode | What it does | Result |
|---|---|---|
| **C0** (default) | 2× candidate pool, dedup-truncate | baseline |
| **C1** `USE_RERANKER=1` | Cross-encoder rerank (`BAAI/bge-reranker-v2-m3`) over 3× pool | **RAGAS winner** — context precision 0.592 → 0.616, best on all 4 metrics |
| **C2** `RETRIEVAL_MODE=mmr` | Maximal Marginal Relevance over 6× pool | underperformed (0.557) |
| **C3** `RETRIEVAL_MODE=graph_boost` | Re-score chunks by graph-neighbor entity mentions | underperformed (0.585) |

Team B also does fuzzy alias matching + semantic fallback for entity resolution at query time. Team A has neither.

---

### 6. Entity Deduplication

**Team A**: none.

**Team B**: two-stage canonicalization (`canonicalize.py`, opt-in via `KG_CANONICALIZE=1`):
1. **Rule normalization** — NFKC unicode fold, strip punctuation, lowercase, WordNet lemmatize, plural fallback
2. **Embedding clustering** — greedy single-link on BGE embeddings at cosine ≥ 0.92, catches near-dupes like "Photosystem II" vs "PSII"

---

### 7. Answer Synthesis

**Team A** can return an **extractive answer** (best matching sentences from top chunks) with no LLM call at all. LLM synthesis is optional.

**Team B** always synthesizes with an LLM via `qa_client.py`. Output is a strict Pydantic schema:
```json
{ "question": "...", "answer": "...", "citations": [...], "reasoning": "..." }
```
Citations are validated against a bounded allow-list. If context is insufficient, the model returns exactly `"I don't know based on the provided context."` The `reasoning` field gets a KG proof appended (matched entities + traversed edges).

---

### 8. Everything Else

| | Team A | Team B |
|---|---|---|
| **Services** | Neo4j only | Neo4j + Qdrant (Docker Compose provided) |
| **Tests** | None | Full pytest suite (unit, integration, temporal, invalidation) |
| **UI** | None | Streamlit chat UI (`app.py`) |
| **Temporal model** | None | Bi-temporal edges (`valid_at`, `invalid_at`, `expired_at`) on Neo4j relationships for point-in-time queries |
| **Personalization** | None | Graphiti-based per-user episode ingestion + per-user subgraph retrieval |
| **Graph communities** | None | Community detection (`communities.py`, opt-in) |
| **Visualization** | None | `visualize_graph.py` — pyvis interactive HTML, always runs after ingest |

---

## Summary

Team A is the **simpler proof-of-concept** — GLiNER anchors entities, everything lives in Neo4j, extractive answers work without an LLM call. Team B is the **research-grade system** — dual stores with deterministic cross-linking, typed schema, local BGE embeddings, a benchmarked retrieval pipeline with reranking experiments, and layered extras (canonicalization, temporal edges, personalization, tests, UI). The core extraction loop is shared; the retrieval and infrastructure layers are completely different.
