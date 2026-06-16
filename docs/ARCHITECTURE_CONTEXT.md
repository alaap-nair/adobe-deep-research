# Architecture Context — Adobe × UpSync Deep Research over a Biology Knowledge Graph

> Context dump for handing to another prompt. Describes the **current state of this branch**
> (`consolidate/main-subdirs`, detached HEAD) as of 2026-06-05. Where the code diverges from
> the root `CLAUDE.md`, this document reflects the **code**, and the divergence is called out.

---

## 1. Project goal

Replicate large-model "deep research" answer quality using **small, lightweight LLMs** grounded
in a domain-specific **Knowledge Graph + RAG** over OpenStax Biology — instead of SERP/web
retrieval. Proof-of-concept domain: a "Biology Expert" over OpenStax Biology 2e
(Ch. 8 Photosynthesis, Ch. 14–15 DNA/Genes/Proteins, Ch. 24 Fungi; the root-level genetics
workstream focuses on Ch. 14/15/17 DNA, Genes & Proteins, Biotech).

Pipeline shape (shared by both teams): **ingest → entity/triple extraction → KG construction →
hybrid retrieval → answer synthesis → RAGAS evaluation.**

---

## 2. Repository layout (important — there are THREE layers)

The branch consolidates two competing sub-team implementations via subtree merges, **plus** a
working-tree (untracked) top-level evaluation layer:

```
team_a/        # Sub-team: Camillia & Shreya — UNIFIED single-store (Neo4j-only)
team_b/        # Sub-team: Jason & Shreyas  — DECOUPLED dual-store (Neo4j + Qdrant)
src/           # (untracked) genetics-domain eval/extraction layer — see §6
data/          # (untracked) genetics PDFs + ground_truth.json (20 Qs)
outputs/       # (untracked) answers/, traces, llm_cache/, checkpoints/, RAGAS results
docs/          # (untracked) workstream issue logs + this file
CLAUDE.md      # (untracked) project instructions — describes team_b's design lineage
README.md      # (tracked) explains the two-team layout
```

- **`team_b/` is the canonical, most complete implementation** and matches the architecture that
  `CLAUDE.md` describes. When in doubt, read `team_b/src/`.
- `team_a/` is the alternative single-store design (everything inside Neo4j).
- The untracked root `src/` (5 files) imports module names (`query_engine`, `qa_client`,
  `config`, `graph_schema`) that live in the canonical pipeline — it is the genetics-focused
  evaluation/extraction harness layered on top of the dual-store pipeline. It is in a
  transitional state (the full pipeline `src/` was consolidated into `team_b/src/`).

---

## 3. Team B architecture (canonical dual-store) — `team_b/`

### Pipeline flow

```
Input (.txt / .pdf)
  → parse_pdf.py            (PyMuPDF; LlamaParse optional)
  → extract_triples.py      (OpenRouter → small LLM → TYPED JSON triples)
  → schema.py               (Pydantic: Triple / ExtractionResult validation)
  → extract_entitites.py    (derive + dedupe entities from triples)  [filename typo is intentional]
  → run_all.py              (orchestrator; saves outputs/triples*.json)
  → graph_schema.py         (deterministic IDs + GraphEntity / GraphRelation models)
      → canonicalize.py     (optional W3A entity dedup; gated by KG_CANONICALIZE=1)
  → build_graph.py          (Neo4j ingest: :Entity nodes, :RELATES_TO edges)
  → build_vectorstore.py    (Qdrant ingest: 3 collections, BGE-large embeddings)
      → chunking.py         (paragraph-first, sentence-fallback source chunking)
  → visualize_graph.py      (pyvis interactive HTML — always runs, no service needed)

Query time:
  ask.py "question"
  → query_engine.py         (hybrid retrieval: Qdrant vectors + Neo4j traversal)
      → reranker.py         (optional cross-encoder rerank; gated by USE_RERANKER=1)
  → qa_client.py            (OpenRouter LLM answer synthesis → strict JSON schema)
  → outputs/answers/<slug>.json  +  <slug>_trace.json
```

### Extraction — TYPED, not schema-free (divergence from CLAUDE.md)

`CLAUDE.md` says "schema-free extraction." **The code is now schema-CONSTRAINED.**
`extract_triples.py` injects a fixed vocabulary from `domain_schema.py` into the system prompt
and drops any triple whose `head_type` / `tail_type` / `relation_type` is out-of-vocabulary.

Each extracted triple is:
```json
{ "head", "head_type", "relation", "relation_type", "tail", "tail_type", "evidence" }
```
- `head_type` / `tail_type` ∈ `domain_schema.NODE_TYPES` (11 types: Organism, Molecule, Enzyme,
  CellularStructure, Process, Pathway, TaxonomicGroup, EcologicalRole, ChemicalReaction, Gene,
  GeneticElement).
- `relation_type` ∈ `domain_schema.RELATION_TYPES` (23 types: CATALYZES, PRODUCES, CONSUMES,
  OCCURS_IN, PART_OF, DECOMPOSES, ENCODES, TRANSCRIBES_TO, TRANSLATES_TO, REPLICATES, BINDS,
  PAIRS_WITH, REGULATES, …).
- `relation` is the free-text surface verb (1–3 words); `relation_type` is the canonical label.
- `evidence` must be an exact sentence copied from the source.
- Single-shot prompt per passage, `temperature=0`, no iterative refinement.
- Rate-limit handling: exponential backoff on HTTP 429, then falls back from a `:free` model to
  the paid route.

### Dual-database deterministic ID scheme (the core design)

Both DBs share the same IDs so a Qdrant hit jumps straight to a Neo4j node. All ID logic is in
`graph_schema.py`:
- **Entity ID**: `ent:<normalized_name>` — e.g. `ent:atp_synthase`. `normalize_name()` =
  strip → lowercase → collapse whitespace → spaces-to-underscores.
- **Triple ID**: `triple:<sha256_16>` of `normalize(head)|normalize(relation)|normalize(tail)`.
- **Qdrant point ID**: `uuid5(NAMESPACE_URL, string_id)` — deterministic UUID from the string ID.
- **Chunk ID**: `chunk:<source_prefix>:<index>`.
- Upserts are idempotent: Neo4j `MERGE`, Qdrant deterministic point IDs → re-running is safe.

### Neo4j schema (`build_graph.py`)
- `:Entity` nodes with `entity_id`, `name`, `original_names`, `node_type`.
- Typed entities get a **second label** (e.g. `:Entity:Molecule`) so Cypher can filter by domain
  type. A regex whitelist guards label injection. Untyped entities fall back to plain `:Entity`.
- `:RELATES_TO` edges with `triple_id`, `relation`, `evidence`, `relation_type`.
- Uniqueness constraint on `Entity.entity_id`; index on `RELATES_TO.triple_id`.

### Qdrant collections (`build_vectorstore.py`, `config.py`) — THREE collections
- `entities` — one point per entity, embedding of the entity name; payload has `entity_id`,
  `name`, `original_names`.
- `evidence` — one point per triple, embedding of the evidence sentence; payload has `triple_id`,
  `head_entity_id`, `tail_entity_id`, `relation`, `evidence`.
- `chunks` — one point per source-text chunk; payload has `chunk_id`, `text`, `source_name`,
  `chunk_index`. (CLAUDE.md only mentions two collections; `chunks` is the newer third.)
- Embeddings: `BAAI/bge-large-en-v1.5` (1024-dim), L2-normalized, cosine distance.
- **Qdrant runs in local file mode by default** (`QDRANT_PATH=./qdrant_data`, no server needed);
  set `QDRANT_URL` to use a remote server.

### Hybrid retrieval (`query_engine.py`)

`QueryEngine.retrieve_context(query)` returns `{query_analysis, vector_hits{chunks,evidence},
graph_trace{seed_entity_ids, retrieved_nodes, traversed_edges}}`:
1. **Entity resolution** (`resolve_query_entities`): keyword extraction → fuzzy alias match
   (difflib, ≥0.78) against a cached entity catalog → semantic fallback via Qdrant. Ranked
   keyword > fuzzy > semantic.
2. **Vector retrieval**: `search_chunks`, `search_evidence`, `search_entities` over Qdrant, with
   dedup by id and normalized text.
3. **Graph expansion**: for each seed entity, `get_entity_neighborhood(entity_id, hops)` runs a
   variable-length Cypher traversal (default `QA_GRAPH_HOPS=2`) and collects nodes + edges.
4. **Chunk-selection strategies** (A10 Part 2 context-precision experiments), chosen by env:
   - default (C0): top-k from a 2× pool, dedupe-truncate.
   - `USE_RERANKER=1` (C1, the shipped winner): cross-encoder rerank of a 3× pool via
     `reranker.py`. Default reranker model = **`BAAI/bge-reranker-v2-m3`** (replaced the weaker
     MS-MARCO MiniLM, which had *regressed* precision).
   - `RETRIEVAL_MODE=mmr` (C2): Maximal Marginal Relevance over a 6× pool (relevance vs diversity,
     `lambda=0.7`).
   - `RETRIEVAL_MODE=graph_boost` (C3): re-rank chunks by graph connectivity — boost chunks
     mentioning entities within N hops of the question's entities in Neo4j (dual-store advantage).

### Answer synthesis (`qa_client.py`, `ask.py`)
- `ask.py "question"` → fails fast if Neo4j down or Qdrant collections missing/empty → retrieve
  context → synthesize → write `outputs/answers/<slug>.json` and `<slug>_trace.json`, print JSON.
- Output schema (Pydantic `QAResponse`): `{question, answer, citations[], reasoning}`.
- System prompt is strict: answer **only** from supplied context; if insufficient, return exactly
  `"I don't know based on the provided context."`; copy citations verbatim from an allow-list.
- `parse_llm_json` is hardened for small/cheap models: strips markdown fences, trailing commas,
  prose preambles, recovers fields from truncated output via regex, and unescapes via
  `json.loads` (preserves multi-byte UTF-8 like "β-carotene").
- Citations are validated against a bounded allow-list (`build_allowed_citations`); `reasoning`
  is rebuilt as a KG "proof" (matched entities + traversed edges + graph evidence) appended to
  the model's reasoning (`build_graph_proof`).
- `max_tokens=768` default (fits gemma-2-27b's 8K context, avoids phi-4 truncation).

### Canonicalization (`canonicalize.py`, W3A — opt-in via `KG_CANONICALIZE=1`)
Two-stage entity dedup: (1) **rule_normalize** — NFKC fold, strip punctuation, lowercase,
WordNet lemmatize (with a Greek/Latin-aware plural fallback); (2) **cluster_entities** — greedy
single-link clustering on BGE embeddings at cosine ≥ 0.92. Rewrites relations to canonical IDs,
drops self-loops, recomputes triple IDs.

---

## 4. Team A architecture (alternative single-store) — `team_a/`

**Unified store: graph + chunks + vectors all live inside Neo4j** (no Qdrant). Key differences:
- Stores `(:Entity)` nodes + relationships AND `(:Chunk)` nodes that carry `embedding` vectors,
  using a **Neo4j native vector index** for semantic search.
- `(:Chunk)-[:MENTIONS]->(:Entity)` links chunks to entities.
- Entry points: `team_a/src/run_all.py` (extract), `team_a/src/build_graph.py` (graph + vector
  index), `team_a/bio_qa.py` / `team_a/ask.py` (query), `team_a/neo4j_engine.py`.
- Uses OpenRouter for extraction; PDF OCR via Mistral; embeddings configurable (README shows
  `openai/text-embedding-3-small`).
- Pitch: simpler infra, fewer moving parts. Team B's pitch: each store optimized for its
  retrieval type, linked by shared IDs.

Both teams share BGE embeddings (in B), a DeepSeek/Mistral-class synthesis model, and RAGAS eval.

---

## 5. Evaluation (RAGAS)

- Metrics: **Faithfulness, Answer Relevancy, Context Recall, Context Precision**
  (`LLMContextPrecisionWithoutReference`). RAGAS judge LLM via OpenRouter; embeddings =
  `BAAI/bge-large-en-v1.5`.
- Ground truth: `data/ground_truth.json`. The root genetics set has **20 questions**;
  `team_b/data/ground_truth.json` has **40**.
- **Assignment #10 Part 1**: regenerated team_b's 40 reference answers with
  **`anthropic/claude-opus-4.8`** (corpus-grounded — retrieves top-10 passages first, sets a
  `grounded` flag, falls back to model knowledge if out-of-corpus). 35/40 grounded, 5 flagged
  out-of-corpus. Original curated answers preserved in `ground_truth.curated.json` /
  `ground_truth_human`.
- **Assignment #10 Part 2** (context precision): controlled matrix C0–C3 (see §3 retrieval
  strategies), synthesis + judge pinned to `mistral-small-3.2-24b`. **Winner = C1
  `bge-reranker-v2-m3`**: context_precision 0.592 → 0.616, and uniquely dominates the baseline on
  all four metrics. MMR (0.557) and graph-boost (0.585) underperformed. Hardest multi-hop
  question (q6, RuBisCO→fungal-decomposition cross-chapter bridge) stays unsolved at 0.167.
  Full writeup: `team_b/docs/assignment_10.md` (and `assignment_9.md`).

---

## 6. Root-level genetics workstream (untracked `src/`, `data/`, `outputs/`)

A genetics-domain layer built on the dual-store pipeline, with its **own** domain schema
(`src/bio_schema.py`: Gene, Protein, Molecule, CellStructure, BiologicalProcess, Pathway,
Organism, Disease, Technique …) — distinct from `team_b/src/domain_schema.py`. Files:
- `src/batch_eval.py` — run all ground-truth questions through the live pipeline, save traces,
  emit a rubric-format summary. Imports `query_engine`, `qa_client`, `config`.
- `src/ragas_eval.py` — RAGAS over the 20-question genetics ground truth; `--from-traces` reuses
  saved traces, else runs live queries. `RAGAS_MODEL` defaults to
  `mistralai/mistral-small-3.1-24b-instruct:free`.
- `src/check_coverage.py` — verify ground-truth question entities exist as KG nodes; reports
  coverage %. Imports `graph_schema` (`normalize_name`, `entity_id`).
- `src/generate_docx.py` — export Assignment 7 deliverables to a Word doc (`python-docx`).
- `src/bio_schema.py` — genetics node/relation vocabulary for extraction.
- `data/`: `Ch14_DNA_Structure_and_Function.pdf`, `Ch15_Genes_and_Proteins.pdf`,
  `Ch17_Biotechnology_and_Genomics.pdf`, `ConceptsofBiology-WEB.pdf` (615 pages, ~1.62M chars,
  ~463 extraction chunks), `ground_truth.json` (20 Qs).
- `outputs/`: `answers/` (per-question `.json` + `_trace.json`), `llm_cache/` (sha256-keyed
  extraction cache), `checkpoints/` (per-document resume state), `batch_eval_summary.json`,
  `coverage_report.json`, `graph_visualization.html`.
- `docs/workstream1_issues.md` — full-scale-run issue log (sequential extraction w/o checkpoint,
  silent JSON-parse failures, unbatched Neo4j ingest, etc.) with fixes (disk LLM cache,
  checkpoints, batched UNWIND of 500).

> Note: these 5 untracked `src/` files import the core pipeline modules by name but those modules
> now live in `team_b/src/`; running them requires the canonical pipeline on `PYTHONPATH`. This
> reflects the transitional consolidation state of the working tree.

---

## 7. Configuration & commands

### Key env vars (`.env`, git-ignored; `team_b/src/config.py`)
- `OPENROUTER_API_KEY` — required for extraction, synthesis, RAGAS.
- `MODEL_NAME` / `QA_MODEL` — small LLM for extraction & answer synthesis (via OpenRouter).
- `RAGAS_MODEL` — judge model (defaults to `MODEL_NAME`).
- `NEO4J_URI` (`bolt://localhost:7687`), `NEO4J_USER` (`neo4j`), `NEO4J_PASSWORD` (`password`).
- `QDRANT_PATH` (default `./qdrant_data`, local file mode), `QDRANT_URL` (remote opt-in),
  `QDRANT_PORT` (6333).
- `EMBEDDING_MODEL` (`BAAI/bge-large-en-v1.5`), `EMBEDDING_DIM` (1024).
- Retrieval tuning: `QA_TOP_K_CHUNKS` (5), `QA_TOP_K_ENTITIES` (3), `QA_TOP_K_EVIDENCE` (5),
  `QA_GRAPH_HOPS` (2), `CHUNK_MAX_CHARS` (700), `CHUNK_OVERLAP` (100).
- Feature flags: `KG_CANONICALIZE`, `USE_RERANKER`, `RERANKER_MODEL`, `RETRIEVAL_MODE`.

### Commands (run inside `team_b/`)
```bash
pip install -r requirements.txt
docker compose up -d                      # Neo4j (+ Qdrant if remote)
python3 src/run_all.py data/passage.txt   # extract + ingest + visualize
python3 src/run_all.py path/to/doc.pdf
python3 src/run_all.py data/x.txt --no-graph   # extraction only, skip DBs
python3 ask.py "What does glycolisis produce"  # hybrid QA → JSON + trace
python3 src/run_ragas.py                  # RAGAS eval
```
Graceful degradation: `run_all.py` catches Neo4j/Qdrant connection errors and continues;
visualization always runs.

### Dependencies (team_b)
`python-dotenv`, `requests`, `pydantic>=2`, `pymupdf`, `neo4j>=5`, `qdrant-client>=1.7`,
`sentence-transformers>=2.2`, `pyvis>=0.3`, `networkx>=3`, `nltk>=3.8` (canonicalization),
`ragas>=0.2` + `datasets` + `langchain-openai` + `langchain-huggingface` (eval), `pytest>=7`.

---

## 8. Key design decisions
- **Deterministic shared IDs** across Neo4j + Qdrant → no lookup table, cross-store jumps for free.
- **Idempotent upserts** (Neo4j MERGE, Qdrant deterministic point IDs) → safe re-runs.
- **Schema-constrained extraction** (typed nodes/relations) → less noise than open extraction
  (this supersedes CLAUDE.md's "schema-free" claim).
- **Single-shot prompting**, `temperature=0` → reproducible extraction.
- **Local-file Qdrant by default** → tests/dev need no vector server; Neo4j optional (skipped if
  down). Tests use in-memory Qdrant; Neo4j tests auto-skip if unavailable.
- **Retrieve-then-rerank** (Nogueira & Cho 2019) with a domain-matched cross-encoder, gated/lazy.
- **Grounded ground truth** (Opus 4.8, corpus-anchored) so RAGAS recall/precision stay fair.

### Known divergences from root `CLAUDE.md` (which describes an earlier top-level snapshot)
1. Extraction is **typed/schema-constrained**, not schema-free.
2. Qdrant has **three** collections (`entities`, `evidence`, `chunks`), not two.
3. The reranker (`bge-reranker-v2-m3`) is **implemented and shipped**, not "planned."
4. Canonicalization, hierarchical/paragraph chunking, and RAGAS eval are implemented.
5. The full pipeline now lives under `team_b/src/` (and `team_a/src/`), not a single top-level
   `src/`; the untracked root `src/` is a genetics eval layer on top.
```
