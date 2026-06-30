# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-based schema-free knowledge graph extraction and storage pipeline. Takes scientific text/PDFs as input, uses a single LLM prompt (via OpenRouter + Mistral) to extract `(head, relation, tail, evidence)` triples, validates with Pydantic, then ingests into a dual-database architecture: **Neo4j** (knowledge graph for relationships) and **Qdrant** (vector store for semantic search). Both databases share deterministic IDs so you can query Qdrant and jump straight to Neo4j.

## Commands

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt
docker compose up -d  # Start Neo4j + Qdrant

# Run full pipeline (extract + ingest + visualize)
python src/run_all.py
python src/run_all.py data/biology_7_2.txt
python src/run_all.py path/to/document.pdf
python src/run_all.py data/passage.txt --no-graph  # extraction only, skip DB ingestion

# Run individual components
python src/build_graph.py outputs/triples.json       # Neo4j ingestion only
python src/build_vectorstore.py outputs/triples.json  # Qdrant ingestion only
python src/visualize_graph.py outputs/triples.json    # Generate HTML visualization
python src/query_engine.py "What does glycolysis produce?"  # Hybrid query

# Evaluation
python src/evaluate.py outputs/triples.json data/passage.txt

# Tests
python -m pytest tests/ -v                           # all tests
python -m pytest tests/test_graph_schema.py -v        # ID/schema tests (no services needed)
python -m pytest tests/test_build_vectorstore.py -v   # Qdrant tests (in-memory, no server needed)
python -m pytest tests/test_build_graph.py -v         # Neo4j tests (skipped if no Neo4j running)
python -m pytest tests/test_integration.py -v         # end-to-end with real data
```

## Architecture

```
Input (Text/PDF)
  -> parse_pdf.py (PyMuPDF or LlamaParse)
  -> extract_triples.py (OpenRouter API -> Mistral LLM -> JSON triples)
  -> schema.py (Pydantic validation)
  -> extract_entitites.py (deduplicate entities)
  -> run_all.py (orchestrator, saves JSON)
  -> graph_schema.py (deterministic ID generation + GraphEntity/GraphRelation models)
  -> build_graph.py (Neo4j: entities as :Entity nodes, triples as :RELATES_TO edges)
  -> build_vectorstore.py (Qdrant: BGE-large-en-v1.5 embeddings, two collections)
  -> visualize_graph.py (pyvis interactive HTML)
  -> query_engine.py (hybrid: semantic search + graph traversal)
```

### Dual-database ID scheme

The critical design: both databases use the same deterministic IDs so queries can cross-reference.

- **Entity IDs**: `ent:<normalized_name>` (e.g., `ent:atp_synthase`)
- **Triple IDs**: `triple:<sha256_hex_16>` from `head|relation|tail`
- **Qdrant point IDs**: `uuid5(entity_id)` or `uuid5(triple_id)` — deterministic UUID from the string ID

All ID generation lives in `graph_schema.py`. The `normalize_name()` function lowercases, strips, and replaces spaces with underscores.

### Qdrant collections
- `entities` — one point per entity, embedding of the entity name, payload has `entity_id`
- `evidence` — one point per triple, embedding of the evidence sentence, payload has `triple_id`, `head_entity_id`, `tail_entity_id`

### Neo4j schema
- `:Entity` nodes with `entity_id`, `name`, `original_names` properties
- `:RELATES_TO` edges with `triple_id`, `relation`, `evidence` properties
- Uniqueness constraint on `Entity.entity_id`, index on `RELATES_TO.triple_id`

## Dependencies

| Package | Purpose |
|---|---|
| `python-dotenv` | Load `.env` (API keys, model config) |
| `requests` | HTTP calls to OpenRouter API |
| `pydantic>=2.0` | Triple/ExtractionResult/GraphEntity/GraphRelation validation |
| `pymupdf` | Local PDF text extraction (default backend) |
| `neo4j>=5.0` | Neo4j Python driver (Bolt protocol) |
| `qdrant-client>=1.7` | Qdrant Python SDK (supports `:memory:` mode for tests) |
| `sentence-transformers>=2.2` | Embedding model (BAAI/bge-large-en-v1.5, 1024-dim) |
| `pyvis>=0.3` | Interactive HTML graph visualization |
| `networkx>=3.0` | Graph data structure for visualization |
| `pytest>=7.0` | Test framework |

## Configuration

All config lives in `.env` (git-ignored) and `src/config.py`:
- `OPENROUTER_API_KEY` — required for LLM extraction
- `NEO4J_URI` (default: `bolt://localhost:7687`), `NEO4J_USER` (default: `neo4j`), `NEO4J_PASSWORD` (default: `password`)
- `QDRANT_URL` (default: `localhost`), `QDRANT_PORT` (default: `6333`)
- `EMBEDDING_MODEL` (default: `BAAI/bge-large-en-v1.5`), `EMBEDDING_DIM` (default: `1024`)
- `MODEL_NAME` — LLM model for extraction

## Key design decisions
- **Schema-free extraction**: No predefined entity/relation types; the LLM generates labels from context.
- **Single-shot prompting**: One API call per passage, no iterative refinement.
- **Deterministic IDs**: Same input always produces the same ID across both DBs — no lookup table needed.
- **Idempotent upserts**: Both Neo4j (MERGE) and Qdrant use deterministic IDs, so re-running is safe.
- **Graceful degradation**: `run_all.py` catches connection errors for Neo4j/Qdrant and continues. Pass `--no-graph` to skip entirely.
- **Tests use in-memory Qdrant**: No server required for vectorstore tests. Neo4j tests auto-skip if unavailable.

## Planned (not yet implemented)
- BGE-reranker-v2-m3 for passage reranking
- Hierarchical chunking with LLM router
- Precision/Recall/F1 evaluation against gold-standard annotations

## File naming note

`extract_entitites.py` has a typo ("entitites") — this is the current filename, do not rename without updating all imports in `run_all.py`.
