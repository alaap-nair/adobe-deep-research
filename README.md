# Adobe × UpSync — Deep Research over a Biology Knowledge Graph

Replicate large-model "deep research" quality with **small, lightweight LLMs** grounded in a
domain-specific **Knowledge Graph + RAG** over OpenStax Biology — instead of SERP-based web
retrieval. Proof-of-concept domain: a "Biology Expert" over OpenStax Biology 2e
(Ch. 8 Photosynthesis, Ch. 14–15 DNA/Genes, Ch. 24 Fungi).

## Two architectures, side by side

The project evaluates **two competing KG-RAG architectures** built by two sub-teams. Each lives
in its own self-contained subdirectory so they can be run and benchmarked independently:

| Directory | Sub-team | Architecture |
|-----------|----------|--------------|
| [`team_a/`](team_a/) | Camillia & Shreya | **Unified single-store** — knowledge graph, text chunks, and vector embeddings all live inside Neo4j. Simpler infra, fewer moving parts. |
| [`team_b/`](team_b/) | Jason & Shreyas | **Decoupled dual-store** — Neo4j for the graph + Qdrant for vectors, linked by a shared deterministic ID scheme. Each store optimized for its retrieval type. |

Both share the broad pipeline (ingest → entity/triple extraction → KG construction → hybrid
retrieval → answer synthesis), BGE embeddings, a DeepSeek/Mistral-class synthesis model, and
RAGAS evaluation. They diverge in storage, parsing, extraction, chunking, and reranking — see
each subdirectory's own README for setup and run instructions.

## Layout

```
team_a/   # unified single-store (Neo4j-only) implementation + its data/outputs/tests
team_b/   # decoupled dual-store (Neo4j + Qdrant) implementation + its data/outputs/tests
.env.example
```

This top-level branch consolidates the two team branches (`team1` → `team_a/`, `team2/…` →
`team_b/`) via subtree merges, so each subdirectory retains its full git history. Earlier
exploratory branches (`team-1`, `alaap/assignment-two`) are preserved as `archive/*` tags.

## Getting started

Pick an architecture and follow its README. Do not commit API keys — keep them in a
`.env` (gitignored).

```bash
cd team_b   # or team_a
cat README.md
```

Each subdirectory has its own `requirements.txt` and `.env` expectations
(an `OPENROUTER_API_KEY`, plus local Neo4j / Qdrant).
