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

Pick an architecture and follow its README. Do not commit API keys; keep them in
a `.env` file in the repo root or in the team subdirectory you are running from.
Both directories are self-contained, so run commands from inside `team_a/` or
`team_b/`.

```bash
cd team_a   # or team_b
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimum shared `.env` values:

```env
OPENROUTER_API_KEY=sk-...
MODEL_NAME=meta-llama/llama-3.1-8b-instruct
QA_MODEL=meta-llama/llama-3.1-8b-instruct
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

Team A uses one Neo4j store for graph data and chunk vectors. Team B uses Neo4j
for the graph and Qdrant for vectors; `team_b/docker-compose.yml` starts both
services with the defaults above.

## Quick demo commands

Team A unified-store demo:

```bash
cd team_a
source .venv/bin/activate
python src/run_all.py data/passage.txt
python src/build_graph.py --input data/passage.txt --triples triples_passage.csv
python ask.py "What does glycolysis produce?"
```

Team B dual-store CLI demo:

```bash
cd team_b
source .venv/bin/activate
docker compose up -d
python src/run_all.py data/passage.txt
python ask.py "What does glycolysis produce?"
```

Team B Streamlit demo:

```bash
cd team_b
source .venv/bin/activate
streamlit run app.py
```

See [`team_a/README.md`](team_a/README.md) and
[`team_b/README.md`](team_b/README.md) for architecture-specific setup,
environment variables, evaluation scripts, and demo notes.
