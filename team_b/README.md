# Team B: Dual Neo4j + Qdrant Store

Team B is the decoupled implementation. Neo4j stores the typed knowledge graph,
Qdrant stores vectors for entities, evidence, and chunks, and deterministic IDs
link results across both stores.

## Setup

Run these commands from `team_b/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in `team_b/` or in the repo root:

```env
OPENROUTER_API_KEY=...
MODEL_NAME=meta-llama/llama-3.1-8b-instruct
QA_MODEL=meta-llama/llama-3.1-8b-instruct

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional. Omit QDRANT_URL to use local file storage under ./qdrant_data.
QDRANT_URL=http://localhost:6333
QDRANT_PATH=./qdrant_data
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIM=1024
```

Start the stores:

```bash
docker compose up -d
```

By default, `docker-compose.yml` uses `neo4j/password`, matching the env values
above. Qdrant can also run in local file mode if `QDRANT_URL` is unset, but the
Docker flow is the easiest way to reproduce the full demo.

## Ingest

Build the triples, graph, and vector stores before asking questions:

```bash
python3 src/run_all.py data/passage.txt
```

Useful variants:

```bash
python3 src/run_all.py data/biology_7_2.txt
python3 src/run_all.py data/genetics/14.2\ DNA\ Structure\ and\ Sequencing\ -\ Biology\ 2e\ _\ OpenStax.txt
python3 src/run_all.py data/passage.txt --no-graph
```

Ingestion writes extracted triples under `outputs/`, updates Neo4j/Qdrant when
available, and writes `outputs/graph_visualization.html` for inspection.

## Ask

Query the system with the CLI deliverable:

```bash
python3 ask.py "What does glycolisis produce"
```

The command should print JSON with `question`, `answer`, `citations`, and `reasoning`, and also write matching files under `outputs/answers/`.

Temporal query examples:

```bash
python3 ask.py --as-of 2024-01-31 "What is the ETC located in?"
python3 ask.py --include-invalid "What is the ETC located in?"
```

## Chat UI

A clean Streamlit chat interface over the same dual-store pipeline, showing the
final answer, citations/evidence, retrieved chunks, and the grounding graph
nodes/edges (Assignment-10 demo format).

```bash
# 1. Start the stores and ingest a corpus (see "Ingest" above)
docker compose up -d            # Neo4j + Qdrant
python3 src/run_all.py data/passage.txt

# 2. Launch the UI
streamlit run app.py            # http://localhost:8501
```

The UI serves the **benchmark-winning** retrieval config from Assignment 10
Part 2 (`bge-reranker-v2-m3`) by default; a sidebar toggle lets you compare it
against the no-rerank baseline. An **Answer mode** switch toggles between a
**Generalized** answer (standard KAG pipeline) and a **Personalized** answer
(routed through the Graphiti per-user graph seam in `src/personalization.py` —
currently a clearly-labeled preview stub; the retrieval and answer are real).

Architecture: `app.py` (Streamlit) → `src/ui_backend.py` (`ask()` adapter) →
`QueryEngine.retrieve_context()` + `qa_client.answer_question()`.

## Evaluation And Experiments

Run the included test suite:

```bash
pytest
```

Run RAGAS/model comparison helpers:

```bash
python3 src/run_ragas.py
./scripts/run_rerank_experiment.sh
./scripts/run_precision_experiments.sh
python3 scripts/summarize_model_runs.py
```

The tracked `outputs/` directory contains prior experiment artifacts, model
runs, traces, and visualizations so the Adobe team can inspect previous results
and run follow-up experiments from the same baseline.
