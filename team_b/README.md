# adobe-deep-research

Do NOT commit your API key.
Use a .env file.
Ensure .env is in .gitignore.
If you accidentally commit a key, revoke it immediately.

Please add a .env file in the root of the repo. In your .env file, add this:
OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxx

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Ingest

Build the triples, graph, and vector stores before asking questions:

```bash
python3 src/run_all.py data/passage.txt
```

## Ask

Query the system with the CLI deliverable:

```bash
python3 ask.py "What does glycolisis produce"
```

The command should print JSON with `question`, `answer`, `citations`, and `reasoning`, and also write matching files under `outputs/answers/`.

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
