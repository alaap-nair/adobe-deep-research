# adobe-deep-research

Do not commit API keys. Use a `.env` file in the repo root and keep it out of git.

## Setup

Use the local virtual environment:

```bash
source .venv/bin/activate
```

If you need the older OpenRouter setup from the existing repo, keep these values in `.env`:

```env
OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxx
```

This project also supports OpenAI-compatible settings for answer synthesis:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
```

For Neo4j-backed retrieval, also set:

```env
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

## Biology QA

Put biology source material into `data/bio/` as `.txt`, `.md`, or `.pdf` files, then run:

```bash
.venv/bin/python bio_qa.py --backend neo4j --answer-mode llm --plain "What is DNA?"
```

If you want retrieval without LLM synthesis:

```bash
.venv/bin/python bio_qa.py --backend neo4j --plain "What is DNA?"
```

To ingest or refresh the Neo4j graph:

```bash
.venv/bin/python bio_qa.py --backend neo4j --ingest-neo4j --reset-graph
```

The graph stores `BioSource`, `BioChunk`, `BioTerm`, and `BioConcept` nodes with `HAS_CHUNK`, `HAS_TERM`, `MENTIONS`, `RELATED_TO`, and `NEXT` relationships.

To save the answer in evaluation-ready JSON:

```bash
.venv/bin/python bio_qa.py "What happens during mitosis?" --save results/mitosis.json
```

## RAGAS

The evaluator expects JSON rows with at least:

- `user_input`
- `response`
- `retrieved_contexts`

Example:

```bash
.venv/bin/python ragas_eval.py data/eval/sample_eval.json
```

## Existing Repo Files

The original repository content is still present under `src/` and related files such as `requirements.txt` and `data/passage.txt`.

## Files

- `bio_qa.py`: local or Neo4j-backed retriever plus extractive answer builder for `.txt`, `.md`, and `.pdf` sources
- `neo4j_bio_graph.py`: Neo4j schema, ingestion, and chunk retrieval
- `ragas_eval.py`: RAGAS evaluator
- `src/`: original graph-building scripts already present in the remote repo
