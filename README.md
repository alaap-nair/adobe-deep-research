# Biology QA with RAGAS

This workspace now has a minimal retrieval QA flow for biology questions, optional Neo4j-backed retrieval, and a separate `ragas` evaluation script.

## Setup

Use the local virtual environment:

```bash
source .venv/bin/activate
```

If you want LLM-backed RAGAS metrics, create `.env` from `.env.example` and set `OPENAI_API_KEY`.

For Neo4j-backed retrieval, also set:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

## Ask a biology question

Put your biology source material into `data/bio/` as `.txt`, `.md`, or `.pdf` files, then run:

```bash
.venv/bin/python bio_qa.py "What happens during mitosis?"
```

For plain terminal output instead of JSON:

```bash
.venv/bin/python bio_qa.py --plain "Which organisms are classified as eukaryotes?"
```

The imported biology textbook can stay in `data/bio/` and will be searched alongside any other notes you add.

## Use Neo4j as the pipeline

Ingest every source file under `data/bio/` into Neo4j:

```bash
.venv/bin/python bio_qa.py --backend neo4j --ingest-neo4j --reset-graph
```

Then ask questions against the graph:

```bash
.venv/bin/python bio_qa.py --backend neo4j --plain "Which organisms are classified as eukaryotes?"
```

For the best terminal answers, use OpenAI synthesis on top of the Neo4j retrieval:

```bash
.venv/bin/python bio_qa.py --backend neo4j --answer-mode llm --plain "How do enzymes achieve syn-addition in the citric acid cycle despite the thermodynamic preference for anti-addition?"
```

You can also ingest and ask in one command:

```bash
.venv/bin/python bio_qa.py --backend neo4j --ingest-neo4j --plain "What is DNA?"
```

The graph stores `BioSource`, `BioChunk`, `BioTerm`, and `BioConcept` nodes with `HAS_CHUNK`, `HAS_TERM`, `MENTIONS`, `RELATED_TO`, and `NEXT` relationships.

To save the answer in evaluation-ready JSON:

```bash
.venv/bin/python bio_qa.py "What happens during mitosis?" --save results/mitosis.json
```

## Run RAGAS

The evaluator expects JSON rows with at least:

- `user_input`
- `response`
- `retrieved_contexts`

If you also provide `reference`, it will run additional reference-based metrics.

Example:

```bash
.venv/bin/python ragas_eval.py data/eval/sample_eval.json
```

Or against saved answer files:

```bash
.venv/bin/python ragas_eval.py results/mitosis.json
```

## Files

- `bio_qa.py`: local or Neo4j-backed retriever plus extractive answer builder for `.txt`, `.md`, and `.pdf` sources
- `neo4j_bio_graph.py`: Neo4j schema, ingestion, and chunk retrieval
- `ragas_eval.py`: RAGAS 0.4.3 evaluator using OpenAI-backed metrics
- `data/bio/`: sample biology documents
- `data/eval/sample_eval.json`: sample evaluation dataset
