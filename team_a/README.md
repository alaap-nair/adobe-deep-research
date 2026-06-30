# Team A: Unified Neo4j Store

Team A is the unified-store implementation. It keeps the knowledge graph,
source chunks, and chunk embeddings in one Neo4j database.

- **Graph structure**: `(:Entity)` nodes + relationships from extracted triples
- **Vectors inside Neo4j**: `(:Chunk)` nodes with `embedding` vectors + a Neo4j **vector index**

## Setup

Run these commands from `team_a/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in `team_a/` or in the repo root:

```env
OPENROUTER_API_KEY=...
MISTRAL_API_KEY=...        # optional, used for PDF OCR in src/parse_pdf.py
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
EMBEDDING_MODEL=openai/text-embedding-3-small
QA_MODEL=meta-llama/llama-3.1-8b-instruct
```

Start Neo4j locally before graph build/query. One simple option is:

```bash
docker run --rm --name team-a-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5-community
```

## Run The Pipeline

Extract entities/triples from a text or PDF source:

```bash
python src/run_all.py data/passage.txt
```

For `data/passage.txt`, this writes `triples_passage.csv`.

Build the Neo4j graph, chunks, and vector index:

```bash
python src/build_graph.py --input data/passage.txt --triples triples_passage.csv
```

Ask a question against the Neo4j graph/vector index:

```bash
python ask.py "What does glycolysis produce?"
```

The CLI prints JSON with `question`, `answer`, `citations`, `reasoning`, and a
`kg_trace`, and appends runs to `outputs.txt`.

## Other Useful Commands

Run the local biology QA baseline:

```bash
python bio_qa.py --plain "What is DNA?"
```

Run the RAGAS comparison scripts:

```bash
python run_ragas_comparison.py
python ragas_eval.py data/after.json
```

## Visualize In Neo4j

In Neo4j Browser:

```cypher
MATCH (e:Entity)-[r]->(e2:Entity)
RETURN e, r, e2
LIMIT 50;
```

```cypher
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN c, e
LIMIT 50;
```

### Documentation links (references)

- Neo4j Vector Indexes: `https://neo4j.com/docs/cypher-manual/5/indexes/semantic-indexes/vector-indexes/`
- Neo4j Python Driver: `https://neo4j.com/docs/python-manual/current/`
- OpenRouter Embeddings API: `https://openrouter.ai/docs/api/reference/embeddings`
- Mistral OCR endpoint: `https://docs.mistral.ai/api/endpoint/ocr`
