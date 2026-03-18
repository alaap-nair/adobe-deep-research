# adobe-deep-research

Do NOT commit your API key.
Use a .env file.
Ensure .env is in .gitignore.
If you accidentally commit a key, revoke it immediately.

Please add a .env file in the root of the repo. In your .env file, add this:
OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxx


Install:
pip install python-dotenv

"from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment")"

---

## Week 3: Neo4j “Engine” (Unified Store)

This week’s deliverable is a Neo4j-backed engine that stores:

- **Graph structure**: `(:Entity)` nodes + relationships from extracted triples
- **Vectors inside Neo4j**: `(:Chunk)` nodes with `embedding` vectors + a Neo4j **vector index**

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. `.env` (root):

```bash
OPENROUTER_API_KEY=...
MISTRAL_API_KEY=...        # used for PDF OCR in src/parse_pdf.py
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
# optional
NEO4J_DATABASE=neo4j
EMBEDDING_MODEL=openai/text-embedding-3-small
```

### Run

1. Extract triples:

```bash
python src/run_all.py src/7.2_glycolysis.pdf
```

2. Build Neo4j graph + vector index:

```bash
python src/build_graph.py --input src/7.2_glycolysis.pdf --triples triples.csv
```

### Visualize (for screenshot)

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