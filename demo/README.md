# Live Personalized Graph Demo — Assignment 11, Part 4

A polished, Adobe Spectrum-styled walkthrough of the **Graphiti-powered personalized
knowledge graph**: a user (Maya) queries and uploads → each interaction is logged as an
**episode** → entities are extracted and linked to her **user-profile node** → the
**personalized context is retrieved and surfaced** in a side-by-side answer comparison.

```
demo/
  web/       Next.js + React Flow frontend (Adobe Spectrum light, Vercel-deployable)
  backend/   FastAPI service wrapping live Graphiti on the project's Neo4j
```

## What is live vs. curated (disclose this when presenting)

| Piece | Status |
|---|---|
| Maya's 3 episodes ingested into Neo4j via Graphiti (`seed_maya.py`) | **Live** — real `add_episode` calls, real LLM extraction. Inspect in Neo4j Browser. |
| Final Generalized-vs-Personalized answer (`POST /ask`) | **Live** — real Graphiti hybrid search over Maya's `group_id` + live LLM synthesis. |
| Episode panels and the animated subgraph (`/episodes`, `/subgraph`) | **Curated** — corpus-grounded snapshot data (real OpenStax Ch.8 entities) for a clean, stable visual. |
| Deployed Vercel link | **Snapshot replay** — the frozen real run; no backend. The UI shows a `Snapshot` badge. |

## Run the frontend (works standalone — snapshot fallback)

```bash
cd demo/web
npm install
npm run dev          # http://localhost:3000
```

With no backend running, the app transparently uses `public/snapshot.json` and shows a
`Snapshot` badge. This is the path the Vercel deployment uses.

## Run the live backend (full Graphiti)

Prerequisites:
1. **Neo4j** up: from repo root `docker compose up -d neo4j` (bolt on :7687).
2. **An LLM credential** in `.env`:
   - `ANTHROPIC_API_KEY` (preferred → Graphiti uses `claude-haiku-4-5`, synthesis uses `claude-opus-4-8`), or
   - `OPENROUTER_API_KEY` (→ `OpenAIGenericClient` at OpenRouter; set `GRAPHITI_LLM_MODEL`, default `openai/gpt-4o-mini`).
   - Embeddings are **local** (`BAAI/bge-large-en-v1.5` via sentence-transformers) — no embedding key needed.
3. **Python** with deps: `pip install -r demo/backend/requirements.txt` (uses the project's existing `sentence-transformers` / `neo4j`).

```bash
# from repo root
python -m demo.backend.seed_maya --clear     # ingest Maya's episodes (one-time)
uvicorn demo.backend.server:app --port 8000  # serve the API
python -m demo.backend.snapshot              # (optional) freeze live /ask into snapshot.json
```

Point the frontend at the backend:

```bash
cd demo/web
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

The badge flips to `Live · Graphiti` when the backend answers. Inspect Maya's real
subgraph in the Neo4j Browser (http://localhost:7474):

```cypher
MATCH (n {group_id: "maya"}) RETURN n LIMIT 100
```

## Deploy the shareable link

```bash
cd demo/web && vercel deploy
```

The deployed app ships `public/snapshot.json` and runs entirely client-side (the live
backend is local-only), so the POC link replays the same real run.
