# Assignment 11 — Part 4: Live Personalized Graph Demo

**Team 2 (Jason & Shreyas) · Dual-store KG/RAG**

A working, demo-ready visualization of the **personalized knowledge-graph** feature we
researched in Assignment 10 (Graphiti). It runs the full workflow from the A10 diagram end
to end with real data: a user interacts → an **episode** is logged → **entities are extracted
and linked to a per-user profile node** → the **personalized context is retrieved and surfaced**
in the answer. Built in Adobe's Spectrum design language for the live POC demo.

---

## What the demo shows

The demo follows **Maya**, a cell-biology student studying photosynthesis, through a guided,
presenter-driven step-through. Each step maps directly to the Graphiti workflow:

| Stage | What happens | Workflow step |
|---|---|---|
| **Episode 1 — Query** | Maya asks *"How does the Calvin cycle fix carbon?"* | Episode logged → entities (Calvin cycle, RuBisCO, CO₂, RuBP, 3-PGA, G3P, stroma) extracted and linked to her profile node |
| **Episode 2 — Upload** | Maya uploads her lecture notes on the light reactions | A second entity cluster (Photosystem II/I, electron transport chain, ATP synthase, NADPH, ATP…) forms in her subgraph |
| **Episode 3 — Query** | Maya asks *"What links the light reactions to the Calvin cycle?"* | Temporal **bridge edges** (ATP/NADPH → Calvin cycle) connect the two clusters |
| **Ask (live)** | A new question — *"Walk me through how light energy ends up stored in sugar."* — runs through the pipeline | Personalized context is retrieved from Maya's subgraph |
| **Compare** | The same question, answered two ways | **Generalized** (no personal context) vs **Personalized** (Maya's subgraph injected) |

The payoff is the **Compare** screen: the generalized answer is a correct textbook explanation;
the personalized answer explicitly ties the explanation back to **what Maya previously asked and
uploaded**, and cites her own episodes. This is the concrete proof that the per-user graph changes
the outcome — not just decorates it.

---

## What is live vs. simulated (full transparency)

Per the assignment, here is exactly what runs live versus what is curated for presentation:

| Component | Status |
|---|---|
| Maya's three episodes ingested into Neo4j via **real Graphiti** (`add_episode`) | **LIVE** — real LLM entity/edge extraction; **17 entities** are created in Neo4j under `group_id="maya"` and are inspectable in the Neo4j Browser. |
| The final **Generalized vs Personalized** answer | **LIVE** — `POST /ask` runs a real Graphiti hybrid search over Maya's subgraph and synthesizes both answers with an LLM at demo time. |
| The episode panels and the animated subgraph growth | **Curated** — corpus-grounded data (the real OpenStax Biology Ch. 8 entities) with stable IDs, so the on-screen graph is clean and legible. The same content is shown flowing through live Graphiti during ingestion. |
| The deployed (shareable) link | **Snapshot replay** — a frozen capture of the real run, so the link works with no backend. The UI shows a `Snapshot` badge; when connected to the live backend it shows `Live · Graphiti`. |

Both the live ingest and the live answer use real Graphiti; the curated visual exists only so the
graph reads cleanly on screen.

---

## Architecture

- **Frontend:** Next.js + React Flow + Tailwind (Adobe Spectrum-light theme, Adobe red accent,
  Source Sans type). Presenter step-through with animated graph growth and a side-by-side answer
  comparison. Deployable to Vercel.
- **Backend:** FastAPI wrapping **`getzep/graphiti`** on our existing Neo4j instance (the same
  dual-store infrastructure from Parts 1–3 — Graphiti's persistence layer sits on Neo4j as the A10
  research described). Entity extraction via LLM; embeddings reuse our local `BAAI/bge-large-en-v1.5`
  model (no extra embedding cost). Each user is partitioned by `group_id`.
- **Data:** OpenStax Biology Ch. 8 (photosynthesis) — the same corpus the rest of our system uses,
  so the entities are authentic.

---

## How to view it

- **Live demo (local):** bring up Neo4j, seed Maya's episodes, and run the FastAPI backend + the web
  app; the badge reads `Live · Graphiti`. Full commands are in `demo/README.md`.
- **Shareable link:** the deployed app replays the frozen real run (badge reads `Snapshot`).
- **Screenshots / recording:** see `demo/assets/` (episode-by-episode graph growth, the live badge,
  and the Generalized-vs-Personalized comparison). A short screen recording of the live run is
  included with this submission.

---

## Why it matters

This demonstrates the architectural difference personalization makes in practice: the same retrieval
pipeline, given a per-user temporal subgraph, produces an answer grounded in *that user's* history —
with citations that trace back to their own documents and questions. It's the building block for a
Graphiti-powered personalization layer sitting in parallel with our existing dual-store retrieval,
exactly as proposed in the A10 research.
