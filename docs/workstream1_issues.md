# Workstream 1 — Full-Scale Dataset Issue Log

Target corpus: `data/ConceptsofBiology-WEB.pdf` — 615 pages, ~1.62M chars, ~463 LLM extraction chunks.

This file is gitignored. Each entry follows: **Issue → Where → Fix (or Blocked)**.

---

## Pre-run shortcuts identified by code audit (not yet hit at runtime, but known to break at scale)

### S1. LLM extraction is fully sequential with no checkpoint or cache
- **Where:** `src/extract_triples.py:194-220` — chunks iterated one at a time, dedup `seen` set is in-memory only.
- **Why it breaks at scale:** ~463 LLM calls. A single transient 5xx near the end of the run discards the entire batch. Re-runs cost full $$ again.
- **Fix:** disk-backed cache keyed by `sha256(model + system_prompt + chunk_text)` under `outputs/llm_cache/`, plus a per-document checkpoint JSON so a crashed run resumes from the last completed chunk.

### S2. Silent JSON parse failures
- **Where:** `src/extract_triples.py:177-180` — JSON parse error returns `[]` and only prints to stdout.
- **Why it breaks at scale:** at 463 chunks, a 1% failure rate = 4–5 lost chunks worth of triples that nobody notices.
- **Fix:** record failures to `outputs/run_logs/extraction_failures.jsonl` with raw response, and one retry with a stricter "JSON only, no prose" reminder appended.

### S3. Neo4j ingestion is not batched
- **Where:** `src/build_graph.py:43-66` — `upsert_entities` / `upsert_relations` send the whole list in a single `UNWIND` per session.
- **Why it breaks at scale:** at thousands of relations per UNWIND, Neo4j may hit `dbms.memory.transaction.total.max` or stall on a single tx.
- **Fix:** batch in groups of 500 entities / 500 relations per transaction.

### S4. Qdrant embedding has no batch_size, progress bar disabled
- **Where:** `src/build_vectorstore.py:65-70` — `model.encode(texts, normalize_embeddings=True, show_progress_bar=False)`.
- **Why it breaks at scale:** BGE-large + thousands of evidence strings is the most likely OOM. No progress feedback during a long encode.
- **Fix:** pass `batch_size=32`, re-enable progress bar, chunk Qdrant upserts to 256 points per call.

### S5. Entity canonicalization is purely lexical
- **Where:** `src/extract_entitites.py` + `src/graph_schema.py:18-32` — strip articles, lowercase, naive singularize, then ID by `normalize_name`.
- **Why it breaks at scale:** "DNA polymerase" / "DNA polymerases" / "DNA polymerase enzyme" all get distinct entity_ids because `normalize_name` only lowercases-and-underscores. The graph fragments.
- **Fix:** after rule-based pass, run a cosine-similarity merge pass over entity-name embeddings (BGE, threshold ~0.92) and keep an alias map. Cluster centroid name = the most-frequent surface form.

### S6. Single-document orchestrator
- **Where:** `src/run_all.py` — accepts one path, processes one passage, writes one triples JSON.
- **Why it breaks at scale:** the team has been running it 3× by hand for 3 chapter PDFs (`outputs/ingest_chapters.log`). At full corpus we want one command to ingest the whole book.
- **Fix:** new `src/ingest_corpus.py` that takes a PDF or directory, parses (with cached parsed text), runs extraction with checkpointing, then ingests once into Neo4j+Qdrant. Skips pyvis viz when entity count > threshold.

### S7. PyMuPDF returns one giant string with no per-page provenance
- **Where:** `src/parse_pdf.py:34-46` — pages joined with `\n\n` and we lose page numbers.
- **Why it matters at scale:** for citations and debugging, "this triple came from page 247" is much more useful than "this triple came from `ConceptsofBiology-WEB.pdf`".
- **Fix:** new `parse_pdf_pages()` returning a list of `(page_number, text)` and a parsed-text cache under `outputs/parsed/<doc>.jsonl` so we don't re-parse on rerun.

### S8. PyVis at thousands of nodes is unusable
- **Where:** `src/run_all.py:115-117` — always generates HTML.
- **Fix:** skip when entities > 200, write a stats summary instead.

### S9. ExtractionResult.entities expects `list[str]`
- **Where:** `src/schema.py:28-36` — `entities: list[str]` but downstream we'd benefit from carrying canonical → original-names mapping.
- **Fix:** keep backwards compat (still write `list[str]` to JSON for now), but add canonical alias map alongside in the orchestrator output.

---

## Runtime issues encountered during the full-corpus run

### R1. Docker daemon not running (blocker for DB ingest)
- **Where:** `docker ps` returns "failed to connect to the docker API at unix:///Users/shreyasnair/.docker/run/docker.sock".
- **Impact:** Neo4j + Qdrant containers can't start — full-corpus ingest must run with `--no-graph` or wait for Docker Desktop.
- **Fix:** user starts Docker Desktop, then `docker compose up -d`; we re-run with DBs.

### R2. pyvis crashes on import via IPython/prompt_toolkit on Python 3.13
- **Where:** `src/visualize_graph.py:16` → pyvis → IPython → prompt_toolkit → xml.dom.minidom → pyexpat.
- **Symptom:** `ImportError: dlopen … pyexpat.cpython-313-darwin.so: Symbol not found: _XML_SetAllocTrackerActivationThreshold` (Python 3.13 build expects a newer libexpat than ships with macOS).
- **Why it shows up at scale:** the small-test smoke run used to exit after the model used line — at corpus scale we always reach the viz step, which transitively touches pyexpat.
- **Fix:** wrapped pyvis import in a try/except inside `maybe_visualize` so it's a soft skip; the run continues. Long-term: pin Python to 3.12, or replace pyvis with a static graphviz/dot dump.

### R3. (small-input smoke test) End-to-end orchestrator works on `data/biology_7_2.txt`
- **Result:** 33 entities, 35 relations extracted in one pass, per-source triples + corpus triples both written. LLM cache populated (3 calls). After R2 fix, run completes cleanly with `--no-graph`.

### R4. Triple.relation validator was too strict (5 words) and silently dropped real triples
- **Where:** `src/schema.py:19-24` — `relation_is_concise` rejected anything > 5 words.
- **Symptom (caught only at scale):** chunks 13 and 32 of the 615-page corpus produced relations like "can move toward or away from", "is the smallest unit of biological structure", "is a thoroughly tested and confirmed explanation for". With the old rule these were raised as Pydantic errors and the chunks lost those triples (logged to `outputs/run_logs/extraction_failures.jsonl` after I added that file). At small dataset sizes this never tripped because chunks were short and the LLM stayed in 1-3 word relations.
- **Fix:** raised the bound to 8 words. Sentence-level fragments are still rejected, real multi-word biology relations are now kept. Re-running with the disk LLM cache costs no API calls and recovers the dropped triples.

### R5. stdout buffering hid in-flight progress on background extraction
- **Where:** `python src/ingest_corpus.py … > outputs/run_logs/extract_full.log 2>&1` showed an empty log file even after dozens of chunks had completed, because Python buffers stdout when redirected to a file.
- **Why it matters at scale:** the only way to monitor a 60-minute run was via the checkpoint JSON and LLM cache file count — fine for a developer who knows where those live, bad for ops or a teammate.
- **Fix:** for the next run, prepend `python -u` (or `PYTHONUNBUFFERED=1`) so each chunk's `print()` flushes immediately. Not a code change — a runbook note.

### R6. Chunk count was 622, not the 463 we sized for
- **Where:** `src/extract_triples.py:chunk_text` is paragraph-first, sentence-fallback. For dense textbook prose with many short paragraphs, it produces ~33% more chunks than `total_chars / CHUNK_CHAR_LIMIT` predicts.
- **Why it matters:** API-call budgeting was off by 1.3×. Not fatal (resumable + cached), but worth knowing for capacity planning.
- **Fix:** none yet — the LLM cache absorbs the cost on reruns. If we wanted denser chunks we'd add a "minimum chunk size" parameter that greedily concatenates short paragraphs even past the limit. Logged for now.

### R7. Local Qdrant file mode vs Docker Qdrant — only one can win
- **Where:** `src/config.py:24` defaulted `QDRANT_URL=None` → local file mode at `qdrant_data/`. Docker compose also exposes a Qdrant on `localhost:6333` with its own volume. We were ingesting into one and querying from the other.
- **Why it matters at scale:** silent split-brain. `qdrant_data/` had stale collections from earlier work; the docker container started empty.
- **Fix:** set `QDRANT_URL=http://localhost` and `QDRANT_PORT=6333` in `.env` so all of {orchestrator, query_engine, build_vectorstore} talk to the docker Qdrant. The `qdrant_data/` dir is now orphaned and can be deleted (left in place since it's harmless).

### R8. Pre-existing Neo4j data from per-chapter runs inflates node count
- **Where:** Neo4j `MATCH (n:Entity) RETURN count(n)` returns 8197, but our orchestrator only built 6570 entities for this corpus.
- **Why:** prior runs of `run_all.py` on `Ch14`, `Ch15`, `Ch17` left ~1600 nodes in the DB. Deterministic IDs mean MERGE didn't double-count for overlap, but non-overlapping entities from those chapter runs are still there.
- **Fix:** if we want a clean baseline, run `MATCH (n) DETACH DELETE n` before ingest. Not done by default because some teammates rely on the chapter data.

### R9. Working but worth flagging — entity quality at scale
- **Where:** top-degree entities include "scientists" (65 edges) and "proteins" (88) which is plural while many other entities are singular. The lexical singularizer only handles trivial cases; the BGE merge at threshold 0.92 didn't collapse "protein" / "proteins".
- **Why it matters:** the graph fragments slightly along plural/singular lines.
- **Fix candidates (not implemented):** lower the merge threshold to 0.88, or add a stronger morphological normalizer (e.g. spaCy lemma) before ID generation. Logged as a quality-not-stability issue.

### R10. Triple validator recovery on cached re-run
- **Where:** R4 fix (validator: 5 → 8 words) was deployed mid-extraction. The re-run revalidated all 622 cached LLM responses against the new rule and recovered the previously-dropped triples.
- **Result:** 8431 → 8388 triples after embedding canonicalize + dedup. Validation failures dropped from 74 to a comparable count (now mostly genuinely-too-long sentence fragments, not biology relations).

---

## Final corpus run — confirmation

`outputs/run_logs/ingest_corpus_summary.json`:

```json
{
  "model": "mistralai/mistral-small-3.2-24b-instruct",
  "sources": ["data/ConceptsofBiology-WEB.pdf"],
  "num_triples": 8388,
  "num_entities": 6570,
  "num_relations": 8388,
  "alias_map_size": 7611,
  "num_chunks": 2738,
  "neo4j": "ok",
  "qdrant": "ok"
}
```

- **615-page PDF, 1.62M chars** parsed once, cached.
- **622 LLM extraction calls** (Mistral Small 3.2 via OpenRouter), all cached on disk under `outputs/llm_cache/` for free reruns.
- **8431 raw triples** → **8388 after dedup + embedding-based canonicalization** (1041 surface forms collapsed).
- **6570 entities** ingested into Neo4j (batched 500/tx) and Qdrant (BGE-large embeddings, batch_size=32).
- **2738 paragraph-based chunks** indexed for retrieval.
- End-to-end QA validated: query "What does glycolysis produce?" returns relevant triples (`Glycolysis --produces--> ...`) and citation-able chunks from the textbook.

### Code artifacts produced
- `src/ingest_corpus.py` — new multi-document orchestrator, replaces hand-running `run_all.py` per file.
- `src/extract_triples.py` — LLM disk cache + checkpoint resume + structured failure log + 5xx backoff.
- `src/extract_entitites.py` — `build_alias_map()` for embedding-based entity canonicalization.
- `src/build_graph.py` — batched Neo4j upserts (500/tx).
- `src/build_vectorstore.py` — `batch_size=32`, progress bar, 256-point Qdrant upsert chunking.
- `src/parse_pdf.py` — `parse_pdf_pages()` for per-page provenance.
- `src/schema.py` — relaxed relation-length validator from 5 → 8 words.
- `src/ingest_corpus.py:maybe_visualize` — pyvis is now soft-skipped when graph > 200 entities or import fails.

### Open follow-ups (not blockers for this workstream)
- R6 (chunk-count budgeting refinement)
- R8 (clean-baseline Neo4j wipe before ingest)
- R9 (plural/singular entity merging)
- R2 (long-term: pin Python 3.12 or replace pyvis)
