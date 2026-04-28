# Full-scale pipeline run log (600+ page stress test)

## Scope
- Objective: remove small-data shortcuts, ingest full corpus (including `src/bio_textbook_full.pdf`), and verify end-to-end retrieval + generation.
- Run command:
  - `venv/bin/python bio_qa.py --backend neo4j --ingest-neo4j --reset-graph --data-dir src`

## Issues encountered and resolutions

### 1) In-memory PDF materialization (scalability risk)
- **What broke / risk:** The original ingestion path loaded full PDF text into one giant string (`read_source_text` + `split_text`), then built a full list of chunks in memory. This is fragile for 600+ pages and can trigger memory spikes.
- **Where:** `bio_qa.py` in `read_source_text`, `iter_source_chunks`, `ingest_into_neo4j`, and local retrieval path.
- **Fix applied:**
  - Added `iter_split_text()` and `iter_source_pieces()` generators to stream chunk creation.
  - PDF extraction now processes page-by-page and yields chunks incrementally.
  - Ingestion now consumes chunk iterators instead of full lists.
  - Local retrieval path switched to streaming top-k ranking (`rank_chunks_streaming`) to avoid loading all chunks at once.
- **Status:** Resolved.

### 2) Neo4j write amplification (per-chunk round-trip bottleneck)
- **What broke / risk:** Original `ingest_source` executed multiple `session.run` calls per chunk (chunk create, terms, concepts, concept-pairs, NEXT edges). At larger sizes this causes heavy network/transaction overhead.
- **Where:** `neo4j_bio_graph.py` in `ingest_source`.
- **Fix applied:**
  - Reworked ingestion to batched `UNWIND $rows` writes with configurable batch size.
  - Added `_batched()` helper.
  - Created chunk, term links, concept links, and `RELATED_TO` in one batch query.
  - Buffered and bulk-wrote `NEXT` edges in batches.
  - Kept cleanup queries for orphan `BioTerm`/`BioConcept`.
- **Status:** Resolved.

### 3) dotenv auto-discovery assertion error in inline scripts
- **What broke:** `load_dotenv()` in stdin-run scripts hit `AssertionError` from `find_dotenv()` stack inspection.
- **Where:** verification helper scripts executed via heredoc.
- **Fix applied:** Use explicit path load: `load_dotenv(Path('/Users/shreyasatheesh/adobe-deep-research/.env'))`.
- **Status:** Resolved (operational workaround; core pipeline unaffected).

### 4) LLM generation stage env mismatch (`OPENAI_API_KEY` vs `OPENROUTER_API_KEY`)
- **What broke:** End-to-end LLM generation failed with:
  - `OSError: OPENAI_API_KEY is required for LLM answer synthesis.`
- **Where:** `bio_qa.py` in `synthesize_answer_with_llm()`.
- **Cause:** Environment had `OPENROUTER_API_KEY` but not `OPENAI_API_KEY`.
- **Fix applied for run:** Aliased key at runtime (`OPENAI_API_KEY <- OPENROUTER_API_KEY`) and set `OPENAI_BASE_URL=https://openrouter.ai/api/v1`.
- **Status:** Resolved for execution.  
- **Note:** If desired, this can be permanently hardened in code by supporting fallback to `OPENROUTER_API_KEY` in `get_openai_client()`.

## Full-corpus ingestion results
- Ingestion output:
  - `Ingested 7.2_glycolysis.pdf: 14 chunks in 0.9s`
  - `Ingested bio_textbook_full.pdf: 2431 chunks in 10.4s`
  - `Ingested ch33.pdf: 97 chunks in 0.5s`
  - `Ingested ch37.pdf: 141 chunks in 0.6s`
  - `Ingested ch41.pdf: 86 chunks in 0.4s`
  - `Ingested ch42.pdf: 125 chunks in 0.8s`
  - `Ingested 6 source files into Neo4j.`

## Post-ingest graph verification
- Node counts:
  - `BioSource`: 6
  - `BioChunk`: 2894
  - `BioTerm`: 16019
  - `BioConcept`: 65001
- Source names in graph:
  - `7.2_glycolysis.pdf`, `bio_textbook_full.pdf`, `ch33.pdf`, `ch37.pdf`, `ch41.pdf`, `ch42.pdf`

## End-to-end run verification
- Retrieval + extractive generation:
  - `venv/bin/python bio_qa.py --backend neo4j --answer-mode extractive --plain "..."`
  - Completed successfully.
- Retrieval + LLM generation:
  - Executed with OpenRouter key alias to `OPENAI_API_KEY`.
  - Completed successfully and returned grounded response text.

## Remaining unresolved blockers
- None currently blocking full-corpus ingestion/retrieval/generation.
