# Assignment #10 — Ground Truth + Context-Precision Refinement (Team B, dual-store)

> Scope of this doc: **Part 1 (Ground Truth)** and **Part 2 (Context Precision)**, the two
> parts implemented in code. Parts 3–5 (demos, failure showcase, personalized-graph R&D) are
> document/research deliverables prepared separately in the submission tabs.
> Architecture: decoupled dual-store (Neo4j graph + Qdrant vectors), BGE-large embeddings.

## Part 1 — Ground Truth via a larger model

**What we did.** Regenerated the reference answer for all 40 evaluation questions in
`data/ground_truth.json` using **`anthropic/claude-opus-4.8`** (via OpenRouter) — the most
capable model the assignment named ("opus 4.8 if your tuff"). The generator
(`scripts/gen_ground_truth.py`) is *corpus-grounded*: for each question it retrieves the top-10
passages from the dual-store via `QueryEngine.retrieve_context()`, hands them to Opus 4.8, and
asks for an authoritative, self-contained reference answer. The model also self-reports a
`grounded` flag; if the passages don't support an answer it sets `grounded=false` and falls
back to its own (still factual) biology knowledge.

**Why grounded generation.** Ground-truth answers are what RAGAS `context_recall` and
`context_precision` are scored against. Grounding the gold answers in the actual ingested
corpus keeps recall fair (the reference's claims are findable in-corpus) instead of penalizing
retrieval for facts that only exist in the model's pretraining.

**Provenance preserved.** The original hand-written set was backed up to
`data/ground_truth.curated.json`, and every row keeps its prior answer under
`ground_truth_human` alongside the new `ground_truth`, `ground_truth_model`, and
`ground_truth_grounded` fields — so the dataset is auditable and reversible.

**Result.** 40/40 regenerated; **35 grounded**, **5 flagged out-of-corpus** (q10 nitrogen-fixing
bacteria, q23 & q30 CRISPR, q29 the definition of "genetics", q40 wobble hypothesis) — matching
the corpus boundaries (OpenStax Ch. 8 + 14 + 15 + 24). The Opus answers are consistently more
complete and mechanistic than the curated ones, e.g.:

> **q1 — "What enzyme catalyzes carbon fixation in the Calvin cycle?"**
> *Curated:* "RuBisCO … attaches CO2 to RuBP, producing two 3-PGA molecules …"
> *Opus 4.8:* "… catalyzed by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO).
> RuBisCO catalyzes the reaction between CO₂ and the five-carbon RuBP, producing two molecules
> of the three-carbon 3-phosphoglyceric acid (3-PGA), thereby incorporating inorganic carbon
> into an organic molecule."

**Reproduce:** `.venv/bin/python scripts/gen_ground_truth.py` (or `--limit N` for a smoke test,
`GT_MODEL=…` to swap the generator).

---

## Part 2 — Improving context precision

**Background / problem.** In Assignment #9 we added a generic MS-MARCO MiniLM cross-encoder
reranker, which *lowered* context_precision (0.567 → 0.543). Manual scoring showed why: the
web-trained cross-encoder rewards lexical overlap with the question's named entities, so on a
multi-hop question like q6 ("How does carbon fixed by RuBisCO … return to the soil through
fungal activity?") it pushed Ch. 8 RuBisCO chunks above the Ch. 24 *decomposition* chunk that
actually closes the answer chain. The diagnosis: the retriever surfaces the right **topic** but
not the right **mechanism**.

**Approach — evaluate several, ship the winner.** We ran a controlled experiment matrix where
**only the retrieval strategy varies** — synthesis model and the RAGAS judge are both pinned to
`mistral-small-3.2-24b`, and all configs are scored against the new Opus-4.8 ground truth.
Each config is isolated under `outputs/ragas/precision_<label>/` and
`outputs/answers/precision_<label>/` (driver: `scripts/run_precision_experiments.sh`).

| Label | Strategy | Mechanism |
|-------|----------|-----------|
| **C0** baseline | default retrieval | top-k from a 2× pool, dedupe-truncate (A9 default) |
| **C1** bge rerank | `USE_RERANKER=1 RERANKER_MODEL=BAAI/bge-reranker-v2-m3` | swap the web cross-encoder for a stronger multilingual/domain reranker over a 3× pool |
| **C2** MMR | `RETRIEVAL_MODE=mmr` | Maximal Marginal Relevance over a **6× pool** — trades a little relevance for diversity so the cross-chapter chunk survives truncation |
| **C3** graph-boost | `RETRIEVAL_MODE=graph_boost` | dual-store advantage: gather entities within N hops of the question's entities in Neo4j, boost chunks that mention those graph-connected entities, over a 6× pool |

New code: `QueryEngine._mmr_select` and `QueryEngine._graph_boost_select`
(`src/query_engine.py`), dispatched by a `RETRIEVAL_MODE` env flag so the default path is
unchanged and reproducible. C1 reuses the existing `src/reranker.py` hook (it already honors
`RERANKER_MODEL`).

### Before / after RAGAS

All four configs scored against the **new Opus-4.8 ground truth**, judge + synthesis pinned to
`mistral-small-3.2-24b`, 40 questions. (Note: the C0 baseline here is **0.592**, re-baselined on
the new ground truth — not directly comparable to A9's 0.567, which used the old curated answers.)

| Config | Faithfulness | Answer Relevancy | Context Recall | **Context Precision** |
|--------|--------------|------------------|----------------|-----------------------|
| **C0** baseline      | 0.941 | 0.652 | 0.819 | 0.592 |
| **C1** bge rerank    | **0.944** | **0.654** | **0.844** | **0.616** |
| **C2** MMR           | 0.922 | 0.637 | 0.808 | 0.557 |
| **C3** graph-boost   | 0.948 | 0.651 | 0.832 | 0.585 |

(Bold = best in column. C3 edges faithfulness by +0.004 but loses on the target metric.)

**Winner: C1 — `BAAI/bge-reranker-v2-m3`.** Replacing the generic, web-search-trained MS-MARCO
MiniLM cross-encoder (which *regressed* precision to 0.543 in A9) with the stronger
`bge-reranker-v2-m3` cross-encoder raised aggregate **context_precision from 0.592 → 0.616
(+0.024)** — and uniquely among the four configs it *dominates* the baseline on **all four**
metrics, also lifting context_recall +0.025 and faithfulness +0.003. This directly confirms the
A9 hypothesis that the reranker *model*, not the rerank *stage*, was the problem: a relevance
model better matched to dense scientific text ranks the right chunks higher without sacrificing
coverage. **Tradeoffs:** (1) bge-reranker-v2-m3 is a ~2.2 GB model vs ~80 MB for MiniLM, so it
adds memory + per-query cross-encoder latency over a 3× candidate pool — acceptable since it's
gated by `USE_RERANKER` and loaded lazily. (2) The gain is *broad, not multi-hop-specific*: q6
(the cross-chapter RuBisCO→fungal-decomposition bridge) stays at 0.167 context_precision in both
C0 and C1, so the hardest multi-hop retrieval is still unsolved — the lift comes from better
ranking across the many single/standard questions. The two alternatives we tried both
underperformed: **C2 MMR** (0.557) traded away too much relevance for diversity, and **C3
graph-boost** (0.585) — boosting chunks that mention graph-neighbor entities — wasn't selective
enough to beat dense retrieval. We ship C1: `bge-reranker-v2-m3` is now the default reranker
model in `src/reranker.py`, still gated behind `USE_RERANKER=1` so the zero-config default path
is unchanged.

### Eval-integrity fixes folded in
Two parser bugs that could corrupt the answers feeding RAGAS were fixed in
`src/qa_client.parse_llm_json`: (1) a bare top-level JSON array no longer raises `AttributeError`
(returns `{}` so defensive defaults apply), and (2) the regex-recovery path now unescapes via
`json.loads` instead of `unicode_escape`, so multi-byte UTF-8 (e.g. "β-carotene") is preserved
rather than mojibaked. Per-label `--out`/`--traces-dir` prevent the C0–C3 runs from colliding.

**Reproduce:** `bash scripts/run_precision_experiments.sh`
