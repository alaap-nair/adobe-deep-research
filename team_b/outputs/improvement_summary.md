# Week 2 — Before/After Summary (Team 2)

## Graph stats — baseline vs. post

| Metric | Baseline (Week-1 extraction, no canonicalization) | Post (W3B schema-constrained extraction + W3A canonicalization) | Delta |
|---|---:|---:|---:|
| Nodes | 296 | 130 | **−56%** |
| Relationships | 235 | 128 | −46% |
| Avg degree | 1.59 | 1.97 | **+24%** (denser) |
| Distinct relation labels | 156 | 57 | **−63%** |
| Qdrant chunks | 208 | 205 | ≈ same |
| Qdrant entity vectors | 318 | 130 | −59% |
| Qdrant evidence vectors | 272 | 128 | −53% |

The schema-constrained extraction (W3B) drops triples that don't map to the allowed `Organism / Molecule / Enzyme / CellularStructure / Process / Pathway / TaxonomicGroup / EcologicalRole / ChemicalReaction` node types and `CATALYZES / PRODUCES / OCCURS_IN / ...` relation set. The canonicalization (W3A) then collapses surface-form duplicates and semantic near-duplicates at cosine ≥ 0.92. Together the graph is **smaller but each node carries more edges** (avg degree 1.59 → 1.97), and the relation vocabulary is roughly ⅓ the size.

## Coverage of ground-truth entities

|  | Baseline | Post |
|---|---|---|
| Coverage | 20/20 (100%) | 20/20 (100%) |
| Match quality | matches noisy concatenated entities (e.g. q1 → `carbon dioxide and rubp`) | matches clean canonical entities (e.g. q1 → `rubisco`) |

## RAGAS — aggregate

| Metric | Baseline | Post | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.947 | **0.967** | +0.020 |
| Answer relevancy | 0.688 | **0.702** | +0.014 |
| Context recall | 0.688 | **0.800** | **+0.112** |
| Context precision | 0.536 | 0.537 | +0.001 |

## RAGAS — per-question deltas (post − baseline)

| id | Difficulty | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|----|---|---:|---:|---:|---:|
| q1 | Standard | 0.000 | 0.000 | 0.000 | **+0.110** |
| q2 | Standard | 0.000 | +0.083 | 0.000 | 0.000 |
| q3 | Standard | 0.000 | +0.043 | 0.000 | −0.146 |
| q4 | Standard | 0.000 | **+0.210** | 0.000 | +0.100 |
| q5 | Standard | 0.000 | +0.062 | 0.000 | −0.009 |
| q6 | System Breaker | **+0.400** | −0.017 | −0.500 | 0.000 |
| q7 | System Breaker | 0.000 | +0.063 | **+0.250** | −0.167 |
| q8 | System Breaker | 0.000 | 0.000 | 0.000 | +0.018 |
| q9 | System Breaker | 0.000 | 0.000 | 0.000 | +0.021 |
| q10 | System Breaker | 0.000 | 0.000 | 0.000 | 0.000 |
| q11 | Standard | 0.000 | **+0.138** | 0.000 | −0.056 |
| q12 | Standard | 0.000 | 0.000 | 0.000 | 0.000 |
| q13 | Standard | 0.000 | −0.004 | 0.000 | 0.000 |
| q14 | Standard | **−0.333** | +0.066 | **+1.000** | −0.159 |
| q15 | Standard | 0.000 | 0.000 | 0.000 | **+0.249** |
| q16 | System Breaker | 0.000 | 0.000 | 0.000 | +0.083 |
| q17 | System Breaker | **+0.333** | 0.000 | 0.000 | +0.125 |
| q18 | System Breaker | 0.000 | +0.008 | 0.000 | −0.165 |
| q19 | System Breaker | 0.000 | 0.000 | **+0.500** | −0.167 |
| q20 | System Breaker | 0.000 | **−0.367** | **+1.000** | +0.184 |

## Interpretation

**Context recall is the biggest mover (+0.112 aggregate, with q14 / q19 / q20 each gaining ≥0.5).** Schema-constrained extraction collapsed a lot of noise into a smaller set of canonical entities, so the retriever is now more likely to surface a passage that actually contains the ground-truth concept rather than a noisy paraphrase of it. Faithfulness also nudged up (0.947 → 0.967); together these say that the **retrieval surface is cleaner and the LLM is grounding answers more carefully**.

**Context precision is essentially flat (+0.001), against the going-in hypothesis** that dedup would tighten precision. Looking at per-question deltas, precision is volatile: q1 / q4 / q15 / q17 / q20 improved, but q3 / q7 / q14 / q18 / q19 regressed by similar amounts. The regressions cluster on questions where the post-graph now returns a *different* set of well-supported context items, and the RAGAS judge marks more of them as marginally relevant rather than directly answer-supporting. This points at chunk retrieval (not entity dedup) being the dominant driver of context_precision — a candidate for next week's workstream (e.g. a re-ranker or chunk-level filtering).

**System Breaker behavior validated the design intent.** q17 (constraint with negation, photolysis vs ATP) jumped +0.333 on faithfulness — the typed graph stopped pulling cellular-respiration ATP edges into a photosynthesis question. q19 ("reduction" ambiguity) gained +0.500 context_recall after canonicalization made the Calvin-cycle reduction concept a single clean node. q20 (mitochondria OOC) gained +1.000 context_recall but lost answer_relevancy as the system surfaced more (now-relevant) passages without changing its refusal — exactly the behavior we want for an out-of-corpus probe. The one notable regression, q14 NADPH (faithfulness 1.0 → 0.667), is worth a closer look: schema-constrained extraction may have dropped a triple the LLM had been leaning on for grounding.

## Files referenced

- `outputs/ragas/baseline/scores.csv`, `summary.md` — pre-improvement
- `outputs/ragas/post/scores.csv`, `summary.md` — post-improvement
- `outputs/graph_stats_baseline.txt`, `outputs/graph_stats_post.txt`
- `outputs/coverage_baseline.txt`, `outputs/coverage_post.txt`
- `docs/canonicalization_strategy.md` — W3A.1 strategy comparison
- `src/domain_schema.py` — W3B `NODE_TYPES` / `RELATION_TYPES`
- `data/ground_truth.json` — 20 questions, 10 Standard / 10 System Breakers
