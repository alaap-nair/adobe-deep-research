# W3A — Deduplication & Canonicalization Strategy

## Problem

Open extraction over OpenStax Ch. 8 + Ch. 24 produces near-duplicate entity nodes:

- **Surface noise:** `Lichen,` / `Lichen` / `Lichens`, `Photosystem` / `Photosystems`, trailing punctuation, mixed case.
- **Semantic near-duplicates:** `Photosystem II` / `PSII`, `Glomeromycota fungi` / `Glomeromycota`, `Calvin cycle` / `the Calvin cycle`.

Both kinds inflate node count, fragment evidence across "the same" entity, and hurt RAGAS `context_precision` because retrieval returns redundant hits for what is conceptually one node.

## Strategies considered

We looked at three options before settling on a hybrid.

**Option 1 — pure rule-based normalization.** The cheapest route: Unicode-fold the name, strip punctuation, lowercase it, and lemmatize. It's deterministic, easy to step through in a debugger, and adds zero new dependencies. The catch is that it only collapses things that *look* the same after cleanup. "PSII" and "Photosystem II" will still live as two different nodes because they share no characters once you strip them down. Fine as a first pass, not enough on its own.

**Option 2 — BioBERT embedding clustering.** A biology-tuned language model would catch the semantic near-duplicates that rule-based normalization misses, and would probably do it more accurately than a general-purpose embedder. The cost is real, though: BioBERT is roughly 400 MB on disk, runs on a separate inference path from our existing retrieval stack, and noticeably slows ingest. For an intro-biology corpus (OpenStax Ch. 8 and Ch. 24) the marginal accuracy gain over a strong general-purpose embedder didn't justify the operational complexity, so we passed.

**Option 3 — BGE-large embedding clustering at cosine ≥ 0.92.** BGE-large-en-v1.5 is already in our stack — it's the embedder we use for retrieval. Reusing it for canonicalization means no new model to load, no extra dependencies, and the same vectors that decide retrieval relevance also decide entity merges (which is conceptually nice — if two names are "the same" for retrieval, they should be the same entity). It's not biology-tuned the way BioBERT is, but at the chapter-level vocabulary we're working with, the failure modes are vanishingly rare.

**Decision: hybrid.** Run rule-based normalization first to collapse the syntactic noise (it's free and catches the easy cases), then run BGE-large clustering on the cleaned names to catch the semantic near-duplicates. Skip BioBERT.

## Implementation

`src/canonicalize.py` exposes a two-stage hybrid pipeline:

1. **`rule_normalize(name)`** — Unicode NFKC fold → strip punctuation (keep hyphens) → lowercase → collapse whitespace → WordNet lemmatization (with a Greek/Latin suffix preservation list so `glycolysis` does not become `glycolysi`). Output is a clustering key, not a display name.
2. **`cluster_entities(names, threshold=0.92)`** — embed each name with BGE-large-en-v1.5 (the same model used in retrieval), then greedy single-link clustering: walk in input order, join the first centroid at cosine ≥ threshold or open a new cluster.

`apply_canonicalization(entities, relations)` chains the two stages, then:
- picks a canonical display name per cluster (most surface forms wins, ties broken by shortest then lexicographic),
- accumulates all surface forms onto `GraphEntity.original_names` for transparency,
- rewrites every relation to use canonical IDs and recomputes `triple_id`,
- drops self-loops introduced by merges,
- deduplicates triples that collapse to the same `(head, relation, tail)`.

## Wiring

The pipeline is gated by env var `KG_CANONICALIZE=1` and called inside `graph_schema.build_graph_objects()`, so **both Neo4j and Qdrant ingest see canonical IDs**. No separate dedup pass on the live database is needed.

```bash
# Re-ingest with canonicalization on
rm -rf qdrant_data/
# (also wipe Neo4j: MATCH (n) DETACH DELETE n)
KG_CANONICALIZE=1 python src/run_all.py data/*.pdf
```

## Measurement

`scripts/graph_stats.py` reports node count, edge count, average degree, and the top entities by surface-form count. Run it before and after toggling `KG_CANONICALIZE=1` and paste the deltas into the Group Doc. Re-run RAGAS afterward — the hypothesis is that `context_precision` improves because retrieval no longer returns multiple chunks anchored to fragmented copies of the same entity.
