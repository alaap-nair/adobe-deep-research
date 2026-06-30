# Evaluation Framework Research

## 1. dzhng/deep-research

**Overview:** An open-source iterative research agent that uses LLMs to conduct multi-step web research. The system generates search queries, reads results, and iteratively refines its understanding.

**Evaluation approach:**
- Measures research quality through breadth (number of sources consulted) and depth (iterative refinement steps)
- Evaluates coherence of final research reports
- No formal benchmark suite; quality is assessed through manual review of outputs

**Relevance to our pipeline:**
- Their iterative approach could inform how we refine extraction prompts
- The concept of "evidence grounding" aligns with their source attribution

## 2. OpenAI Deep Research

**Overview:** OpenAI's multi-step research system that can browse the web, synthesize findings, and produce structured reports. Uses chain-of-thought reasoning with tool use.

**Benchmarking methodology:**
- Evaluated on Humanity's Last Exam and GAIA benchmarks
- Measures factual accuracy, source quality, and completeness
- Uses both automated metrics and human evaluation

**Relevance to our pipeline:**
- Their structured output validation is similar to our schema approach
- Multi-step verification (extract → validate → verify against source) mirrors their methodology

## 3. Proposed Eval Metrics for Our Pipeline

### 3.1 Evidence Grounding (Automatable)

**Definition:** Does the `evidence` field in each triple appear in (or closely match) the source text?

**Implementation:**
- Exact substring match: `evidence in source_text`
- Fuzzy match: Use sequence matching (e.g., `difflib.SequenceMatcher`) with a similarity threshold (≥ 0.8)
- This catches hallucinated evidence — triples where the LLM fabricated a supporting sentence

**Priority:** HIGH — this is fully automatable and directly measures faithfulness.

### 3.2 Precision / Recall / F1 Against Gold Standard

**Definition:** Compare extracted triples against a manually-annotated gold standard.

**Implementation:**
- Create gold-standard annotations for test passages (e.g., passage.txt, biology_7_2.txt)
- Match extracted triples to gold triples using entity/relation similarity
- Compute precision (fraction of extracted triples that are correct), recall (fraction of gold triples that were extracted), and F1

**Challenge:** Triple matching is non-trivial — "glucose is broken down in glycolysis" vs "glycolysis breaks down glucose" express the same fact differently.

**Priority:** MEDIUM — requires manual annotation effort but provides the most meaningful quality signal.

### 3.3 Relation Consistency Scoring

**Definition:** Are similar relationships expressed with consistent relation labels?

**Implementation:**
- Cluster relation labels by semantic similarity (e.g., "produces" vs "generates" vs "creates")
- Flag inconsistent labeling across triples
- Can use embedding similarity to detect near-duplicate relations

**Priority:** LOW — useful for graph quality but less critical than accuracy metrics.

## 4. Recommended Evaluation Pipeline

```
Source Text
    │
    ├──→ Extract Triples (LLM)
    │        │
    │        ├──→ Schema Validation (Pydantic) ← Step 1: structural quality
    │        │
    │        ├──→ Evidence Grounding Check    ← Step 2: faithfulness
    │        │
    │        └──→ Gold Standard Comparison    ← Step 3: accuracy (when available)
    │
    └──→ Manual Review Dashboard (future)
```

## 5. Next Steps

1. **Implement evidence grounding checker** in `src/evaluate.py` — first automatable metric
2. **Create gold standard** for `data/passage.txt` (small, manageable)
3. **Benchmark across models** — run pipeline with different LLMs and compare scores
4. **Add relation clustering** as graph quality improves
