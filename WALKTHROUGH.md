# Week 2: Data Ingestion & Entity Extraction — Walkthrough

This doc walks you through your **Team 1 two-step pipeline** and where to work in the codebase.

---

## 1. Why you saw “no triples” (fixed)

The LLM **did** return triples, but wrapped in markdown:

```text
```json
{ "triples": [ ... ] }
```
```

`json.loads()` was given that whole string, so it failed and the code fell back to `triples = []`.  

**Fix:** In `src/extract_triples.py` we now strip leading ` ```json ` / ` ``` ` and trailing ` ``` ` before parsing, so those triples are saved correctly.

**What to do:** Run again from project root:

```bash
source venv/bin/activate
python src/run_all.py
```

You should see a non-zero “Number of triples” and a filled `triples.csv`.

---

## 2. Where the code lives

| What | Where |
|------|--------|
| **Orchestration** (read text → entities → triples → save CSV) | `src/run_all.py` |
| **Step 1: Entity extraction (GLiNER)** | `src/extract_entitites.py` |
| **Step 2: Relation extraction (LLM)** | `src/extract_triples.py` |
| **Input text** | `data/passage.txt` or `data/7.2_glycolysis.pdf` (see `data/README.md`) |
| **Output triples** | `triples.csv` (project root) |
| **Output schema** | `schema/triples_schema.json`, `schema/pipeline_output_schema.json`; validation in `src/schema.py` |
| **PDF → text** | `src/parse_pdf.py` (textbook PDF ingestion) |
| **Secrets** | `.env` → `OPENROUTER_API_KEY`, `MISTRAL_API_KEY` |

You do **not** need to touch `src/build_graph.py` for the Week 2 deliverable.

---

## 3. Pipeline flow (step-by-step)

Your assignment is: **raw text → structured triplets (Subject → Predicate → Object)**.

### Step 0: Parsing & chunking

- **Input:** For this week you’re allowed to “hack the boring parts”: use a short snippet (e.g. 1–2 pages).
- **Current setup:** Text is in `data/passage.txt`. `run_all.py` reads it as one string (single “chunk”). No PDF/OCR in the loop yet.
- **Later:** When you add Mistral OCR / Docling / LlamaIndex chunking, that will live in a parsing step before `load_text()` or inside a new `src/parse_and_chunk.py`; for now you’re unblocked.

### Step 1: Entity extraction (GLiNER)

- **File:** `src/extract_entitites.py`
- **Role:** From the raw text, extract **nodes** (entities) using a fixed **ontology** (node types).
- **Ontology:** `NODE_TYPES` in that file: Protein, Enzyme, Molecule, Cellular Process, Organelle, Cell Type, Biological Pathway.
- **Output:** A list of entity strings (e.g. `['ATP', 'glycolysis', 'mitochondrion', ...]`).
- **To change:** Add/remove types in `NODE_TYPES` or switch GLiNER model in `extract_entities()`.

### Step 2: Relation extraction (LLM via OpenRouter)

- **File:** `src/extract_triples.py`
- **Role:** Take the **same text** + **entity list** and call the LLM to output **edges**: only the allowed relation types between those entities.
- **Model:** `mistralai/mistral-small-3.1-24b-instruct:free`, `temperature=0`, `top_p=1` (per assignment).
- **Allowed relations:** `ALLOWED_RELATIONS` in that file (e.g. `occurs_in`, `produces`, `converts_to`, …).
- **Output:** A list of triples `{ "head", "relation", "tail", "evidence" }` → written to `triples.csv` by `run_all.py`.

### Step 3: Output

- **File:** `src/run_all.py` → `save_triples_to_csv()`
- **Output:** `triples.csv` at project root: `head, relation, tail, evidence`.

---

## 4. How to run and what to submit

**Run (from repo root, venv activated):**

```bash
python src/run_all.py
```

**Check:**

- Console: “Entities found: […]”, “Model used: …”, “Number of triples: N”.
- `triples.csv`: header + one row per triple.

**Deliverable (assignment):**

- A Python script that turns the biology text into graph data (you have this: `run_all.py` + the two extraction modules).
- Input = short biology text (e.g. from `data/passage.txt` or a `data.txt` from the repo).
- Output = structured JSON or CSV of triplets; you’re using CSV.

---

## 5. Improving the extraction prompt

Assignment says: **prototype the extraction prompt in ChatGPT/Claude first**, then paste it into code.

1. **In ChatGPT/Claude:**  
   Paste your assignment instructions (ontology, relation list, output format).  
   Paste a short paragraph from `data/passage.txt`.  
   Ask for JSON in the exact shape you need (e.g. `{ "triples": [ { "head", "relation", "tail", "evidence" } ] }`).

2. **Iterate:**  
   If the model uses wrong relation types or adds markdown, tighten the prompt (“Use only these relation types”, “Output raw JSON only, no markdown”).

3. **Put the final prompt in code:**  
   The string you build in `src/extract_triples.py` inside `extract_triples(text, entities)` is that prompt. Right now it already includes:
   - “Only use the following relation types: …”
   - “Return ONLY valid JSON. Do not include explanations. Do not include markdown.”
   - The exact expected JSON format.

   You can add 1–2 example triples from your text if you want even more consistent output (e.g. “Glycolysis → occurs_in → cytosol”).

---

## 6. Quick reference: assignment checklist

- [x] Two-step pipeline: GLiNER (entities) → LLM (relations).
- [x] Strict ontology (node types in GLiNER, relation types in LLM).
- [x] OpenRouter + `mistralai/mistral-small-3.1-24b-instruct:free`, temp=0, top_p=1.
- [x] Print “Model used:” and “Number of triples:”.
- [x] Output clean CSV (or JSON) of triplets with head, relation, tail, evidence.
- [x] Parsing MVP: PDF → text via `src/parse_pdf.py` (Mistral OCR 3).
- [x] Output schema: `schema/triples_schema.json` and `schema/pipeline_output_schema.json`; validation in `run_all.py` for quality verification.
- [ ] Chunking MVP: single chunk or simple `text.split('\n\n')` is fine for 1–2 pages.

Once you run `python src/run_all.py` again, you should see triples in `triples.csv` and be in good shape for the Week 2 deliverable and the GM update.

---

## 7. Running on 7.2 glycolysis (Week 3)

To test the pipeline on the **7.2 glycolysis** section:

1. Place your file as `data/7.2_glycolysis.pdf` or `data/7.2_glycolysis.txt`.
2. From project root: `python src/run_all.py data/7.2_glycolysis.pdf` (or `.txt`), or run `./scripts/run_glycolysis.sh`.

See `data/README.md` for details.
