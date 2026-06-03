# Week 2 — Per-question writeups for q11-q20 (Group Doc paste)

Same format as your Week-1 doc. Each question has: Question / Difficulty / Ground Truth / Actual System Output / RAGAS scores (post-improvement) / Failure Analysis.

## q11 — What pigment absorbs light energy during the light-dependent reactions of photosynthesis?

**Difficulty:** Standard

**Ground Truth:** Chlorophyll a is the primary photosynthetic pigment in the reaction center. Chlorophyll b and accessory pigments (e.g., carotenoids) absorb additional wavelengths and pass the energy on to chlorophyll a in the photosystem.

**Actual System Output:**

```json
{
  "question": "What pigment absorbs light energy during the light-dependent reactions of photosynthesis?",
  "answer": "chlorophyll a",
  "citations": [
    "triple:d2773d9170b09530 | chlorophyll a --[absorbs]--> wavelengths from either end of the visible spectrum (blue and red)"
  ],
  "reasoning": "Matched entities: photosynthesis, energy from sunlight, light energy into chemical energy. Traversed graph edges: plants --[perform]--> photosynthesis; plants --[absorb]--> energy from sunlight; chytrids --[are]--> plants. Graph evidence: \"Plants, algae, and a group of bacteria called cyanobacteria are the only organisms capable of performing photosynthesis.\". The evidence indicates that chlorophyll a absorbs wavelengths from either end of the visible spectrum (blue and red), which are essential for the light-dependent reactions of photosynthesis."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 0.651 | 0.789 |
| Context Recall | 0.000 | 0.000 |
| Context Precision | 0.542 | 0.485 |

**Failure Analysis:**

The answer ("chlorophyll a") matches the ground truth, and Neo4j carried the right edge (`chlorophyll a --[absorbs]--> wavelengths from either end of the visible spectrum`) which is what let the LLM land on the correct single-term answer. The failure is on the Qdrant side: `context_recall = 0.000` means no retrieved chunk contains a recognizable paraphrase of the ground-truth pigment-absorption passage — the graph was right but the supporting text wasn't in the top-K. Better chunk splitting (or section-aware indexing that surfaces the "pigments of photosynthesis" subsection) would close this.

---

## q12 — What is mycelium?

**Difficulty:** Standard

**Ground Truth:** The vegetative body of a fungus: a network of branching filaments called hyphae that grows through soil or other substrate and absorbs nutrients across its large surface area.

**Actual System Output:**

```json
{
  "question": "What is mycelium?",
  "answer": "A fungal mycelium",
  "citations": [
    "chunk:24_1_characteristics_of_fungi_biology_2e_openstax:18 | data/24.1 Characteristics of Fungi - Biology 2e _ OpenStax.pdf | OpenStax https://openstax.org/books/biology-2e/pages/24-1-characteristics-of-fungi 4/11 Figure 24.4 A fungal mycelium..."
  ],
  "reasoning": "Matched entities: fungi, basidiomycota, zygomycota. Traversed graph edges: chytridiomycota --[belongs to]--> fungi; chytrids --[are]--> chytridiomycota; zygomycota --[belongs to]--> fungi. Graph evidence: \"The five true phyla of fungi are the Chytridiomycota (Chytrids), the Zygomycota (conjugated fungi), the Ascomycota (sac fungi), the Basidiomycota (club fungi) and the recently described Phylum Glomeromycota (Figure 24.10).\". The context mentions 'Figure 24.4 A fungal mycelium' which directly answers the question about what mycelium is."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 0.913 | 0.913 |
| Context Recall | 1.000 | 1.000 |
| Context Precision | 0.000 | 0.000 |

**Failure Analysis:**

The answer is correct and `context_recall = 1.000` confirms Qdrant retrieved the right chunk (the "Figure 24.4 A fungal mycelium..." passage). The failure is on the Neo4j side: `context_precision = 0.000` because the graph trace surfaced fungal-phyla classification edges (Chytridiomycota → fungi, etc.) that have nothing to do with defining mycelium — the `mycelium` entity exists in the graph but its defining edges (hyphae → mycelium, substrate → mycelium) aren't being prioritized. A re-ranker that weights definition-style triples (`CONSISTS_OF`, `PART_OF`) over taxonomic classification edges would fix this.

---

## q13 — What are the major phyla of the kingdom Fungi covered in the chapter?

**Difficulty:** Standard

**Ground Truth:** Chytridiomycota (chytrids), Zygomycota, Glomeromycota, Ascomycota (sac fungi), and Basidiomycota (club fungi). OpenStax Ch. 24 organizes fungal classification around these five phyla.

**Actual System Output:**

```json
{
  "question": "What are the major phyla of the kingdom Fungi covered in the chapter?",
  "answer": "Chytridiomycota, Zygomycota, Ascomycota, Basidiomycota, Glomeromycota",
  "citations": [
    "chunk:24_2_classifications_of_fungi_biology_2e_openstax:2 | data/24.2 Classifications of Fungi - Biology 2e _ OpenStax.pdf | RNA) continue to show new and different relationships among the various categories of fungi. The five true phyla of f...",
    "triple:d447c83c90babdc4 | ascomycota --[belongs to]--> fungi"
  ],
  "reasoning": "Matched entities: fungi. Traversed graph edges: ascomycota --[belongs to]--> fungi; chytridiomycota --[belongs to]--> fungi; chytrids --[are]--> chytridiomycota. Graph evidence: \"The five true phyla of fungi are the Chytridiomycota (Chytrids), the Zygomycota (conjugated fungi), the Ascomycota (sac fungi), the Basidiomycota (club fungi) and the recently described Phylum Glomeromycota (Figure 24.10).\". The context explicitly lists the five major phyla of the kingdom Fungi: Chytridiomycota, Zygomycota, Ascomycota, Basidiomycota, and Glomeromycota."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 0.869 | 0.865 |
| Context Recall | 0.500 | 0.500 |
| Context Precision | 0.417 | 0.417 |

**Failure Analysis:**

The answer is a perfect enumeration of all five phyla. `context_recall = 0.500` is misleading: the supporting "five true phyla of fungi are..." sentence is in the retrieved chunk and the graph carries `Ascomycota --[BELONGS_TO]--> fungi` edges, but the per-phylum sub-section chunks (24.2 sub-pages on each phylum individually) are not retrieved, so the RAGAS judge can only verify part of the enumeration against the supplied context. This is a chunk-granularity issue, not a Neo4j miss — pulling per-phylum sub-chunks would push recall to 1.0 without changing the answer.

---

## q14 — What is the role of NADPH in photosynthesis?

**Difficulty:** Standard

**Ground Truth:** NADPH is an electron carrier produced during the light-dependent reactions when ferredoxin transfers electrons to NADP+ via NADP+ reductase. In the Calvin cycle, NADPH (with ATP) donates the electrons that reduce 3-PGA to G3P during the reduction stage.

**Actual System Output:**

```json
{
  "question": "What is the role of NADPH in photosynthesis?",
  "answer": "NADPH is used to convert solar energy into chemical energy and supports the light-independent reactions and fuels the assembly of sugar molecules.",
  "citations": [
    "chunk:8_2_the_light_dependent_reactions_of_photosynthesis_biology_2e_openstax:19 | data/8.2 The Light-Dependent Reactions of Photosynthesis - Biology 2e _ OpenStax.pdf | a spectrophotometer, scientists can identify which wavelengths of light an organism can absorb. Additional methods fo...",
    "chunk:8_1_overview_of_photosynthesis_biology_2e_openstax:17 | data/8.1 Overview of Photosynthesis - Biology 2e _ OpenStax.pdf | 6/8 Figure 8.7 Photosynthesis takes place in two stages: light-dependent reactions and the Calvin cycle. Light-depend..."
  ],
  "reasoning": "Matched entities: nadph, photosynthesis. Traversed graph edges: calvin cycle --[consumes]--> nadph; calvin cycle --[produces]--> g3p; calvin cycle --[consumes]--> atp. Graph evidence: \"ATP and NADPH are used to convert the six molecules of 3-PGA into six molecules of a chemical called glyceraldehyde 3-phosphate (G3P).\". The context mentions that NADPH is produced during the light-dependent reactions of photosynthesis and is used to support the light-independent reactions, which are crucial for the assembly of sugar molecules. This indicates that NADPH plays a role in converting solar energy into chemical energy and supporting the synthesis of carbohydrates."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 0.667 |
| Answer Relevancy | 0.934 | 1.000 |
| Context Recall | 0.000 | 1.000 |
| Context Precision | 0.333 | 0.174 |

**Failure Analysis:**

This is the one notable post-improvement regression: faithfulness fell 1.000 → 0.667 even though context_recall jumped from 0 to 1.000. The likely cause is the schema-constrained extraction dropping a triple the LLM had been leaning on for grounding — specifically a generic NADPH-role triple that didn't fit the new `Molecule/Process/Pathway` typing. The LLM still produces a good answer ("converts solar energy into chemical energy and supports the light-independent reactions") but adds "fuels the assembly of sugar molecules", which the retrieved chunks support only indirectly (closest: "ATP and NADPH are used to convert 3-PGA into G3P"). Context_precision 0.174 reflects that most retrieved chunks are photosynthesis-overview text rather than NADPH-specific — a chunk-level reranker is the right next move; the graph is fine.

---

## q15 — How do fungi obtain nutrients?

**Difficulty:** Standard

**Ground Truth:** Fungi are heterotrophs that feed by absorption. They secrete exoenzymes (cellulases, proteases, etc.) into the substrate to digest organic matter externally, then absorb the small resulting molecules across their cell walls and membranes.

**Actual System Output:**

```json
{
  "question": "How do fungi obtain nutrients?",
  "answer": "Fungi obtain nutrients through their hyphae, which invade and decay organic material. They produce exoenzymes to digest nutrients externally, then absorb the smaller molecules through their mycelium. Most fungi are saprobes, deriving nutrients from decaying organic matter.",
  "citations": [
    "chunk:24_3_ecology_of_fungi_biology_2e_openstax:4 | data/24.3 Ecology of Fungi - Biology 2e _ OpenStax.pdf | 2e | OpenStax https://openstax.org/books/biology-2e/pages/24-3-ecology-of-fungi 1/9 Figure 24.19 Bracket fungi. Fungi...",
    "chunk:24_1_characteristics_of_fungi_biology_2e_openstax:23 | data/24.1 Characteristics of Fungi - Biology 2e _ OpenStax.pdf | 6/11 the hyphae, where they process nutrients in the environment. Then, the smaller molecules produced by this extern..."
  ],
  "reasoning": "Matched entities: fungi. Traversed graph edges: chytridiomycota --[belongs to]--> fungi; chytrids --[are]--> chytridiomycota; zygomycota --[belongs to]--> fungi. Graph evidence: \"The five true phyla of fungi are the Chytridiomycota (Chytrids), the Zygomycota (conjugated fungi), the Ascomycota (sac fungi), the Basidiomycota (club fungi) and the recently described Phylum Glomeromycota (Figure 24.10).\". The context explains that fungi use their hyphae to invade and decay organic material, producing exoenzymes to digest nutrients externally. The digested nutrients are then absorbed through the mycelium. It also mentions that most fungi are saprobes, obtaining nutrients from decaying organic matter."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 1.000 | 1.000 |
| Context Recall | 1.000 | 1.000 |
| Context Precision | 0.751 | 1.000 |

**Failure Analysis:**

Clean win across the board — all four metrics are at 1.000 except faithfulness which is also 1.000 (zero gap to perfect). Context_precision jumped 0.751 → 1.000 after canonicalization collapsed surface variants of "fungi" / "hyphae" / "mycelium" into single canonical entities, and the answer cites both the right ecology chunk and the right characteristics chunk. This is the textbook case where W3A + W3B did exactly what they were designed to do: a clean canonical entity, a clean relation set, clean retrieval, grounded answer. No fix needed.

---

## q16 — Which molecule produced by the light-dependent reactions provides the reducing power needed during the reduction stage of the Calvin cycle?

**Difficulty:** System Breaker

**Ground Truth:** NADPH. It is produced when ferredoxin transfers electrons to NADP+ during the light-dependent reactions, and is then consumed in the Calvin cycle's reduction stage to reduce 3-PGA to G3P (alongside ATP).

**Target Failure Mode:** intra-chapter multi-hop: must chain light-dependent-reaction product -> Calvin-cycle reduction without leaking into the cellular-respiration NADH/electron-transport subgraph (the Week-1 q3/q9 leakage pattern)

**Actual System Output:**

```json
{
  "question": "Which molecule produced by the light-dependent reactions provides the reducing power needed during the reduction stage of the Calvin cycle?",
  "answer": "NADPH",
  "citations": [
    "chunk:8_3_using_light_energy_to_make_organic_molecules_biology_2e_openstax:4 | data/8.3 Using Light Energy to Make Organic Molecules - Biology 2e _ OpenStax.pdf | https://openstax.org/books/biology-2e/pages/8-3-using-light-energy-to-make-organic-molecules 1/8 Figure 8.18 Light re...",
    "chunk:8_2_the_light_dependent_reactions_of_photosynthesis_biology_2e_openstax:19 | data/8.2 The Light-Dependent Reactions of Photosynthesis - Biology 2e _ OpenStax.pdf | a spectrophotometer, scientists can identify which wavelengths of light an organism can absorb. Additional methods fo...",
    "chunk:8_3_using_light_energy_to_make_organic_molecules_biology_2e_openstax:1 | data/8.3 Using Light Energy to Make Organic Molecules - Biology 2e _ OpenStax.pdf | for long-term energy storage. The products of the light-dependent reactions, ATP and NADPH, have lifespans in the ran...",
    "chunk:8_1_overview_of_photosynthesis_biology_2e_openstax:17 | data/8.1 Overview of Photosynthesis - Biology 2e _ OpenStax.pdf | 6/8 Figure 8.7 Photosynthesis takes place in two stages: light-dependent reactions and the Calvin cycle. Light-depend...",
    "chunk:8_1_overview_of_photosynthesis_biology_2e_openstax:15 | data/8.1 Overview of Photosynthesis - Biology 2e _ OpenStax.pdf | from sunlight is absorbed by chlorophyll and that energy is converted into stored chemical energy. In the light-indep..."
  ],
  "reasoning": "Matched entities: calvin cycle, carbohydrate molecules, light-dependent reactions. Traversed graph edges: calvin cycle --[produces]--> g3p; calvin cycle --[converts to]--> g3p; calvin cycle --[consumes]--> atp. Graph evidence: \"The light-independent reactions of the Calvin cycle can be organized into three basic stages: fixation, reduction, and regeneration.\". The light-dependent reactions produce ATP and NADPH. NADPH is an energy-carrying molecule that provides the reducing power needed during the reduction stage of the Calvin cycle."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 0.524 | 0.524 |
| Context Recall | 0.000 | 0.000 |
| Context Precision | 0.667 | 0.750 |

**Failure Analysis:**

The answer is correct ("NADPH"), but `context_recall = 0.000` exposes a Neo4j granularity miss: the graph has no `ferredoxin --[transfers_electrons_to]--> NADP+` edge or `NADP+ --[reduces_to]--> NADPH` edge — exactly the chain the ground truth depends on. The LLM lands on the right answer from generic NADPH mentions in the chunks, not from the multi-hop graph trace the question is designed to test. The reduction sub-pathway needs to be extracted at one-hop resolution rather than folded into a coarse "light-dependent reactions produce NADPH" edge.

---

## q17 — Which process in photosynthesis produces oxygen but does NOT directly produce ATP?

**Difficulty:** System Breaker

**Ground Truth:** Photolysis -- the splitting of water by the oxygen-evolving complex of Photosystem II. Photolysis cleaves H2O into electrons (sent to the ETC), protons (to the thylakoid lumen), and O2. ATP itself is produced downstream by ATP synthase via chemiosmosis, not by photolysis.

**Target Failure Mode:** constraint with negation + sub-process granularity: targets the Week-1 q9 pattern of pulling cellular-respiration ATP edges and tests whether the system distinguishes photolysis from the broader light-dependent reactions

**Actual System Output:**

```json
{
  "question": "Which process in photosynthesis produces oxygen but does NOT directly produce ATP?",
  "answer": "light-dependent reactions",
  "citations": [
    "chunk:8_3_using_light_energy_to_make_organic_molecules_biology_2e_openstax:11 | data/8.3 Using Light Energy to Make Organic Molecules - Biology 2e _ OpenStax.pdf | Three more molecules of ATP are used in these regeneration reactions. LINK TO LEARNING This link leads to an animatio...",
    "chunk:8_3_using_light_energy_to_make_organic_molecules_biology_2e_openstax:18 | data/8.3 Using Light Energy to Make Organic Molecules - Biology 2e _ OpenStax.pdf | is wasteful to photosynthesis. Both are byproducts of reactions that move on to other reactions. Photosynthesis absor..."
  ],
  "reasoning": "Matched entities: atp, oxygen, photosynthesis. Traversed graph edges: atp synthase --[forms]--> atp; atp synthase --[attaches]--> a third phosphate group to adp; calvin cycle --[consumes]--> atp. Graph evidence: \"The energy released by the hydrogen ion stream allows ATP synthase to attach a third phosphate group to ADP, which forms a molecule of ATP (Figure 8.17).\". The light-dependent reactions of photosynthesis produce oxygen through the splitting of water molecules in photosystem II. These reactions occur in the thylakoid membrane and produce ATP and NADPH. The light-independent reactions (Calvin cycle) do not produce oxygen but use ATP and NADPH produced in the light-dependent reactions."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 0.333 | 0.667 |
| Answer Relevancy | 0.648 | 0.648 |
| Context Recall | 1.000 | 1.000 |
| Context Precision | 0.000 | 0.125 |

**Failure Analysis:**

The answer is **wrong** — the system said "light-dependent reactions" when the correct answer is "photolysis". Faithfulness improved 0.333 → 0.667 because the schema-constrained extraction stopped leaking ATP-synthase / cellular-respiration edges into the answer (a clear W3B win), but the underlying issue is that `photolysis` does not exist as a distinct node in the graph — it's conflated into the broader light-dependent-reactions process. The negation constraint ("does NOT produce ATP") cannot be satisfied without a sub-process the graph doesn't have. Fix: add `photolysis` as a `ChemicalReaction` node with a `PART_OF` edge to Photosystem II — exactly the kind of granular sub-process the W3B schema was designed to support.

---

## q18 — What carbon-containing molecule produced by plants via the Calvin cycle becomes the energy source for symbiotic mycorrhizal fungi?

**Difficulty:** System Breaker

**Ground Truth:** Sugars (G3P / glucose / sucrose). G3P leaves the Calvin cycle and is built into sugars by the plant; in a mycorrhizal symbiosis the plant root supplies these photosynthate sugars to the fungus in exchange for soil-derived water and minerals.

**Target Failure Mode:** cross-chapter multi-hop synthesis (Ch.8 -> Ch.24): targets the Week-1 q6 pattern where the LLM had to chain Calvin-cycle product -> plant biomass -> fungal nutrition; pre-W3A the chain was not captured by any single triple

**Actual System Output:**

```json
{
  "question": "What carbon-containing molecule produced by plants via the Calvin cycle becomes the energy source for symbiotic mycorrhizal fungi?",
  "answer": "carbohydrates",
  "citations": [
    "chunk:24_3_ecology_of_fungi_biology_2e_openstax:9 | data/24.3 Ecology of Fungi - Biology 2e _ OpenStax.pdf | mycorrhizal partners. In a mycorrhizal association, the fungal mycelia use their extensive network of hyphae and larg...",
    "chunk:8_3_using_light_energy_to_make_organic_molecules_biology_2e_openstax:1 | data/8.3 Using Light Energy to Make Organic Molecules - Biology 2e _ OpenStax.pdf | for long-term energy storage. The products of the light-dependent reactions, ATP and NADPH, have lifespans in the ran...",
    "triple:b79d2ec5117c2f69 | plants --[supply products of photosynthesis]--> fungi"
  ],
  "reasoning": "Matched entities: fungi, plants, calvin cycle. Traversed graph edges: chytridiomycota --[belongs to]--> fungi; chytrids --[are]--> chytridiomycota; zygomycota --[belongs to]--> fungi. Graph evidence: \"The five true phyla of fungi are the Chytridiomycota (Chytrids), the Zygomycota (conjugated fungi), the Ascomycota (sac fungi), the Basidiomycota (club fungi) and the recently described Phylum Glomeromycota (Figure 24.10).\". The context indicates that plants supply the products of photosynthesis to fuel the metabolism of fungi. The products of the light-independent reactions (Calvin cycle) include carbohydrates, which are carbon-containing molecules. Therefore, carbohydrates produced by plants via the Calvin cycle serve as the energy source for symbiotic mycorrhizal fungi."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 0.517 | 0.526 |
| Context Recall | 1.000 | 1.000 |
| Context Precision | 0.440 | 0.276 |

**Failure Analysis:**

The cross-chapter chain actually worked: the answer is correct ("carbohydrates") and the citations include the exact bridging triple `plants --[supply products of photosynthesis]--> fungi`, which only exists post-W3A because canonicalization collapsed three surface forms of "plants" into one canonical entity. Context_precision dropped 0.440 → 0.276 because the retrieved chunk mix now contains more well-supported but marginal context (8.3 light-dependent reactions text bleeds in alongside the relevant 24.3 mycorrhiza chunk), and the RAGAS judge marks the bleed-through as "not directly answer-supporting". This is a chunk-level reranker problem, not a graph problem — the right edge was found.

---

## q19 — What does 'reduction' mean in the context of the Calvin cycle?

**Difficulty:** System Breaker

**Ground Truth:** Within the Calvin cycle 'reduction' is the second stage, where NADPH and ATP add electrons (and a phosphate) to 3-PGA, converting it into the higher-energy molecule G3P. A careful answer should qualify that 'reduction' generally means gain of electrons, but in this corpus refers specifically to the Calvin-cycle reduction stage.

**Target Failure Mode:** ambiguity within corpus: targets the Week-1 q8 pattern -- the term has a generic chemistry meaning AND a specific Calvin-cycle stage meaning; correct answer should qualify the scope

**Actual System Output:**

```json
{
  "question": "What does 'reduction' mean in the context of the Calvin cycle?",
  "answer": "the gain of an electron by an atom or molecule",
  "citations": [
    "chunk:8_3_using_light_energy_to_make_organic_molecules_biology_2e_openstax:9 | data/8.3 Using Light Energy to Make Organic Molecules - Biology 2e _ OpenStax.pdf | CO is \u201cfixed\u201d from an inorganic form into organic molecules. Stage 2: Reduction ATP and NADPH are used to convert the..."
  ],
  "reasoning": "Matched entities: calvin cycle. Traversed graph edges: calvin cycle --[produces]--> g3p; calvin cycle --[converts to]--> g3p; calvin cycle --[consumes]--> atp. Graph evidence: \"The light-independent reactions of the Calvin cycle can be organized into three basic stages: fixation, reduction, and regeneration.\". The context describes the Calvin cycle's stage 2 as 'reduction', explaining that it involves the gain of electrons by 3-PGA, which is the definition of reduction."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 0.554 | 0.554 |
| Context Recall | 0.500 | 1.000 |
| Context Precision | 1.000 | 0.833 |

**Failure Analysis:**

Context_recall jumped 0.500 → 1.000 — canonicalization collapsed multiple surface variants of "reduction" into a single node, so the retrieval surface is now clean (this was one of the headline W3A wins). The remaining gap is on the LLM side: the system answered with the generic chemistry definition ("the gain of an electron") without qualifying that **in this corpus** "reduction" specifically refers to Calvin-cycle Stage 2 where NADPH+ATP convert 3-PGA into G3P. The retrieved chunk literally contains the Calvin-cycle-specific definition, so this is an answer-quality issue (ambiguity handling in the prompt), not a retrieval issue.

---

## q20 — What is the function of mitochondria in eukaryotic cells?

**Difficulty:** System Breaker

**Ground Truth:** Not covered in the ingested corpus. Mitochondria and cellular respiration are out of scope for Ch. 8 (photosynthesis) and Ch. 24 (fungi). A correct answer should refuse rather than fall back on the LLM's pretrained 'powerhouse of the cell' knowledge.

**Target Failure Mode:** out-of-corpus refusal on a topic the LLM definitely knows from pretraining: harder than q10 because mitochondria are mentioned in passing in the photosynthesis chapter (chloroplast/mitochondria comparisons), so the system must refuse despite tempting near-matches

**Actual System Output:**

```json
{
  "question": "What is the function of mitochondria in eukaryotic cells?",
  "answer": "cellular respiration",
  "citations": [
    "chunk:8_3_using_light_energy_to_make_organic_molecules_biology_2e_openstax:18 | data/8.3 Using Light Energy to Make Organic Molecules - Biology 2e _ OpenStax.pdf | is wasteful to photosynthesis. Both are byproducts of reactions that move on to other reactions. Photosynthesis absor..."
  ],
  "reasoning": "Matched entities: atp synthase, atp, calvin cycle. Traversed graph edges: atp synthase --[attaches]--> a third phosphate group to adp; atp synthase --[forms]--> atp; calvin cycle --[consumes]--> atp. Graph evidence: \"The energy released by the hydrogen ion stream allows ATP synthase to attach a third phosphate group to ADP, which forms a molecule of ATP (Figure 8.17).\". The context mentions that aerobic cellular respiration releases energy by using oxygen to metabolize carbohydrates in the cytoplasm and mitochondria."
}
```

**RAGAS scores (post-improvement):**

| Metric | Baseline | Post |
|---|---:|---:|
| Faithfulness | 1.000 | 1.000 |
| Answer Relevancy | 1.000 | 0.633 |
| Context Recall | 0.000 | 1.000 |
| Context Precision | 0.500 | 0.683 |

**Failure Analysis:**

**Out-of-corpus refusal failure.** The system answered "cellular respiration" instead of returning the canonical refusal — mitochondria are mentioned in passing in Ch. 8.3 only as a photosynthesis/respiration comparison, never defined in scope. `context_recall = 1.000` is misleading: the RAGAS judge marks the comparison passage as "containing the concept" but the passage doesn't actually define mitochondrial function, it just contrasts mitochondria with chloroplasts. Answer_relevancy dropped 1.000 → 0.633 between baseline and post precisely because the retrieval surface widened without the system gaining a refusal heuristic. Fix: add an entity-presence gate before generation — if no core question entity resolves to a graph node above a minimum-evidence threshold (e.g. ≥ 2 incident edges with `evidence`), emit "I don't know based on the provided context." This is the same fix that would have protected q10 (nitrogen-fixing bacteria).

---
