# Mock RAGAS Report

This report is based on a synthetic placeholder dataset in `data/eval/mock_assignment_dataset.json`.
It is suitable for testing the tooling and report format only. It is not a valid submission for a real apples-to-apples comparison because the answers and retrieved contexts were not captured from the actual system pipeline.

## Summary Table

| Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---:|---:|---:|---:|
| What are the 2 primary cell types? | 0.96 | 0.95 | 0.94 | 0.90 |
| Which organisms are classified as eukaryotes? | 0.97 | 0.96 | 0.95 | 0.94 |
| Which cell does not contain a nucleus? | 0.98 | 0.97 | 0.93 | 0.95 |
| Which cells are characterized by the absence of membrane bound organelles? | 0.93 | 0.88 | 0.90 | 0.84 |
| What kind of bond links monomers in a polysaccharide? | 0.95 | 0.94 | 0.89 | 0.91 |
| How do enzymes achieve syn-addition in the citric acid cycle despite all the thermodynamic preference for anti-addition? | 0.86 | 0.83 | 0.78 | 0.81 |
| What are some important properties of beta sheets? | 0.94 | 0.90 | 0.88 | 0.89 |
| How are mitochondria, citric acid cycle and ATP production all connected? | 0.97 | 0.96 | 0.95 | 0.93 |
| Which organelle is involved in energy production but is not in the location of glycolysis? | 0.96 | 0.95 | 0.94 | 0.92 |
| If a cell cannot use oxygen, which ATP producing process can still continue? | 0.97 | 0.96 | 0.93 | 0.94 |

## Interpretation

The strongest results appear on direct factual questions where the synthetic retrieved contexts map cleanly onto the expected answer, which is why faithfulness and answer relevancy remain high across most standard questions. The weakest area is the enzyme stereochemistry question, where lower context recall and relevancy suggest the retrieval set is too shallow for mechanistic biochemistry prompts and would likely need more specialized chunks or better query expansion.

The membrane-bound organelles question also shows slightly weaker precision, which is consistent with answers that are technically related but underspecified. In a real evaluation, this pattern would suggest that the system can often retrieve broadly relevant material but still struggles to surface the most exact explanatory chunk for nuanced wording.
