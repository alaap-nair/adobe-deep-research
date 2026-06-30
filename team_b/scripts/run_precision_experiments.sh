#!/usr/bin/env bash
# A10 Part 2 -- context-precision experiment matrix.
#
# Each configuration regenerates all 40 answers under its retrieval setting and
# scores them with RAGAS. The synthesis model and the RAGAS judge are both pinned
# to mistral-small so that ONLY the retrieval strategy varies between runs.
#
# Per-label --out and --traces-dir keep every config isolated (no cross-config
# trace contamination, no scores.csv collision).
#
#   C0  baseline      default retrieval (USE_RERANKER=0, no RETRIEVAL_MODE)
#   C1  bge reranker  cross-encoder rerank with BAAI/bge-reranker-v2-m3
#   C2  mmr           MMR diversity over a 6x candidate pool
#   C3  graph_boost   graph-connectivity rescoring over a 6x pool (dual-store)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
GT=data/ground_truth.json
SYNTH=mistralai/mistral-small-3.2-24b-instruct
export QA_MODEL="$SYNTH"
export RAGAS_JUDGE_MODEL="$SYNTH"

run () {  # run <label> -- remaining args are extra env assignments handled by caller
  local label="$1"
  echo "================ $label ================"
  "$PY" src/run_ragas.py "$GT" \
    --out "outputs/ragas/precision_${label}/" \
    --traces-dir "outputs/answers/precision_${label}/" \
    --force-regenerate
}

# C0 -- baseline (clear any inherited toggles)
USE_RERANKER=0 RETRIEVAL_MODE="" run c0_baseline

# C1 -- domain cross-encoder reranker
USE_RERANKER=1 RERANKER_MODEL="BAAI/bge-reranker-v2-m3" RETRIEVAL_MODE="" run c1_bge_rerank

# C2 -- MMR diversity
USE_RERANKER=0 RETRIEVAL_MODE="mmr" run c2_mmr

# C3 -- graph-aware rescoring (dual-store advantage)
USE_RERANKER=0 RETRIEVAL_MODE="graph_boost" run c3_graph_boost

echo
echo "All runs complete. Aggregate context_precision per config:"
for label in c0_baseline c1_bge_rerank c2_mmr c3_graph_boost; do
  f="outputs/ragas/precision_${label}/summary.md"
  if [ -f "$f" ]; then
    printf "  %-16s " "$label"
    grep -E '\*\*context_precision\*\*' "$f" || echo "(no aggregate)"
  fi
done
