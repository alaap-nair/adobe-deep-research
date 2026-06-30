#!/bin/bash
# Reranker before/after experiment. Re-runs the QA pipeline with the
# cross-encoder reranker enabled and re-scores against the same ground truth,
# so we can diff context_precision (and other RAGAS metrics) cleanly.
#
# Pre-req: outputs/model_runs/mistral_baseline/ already populated by
#   .venv/bin/python scripts/compare_models.py mistral_baseline
#
# Usage: bash scripts/run_rerank_experiment.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LABEL=mistral_rerank
TRACES=outputs/model_runs/${LABEL}/
OUT=outputs/ragas/${LABEL}/
LOG=outputs/model_runs/rerank_experiment.log

mkdir -p "$TRACES" "$OUT"

echo "==== RERANK EXPERIMENT START $(date) ====" | tee -a "$LOG"

# Step 1: regenerate traces with reranker enabled
export USE_RERANKER=1
echo "" | tee -a "$LOG"
echo "==== Regenerate with USE_RERANKER=1  $(date) ====" | tee -a "$LOG"
.venv/bin/python scripts/compare_models.py "$LABEL" \
    --model "mistralai/mistral-small-3.2-24b-instruct" 2>&1 | tee -a "$LOG"

# Step 2: RAGAS on the reranked traces
unset USE_RERANKER  # judge calls don't need reranker
export RAGAS_JUDGE_MODEL="mistralai/mistral-small-3.2-24b-instruct"
echo "" | tee -a "$LOG"
echo "==== RAGAS on ${LABEL}  $(date) ====" | tee -a "$LOG"
.venv/bin/python src/run_ragas.py data/ground_truth.json \
    --out "$OUT" --traces-dir "$TRACES" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== RERANK EXPERIMENT DONE $(date) ====" | tee -a "$LOG"
