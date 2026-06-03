#!/bin/bash
# Run RAGAS evaluation on each per-model trace directory.
# Pin the judge model so cross-model scores are comparable. The QA model is
# set per-label so any missing-trace regeneration uses the matching model.
#
# Usage: bash scripts/run_all_ragas.sh

set -e

cd "$(dirname "$0")/.."

LOG=outputs/model_runs/all_ragas.log
mkdir -p outputs/ragas

model_for() {
    case "$1" in
        mistral_baseline)  echo "mistralai/mistral-small-3.2-24b-instruct" ;;
        qwen25_72b)        echo "qwen/qwen-2.5-72b-instruct" ;;
        gemma2_27b)        echo "google/gemma-2-27b-it" ;;
        phi4)              echo "microsoft/phi-4" ;;
        *)                 echo "" ;;
    esac
}

export RAGAS_JUDGE_MODEL="mistralai/mistral-small-3.2-24b-instruct"

echo "==== START $(date) ====" | tee -a "$LOG"

for label in mistral_baseline qwen25_72b gemma2_27b phi4; do
    OUT=outputs/ragas/${label}/
    TRACES=outputs/model_runs/${label}/
    if [ ! -f "${TRACES}/latency.csv" ]; then
        echo "SKIP ${label}: no traces at ${TRACES}" | tee -a "$LOG"
        continue
    fi
    echo "" | tee -a "$LOG"
    echo "==== RAGAS: ${label}  $(date) ====" | tee -a "$LOG"
    QA_MODEL="$(model_for "$label")" .venv/bin/python src/run_ragas.py \
        data/ground_truth.json \
        --out "$OUT" --traces-dir "$TRACES" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "==== ALL RAGAS DONE $(date) ====" | tee -a "$LOG"
