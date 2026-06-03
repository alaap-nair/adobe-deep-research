"""
summarize_model_runs.py -- Build the model comparison table for
docs/assignment_9.md from per-model latency.csv + RAGAS scores.csv files.

Usage:
    python scripts/summarize_model_runs.py
    python scripts/summarize_model_runs.py --markdown   # markdown table on stdout
"""

import argparse
import csv
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = ["mistral_baseline", "qwen25_72b", "gemma2_27b", "phi4"]


def load_latency(label: str):
    p = os.path.join(ROOT, "outputs", "model_runs", label, "latency.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def load_ragas(label: str):
    p = os.path.join(ROOT, "outputs", "ragas", label, "scores.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def avg(values):
    nums = [float(v) for v in values if v not in (None, "", "nan", "NaN")]
    return statistics.fmean(nums) if nums else None


def fmt(v, dp=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{dp}f}"
    return str(v)


def summarize(label: str) -> dict:
    rows = load_latency(label) or []
    times = [float(r["seconds"]) for r in rows if r.get("seconds")]
    n_total = len(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    n_ref = sum(1 for r in rows if r.get("is_refusal") == "1")

    times_sorted = sorted(times)
    p50 = times_sorted[int(len(times_sorted) * 0.5)] if times_sorted else None
    p95 = (
        times_sorted[min(len(times_sorted) - 1, int(len(times_sorted) * 0.95))]
        if times_sorted
        else None
    )
    mean_lat = statistics.fmean(times) if times else None

    rg = load_ragas(label) or []
    metrics = {
        k: avg([r.get(k, "") for r in rg])
        for k in ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
    }

    return {
        "label": label,
        "n_total": n_total,
        "n_err": n_err,
        "n_refusals": n_ref,
        "p50": p50,
        "p95": p95,
        "mean": mean_lat,
        **metrics,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markdown", action="store_true", help="Emit a markdown table.")
    args = p.parse_args()

    summaries = [summarize(label) for label in LABELS]

    if args.markdown:
        print(
            "| Model | n | Err | Refusals | p50 (s) | p95 (s) | Mean (s) | Faithfulness | Answer Rel | Ctx Recall | Ctx Precision |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for s in summaries:
            print(
                f"| {s['label']} | {s['n_total']} | {s['n_err']} | {s['n_refusals']} "
                f"| {fmt(s['p50'], 2)} | {fmt(s['p95'], 2)} | {fmt(s['mean'], 2)} "
                f"| {fmt(s['faithfulness'])} | {fmt(s['answer_relevancy'])} "
                f"| {fmt(s['context_recall'])} | {fmt(s['context_precision'])} |"
            )
    else:
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
