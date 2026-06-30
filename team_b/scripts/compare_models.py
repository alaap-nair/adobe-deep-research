"""
compare_models.py -- Run all ground-truth questions through the QA pipeline
using a specified LLM, write traces to outputs/model_runs/{label}/.

Retrieval is identical across models (Qdrant + Neo4j state is fixed); only the
synthesis model varies, so per-model trace deltas isolate generator behavior.

Usage:
    python scripts/compare_models.py qwen25_72b
    python scripts/compare_models.py gemma2_27b
    python scripts/compare_models.py phi4
    python scripts/compare_models.py custom --model some/openrouter-model-id

The label becomes the per-run output directory under outputs/model_runs/.
"""

import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

KNOWN_MODELS = {
    "mistral_baseline": "mistralai/mistral-small-3.2-24b-instruct",
    "qwen25_72b": "qwen/qwen-2.5-72b-instruct",
    "gemma2_27b": "google/gemma-2-27b-it",
    "phi4": "microsoft/phi-4",
}

REFUSAL_PREFIX = "i don't know"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label", help="Short label for the run (becomes the output dir name).")
    p.add_argument("--model", help="OpenRouter model id. Overrides KNOWN_MODELS lookup.")
    p.add_argument("--ground-truth", default=os.path.join(ROOT, "data", "ground_truth.json"))
    p.add_argument("--out-root", default=os.path.join(ROOT, "outputs", "model_runs"))
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run the first N questions (useful for smoke tests).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    model_id = args.model or os.getenv("QA_MODEL_OVERRIDE") or KNOWN_MODELS.get(args.label)
    if not model_id:
        print(
            f"ERROR: no model id for label '{args.label}'. "
            f"Pass --model or use one of: {sorted(KNOWN_MODELS)}",
            file=sys.stderr,
        )
        return 1

    out_dir = os.path.join(args.out_root, args.label)
    os.makedirs(out_dir, exist_ok=True)

    # Force the QA model BEFORE importing config / ask / run_ragas so the
    # module-level QA_MODEL constant in src/config.py picks it up.
    os.environ["QA_MODEL"] = model_id

    import config as app_config  # noqa: E402
    import ask  # noqa: E402
    import run_ragas  # noqa: E402
    from query_engine import QueryEngine  # noqa: E402

    # Belt-and-suspenders: if any of these were imported earlier in this
    # process via a shared interpreter, force the runtime values.
    app_config.QA_MODEL = model_id
    ask.OUTPUT_DIR = out_dir
    run_ragas.ANSWERS_DIR = out_dir

    with open(args.ground_truth, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if args.limit > 0:
        rows = rows[: args.limit]

    latency_path = os.path.join(out_dir, "latency.csv")

    print(f"== Model: {model_id}", file=sys.stderr)
    print(f"== Label: {args.label}", file=sys.stderr)
    print(f"== Output dir: {out_dir}", file=sys.stderr)
    print(f"== Questions: {len(rows)}", file=sys.stderr)

    engine = QueryEngine()
    try:
        ask.ensure_query_resources(engine)
        with open(latency_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["id", "question", "seconds", "answer_len", "n_citations", "is_refusal", "error"]
            )
            for row in rows:
                qid = row["id"]
                q = row["question"]
                t0 = time.perf_counter()
                err = ""
                answer_len = 0
                n_citations = 0
                is_refusal = 0
                try:
                    trace = run_ragas.regenerate_trace(q, engine)
                    result = trace.get("result", {}) or {}
                    answer = (result.get("answer") or "").strip()
                    citations = result.get("citations") or []
                    answer_len = len(answer)
                    n_citations = len(citations)
                    is_refusal = int(answer.lower().startswith(REFUSAL_PREFIX))
                except Exception as exc:
                    err = str(exc).replace("\n", " ")[:200]
                elapsed = time.perf_counter() - t0
                writer.writerow(
                    [qid, q, f"{elapsed:.2f}", answer_len, n_citations, is_refusal, err]
                )
                fh.flush()
                tag = "ERR" if err else ("REF" if is_refusal else "ok ")
                print(
                    f"  {qid:>4} {tag} {elapsed:6.2f}s  cites={n_citations:>2}  ans_len={answer_len:>4}"
                    + (f"  err={err[:80]}" if err else ""),
                    file=sys.stderr,
                )
    finally:
        engine.close()

    print(f"\nDone. Traces in {out_dir}", file=sys.stderr)
    print(f"Latency: {latency_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
