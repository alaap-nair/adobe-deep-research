"""
batch_eval.py -- Run all ground truth questions through the pipeline and save results.

Queries each question, generates an answer, saves trace files, and produces
a rubric-format summary for the assignment write-up.

Usage:
    python src/batch_eval.py                    # run all questions
    python src/batch_eval.py --only-new         # run only new questions (set: "new")
    python src/batch_eval.py --dry-run          # show questions without querying
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(ROOT, "data", "ground_truth.json")
ANSWERS_DIR = os.path.join(ROOT, "outputs", "answers")


def slugify(question: str) -> str:
    slug = question.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")[:80].rstrip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-new", action="store_true", help="Only run new questions")
    parser.add_argument("--dry-run", action="store_true", help="Show questions without querying")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        questions = json.load(f)

    if args.only_new:
        questions = [q for q in questions if q.get("set") == "new"]

    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(questions)} questions\n")

    if args.dry_run:
        for i, q in enumerate(questions, 1):
            diff = q.get("difficulty", "?")
            targets = q.get("targets_failure", "")
            print(f"  {i:2d}. [{diff}] {q['question']}")
            if targets:
                print(f"      Targets: {targets}")
        return

    import config as app_config
    from query_engine import QueryEngine
    from qa_client import answer_question

    model_name = getattr(app_config, "QA_MODEL", "") or getattr(app_config, "MODEL_NAME", "")
    engine = QueryEngine()
    os.makedirs(ANSWERS_DIR, exist_ok=True)

    results = []
    try:
        for i, q in enumerate(questions, 1):
            question = q["question"]
            slug = slugify(question)
            print(f"[{i:2d}/{len(questions)}] {question}")

            try:
                context = engine.retrieve_context(question)
                result = answer_question(question, context, model_name)

                answer_path = os.path.join(ANSWERS_DIR, f"{slug}.json")
                trace_path = os.path.join(ANSWERS_DIR, f"{slug}_trace.json")

                with open(answer_path, "w") as f:
                    json.dump(result, f, indent=2)
                with open(trace_path, "w") as f:
                    json.dump({
                        "question": question,
                        "retrieval_context": context,
                        "result": result,
                    }, f, indent=2)

                answer = result.get("answer", "ERROR")
                print(f"         -> {answer[:80]}")

                results.append({
                    "question": question,
                    "difficulty": q.get("difficulty", ""),
                    "ground_truth": q["ground_truth"],
                    "system_output": answer,
                    "status": "ok",
                })

            except Exception as e:
                print(f"         -> ERROR: {e}")
                results.append({
                    "question": question,
                    "difficulty": q.get("difficulty", ""),
                    "ground_truth": q["ground_truth"],
                    "system_output": f"ERROR: {e}",
                    "status": "error",
                })

            # Brief pause to avoid rate limits
            if i < len(questions):
                time.sleep(2)

    finally:
        engine.close()

    # Save batch results summary
    summary_path = os.path.join(ROOT, "outputs", "batch_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print rubric-format summary
    print(f"\n{'='*80}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*80}")
    ok = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    print(f"Completed: {ok}/{len(results)} | Errors: {err}")
    print(f"\nResults saved to {summary_path}")
    print(f"Individual traces in {ANSWERS_DIR}/")


if __name__ == "__main__":
    main()
