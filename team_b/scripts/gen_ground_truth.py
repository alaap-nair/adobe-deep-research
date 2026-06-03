#!/usr/bin/env python
"""gen_ground_truth.py -- regenerate reference answers with a large model (A10 Part 1).

For each question in ``data/ground_truth.json`` we retrieve the relevant corpus
passages via the dual-store ``QueryEngine`` and ask a larger, more accurate
OpenRouter model (``GT_MODEL``, default ``anthropic/claude-3.5-sonnet``) to write
an authoritative, corpus-grounded reference answer. These become the new
``ground_truth`` field that RAGAS scores against.

The original hand-written answers are preserved two ways:
- the whole file is backed up to ``data/ground_truth.curated.json`` (once), and
- each row keeps its prior answer under ``ground_truth_human``.

Usage:
    .venv/bin/python scripts/gen_ground_truth.py                 # all questions
    .venv/bin/python scripts/gen_ground_truth.py --limit 3       # smoke test
    GT_MODEL=openai/gpt-4o .venv/bin/python scripts/gen_ground_truth.py
"""

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from query_engine import QueryEngine  # noqa: E402
from qa_client import call_openrouter, parse_llm_json  # noqa: E402

DEFAULT_GT_MODEL = os.getenv("GT_MODEL", "anthropic/claude-opus-4.8")

GT_SYSTEM_PROMPT = """You are a senior biology expert writing GOLD-STANDARD reference \
answers for an evaluation dataset. These answers are the ground truth other systems \
are graded against, so they must be accurate, complete, and self-contained.

Return ONLY valid JSON matching this schema:
{
  "answer": "...",
  "grounded": true
}

Rules:
- Base the answer primarily on the supplied textbook passages.
- Be precise and complete but concise (1-4 sentences). State the key mechanism or
  definition the question asks for.
- If the passages fully support an answer, set "grounded": true.
- If the passages do NOT contain the needed information, set "grounded": false and
  give the best correct, widely-accepted biology answer you can from your own
  knowledge, while staying factual.
"""


def build_gt_prompt(question: str, chunks: list[dict]) -> str:
    """Format retrieved passages + the question into the reference-answer prompt."""
    if chunks:
        passages = "\n\n".join(
            f"[{i + 1}] ({c.get('source_name', 'unknown')}) {c.get('text', '')}"
            for i, c in enumerate(chunks)
        )
    else:
        passages = "(no passages retrieved)"
    return (
        f"Question: {question}\n\n"
        f"Textbook passages:\n{passages}\n\n"
        "Write the gold-standard reference answer as JSON."
    )


def generate_answer(engine: QueryEngine, question: str, model: str, top_k: int) -> dict:
    """Retrieve passages and ask the large model for a reference answer."""
    ctx = engine.retrieve_context(question, top_k_chunks=top_k)
    chunks = ctx.get("vector_hits", {}).get("chunks", [])
    prompt = build_gt_prompt(question, chunks)
    resp = call_openrouter(
        prompt,
        model,
        temperature=0.0,
        max_tokens=512,
        system_prompt=GT_SYSTEM_PROMPT,
    )
    if "choices" not in resp:
        raise RuntimeError(f"Unexpected OpenRouter response: {resp}")
    content = resp["choices"][0]["message"]["content"]
    payload = parse_llm_json(content)
    answer = str(payload.get("answer") or content or "").strip()
    return {
        "answer": answer,
        "grounded": payload.get("grounded"),
        "n_chunks": len(chunks),
        "sources": sorted({c.get("source_name") for c in chunks if c.get("source_name")}),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ground-truth",
        default=os.path.join(ROOT, "data", "ground_truth.json"),
        help="Path to the ground-truth dataset to regenerate in place.",
    )
    p.add_argument("--model", default=DEFAULT_GT_MODEL, help="OpenRouter model id.")
    p.add_argument("--top-k", type=int, default=10, help="Passages to ground each answer on.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N questions (0 = all).")
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Print generated answers but do not modify the dataset (dry run).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.ground_truth, encoding="utf-8") as handle:
        rows = json.load(handle)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Generating reference answers with {args.model} for {len(rows)} questions...", file=sys.stderr)

    engine = QueryEngine()
    updated: list[dict] = []
    try:
        for i, row in enumerate(rows, 1):
            for attempt in range(3):
                try:
                    result = generate_answer(engine, row["question"], args.model, args.top_k)
                    break
                except Exception as exc:  # network / rate-limit / transient API errors
                    if attempt == 2:
                        raise
                    print(f"  [{row['id']}] retry {attempt + 1} after error: {exc}", file=sys.stderr)
                    time.sleep(5 * (attempt + 1))

            new_row = dict(row)
            if "ground_truth_human" not in new_row:
                new_row["ground_truth_human"] = row.get("ground_truth", "")
            new_row["ground_truth"] = result["answer"]
            new_row["ground_truth_model"] = args.model
            new_row["ground_truth_grounded"] = result["grounded"]
            updated.append(new_row)

            flag = "grounded" if result["grounded"] else "OUT-OF-CORPUS"
            print(
                f"[{i}/{len(rows)}] {row['id']} ({flag}, {result['n_chunks']} chunks): "
                f"{result['answer'][:90]}...",
                file=sys.stderr,
            )
    finally:
        engine.close()

    if args.no_write:
        print("\n--no-write: dataset unchanged.", file=sys.stderr)
        return 0

    # Back up the original curated dataset once.
    backup = os.path.join(os.path.dirname(args.ground_truth), "ground_truth.curated.json")
    if not os.path.exists(backup):
        shutil.copy2(args.ground_truth, backup)
        print(f"Backed up original to {backup}", file=sys.stderr)

    # When --limit was used, only the processed rows were regenerated; merge them
    # back into the full file so we never drop the untouched tail.
    if args.limit:
        with open(args.ground_truth, encoding="utf-8") as handle:
            full = json.load(handle)
        by_id = {r["id"]: r for r in updated}
        full = [by_id.get(r["id"], r) for r in full]
        updated = full

    with open(args.ground_truth, "w", encoding="utf-8") as handle:
        json.dump(updated, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(updated)} rows to {args.ground_truth}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
