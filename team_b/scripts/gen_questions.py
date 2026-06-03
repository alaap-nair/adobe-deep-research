"""
gen_questions.py -- Generate 20 evaluation questions across 4 categories using
the project's QA model.

Categories (5 questions each):
  - single-hop:        answerable from one chunk
  - multi-hop:         requires combining 2+ chunks from the same source
  - cross-chapter:     requires combining chunks from 2+ source files
  - system-breaker:    out-of-corpus / negation / ambiguity

Samples chunk text from Qdrant so questions stay grounded in the ingested
corpus, then asks the configured QA_MODEL to author the question set.

Usage:
    python scripts/gen_questions.py
    python scripts/gen_questions.py --out questions.json --per-source 2
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config as app_config  # noqa: E402
from build_vectorstore import get_client  # noqa: E402
from qa_client import call_openrouter, parse_llm_json  # noqa: E402

CATEGORIES = ["single-hop", "multi-hop", "cross-chapter", "system-breaker"]

GEN_SYSTEM_PROMPT = """You are a biology TA writing an evaluation set for a
hybrid retrieval-augmented QA system.

You will be given a sample of chunks from the system's ingested corpus. Each
chunk includes its source filename so you can identify which chapter it came
from.

Produce EXACTLY 20 questions in JSON, distributed as follows:
- 5 single-hop:    answerable from one chunk in the supplied sample
- 5 multi-hop:     require combining facts from 2+ chunks of the SAME source
- 5 cross-chapter: require combining facts across DIFFERENT source files
- 5 system-breaker: deliberately hard. Use one of these patterns:
    * out-of-corpus topic the LLM knows from pretraining but the supplied
      chunks do not cover (the system should refuse)
    * negation / constraint (e.g., "produces X but NOT Y")
    * ambiguity within the corpus (a term with two valid scoped meanings)
    * sub-process granularity (a specific sub-step the corpus mentions)

Return ONLY valid JSON in this exact shape (no prose, no markdown):
{
  "questions": [
    {
      "id": "g1",
      "question": "...",
      "category": "single-hop",
      "expected_difficulty": "Standard"
    },
    ...
  ]
}

Rules:
- Use ids g1..g20 in order.
- 5 questions per category, in the order: single-hop, multi-hop,
  cross-chapter, system-breaker.
- expected_difficulty is "Standard" for single-hop, "Standard" or "Multi-hop"
  for multi-hop, "Multi-hop" for cross-chapter, "System Breaker" for
  system-breaker.
- Questions must be specific. Avoid trivia; favor mechanism, comparison,
  cause/effect.
- Do not invent facts not present in the chunks for non-system-breaker
  questions."""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(ROOT, "questions.json"))
    p.add_argument(
        "--per-source",
        type=int,
        default=2,
        help="Sample N chunks per source_name (default 2). Higher = bigger prompt.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        help="Override QA_MODEL for generation (e.g., a stronger model than the default).",
    )
    return p.parse_args()


def sample_chunks(per_source: int, seed: int) -> list[dict]:
    """Scroll the chunks collection and pick `per_source` per source_name."""
    rng = random.Random(seed)
    client = get_client()

    collected: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=app_config.CHUNK_COLLECTION,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            collected.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "text": (payload.get("text") or "").strip(),
                    "source_name": payload.get("source_name") or "unknown",
                    "chunk_index": payload.get("chunk_index", 0),
                }
            )
        if offset is None:
            break

    by_source: dict[str, list[dict]] = defaultdict(list)
    for ch in collected:
        if ch["text"]:
            by_source[ch["source_name"]].append(ch)

    samples: list[dict] = []
    for source, chunks in sorted(by_source.items()):
        rng.shuffle(chunks)
        samples.extend(chunks[:per_source])

    print(
        f"  sampled {len(samples)} chunks across {len(by_source)} sources",
        file=sys.stderr,
    )
    return samples


def build_user_prompt(samples: list[dict]) -> str:
    lines = ["Sample of chunks from the ingested corpus:", ""]
    for s in samples:
        lines.append(f"--- source: {s['source_name']} (chunk {s['chunk_index']}) ---")
        lines.append(s["text"])
        lines.append("")
    lines.append(
        "Now generate the 20 questions per the system instructions. "
        "Return JSON only."
    )
    return "\n".join(lines)


def call_generator(prompt: str, model: str) -> dict:
    """Reuse the project's OpenRouter client but with the gen-questions system prompt."""
    import qa_client

    original_prompt = qa_client.SYSTEM_PROMPT
    qa_client.SYSTEM_PROMPT = GEN_SYSTEM_PROMPT
    try:
        # 20 questions with metadata easily exceed the 768-token QA cap;
        # request more headroom for this single call.
        response = call_openrouter(prompt, model, temperature=0.3, max_tokens=3072)
    finally:
        qa_client.SYSTEM_PROMPT = original_prompt

    if "choices" not in response:
        raise RuntimeError(f"Unexpected OpenRouter response: {response}")
    raw = response["choices"][0]["message"]["content"]
    return parse_llm_json(raw)


def validate(payload: dict) -> list[dict]:
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Top-level 'questions' must be a list.")
    if len(questions) != 20:
        raise ValueError(f"Expected 20 questions, got {len(questions)}.")
    counts = defaultdict(int)
    for q in questions:
        for field in ("id", "question", "category", "expected_difficulty"):
            if field not in q or not q[field]:
                raise ValueError(f"Question missing required field '{field}': {q}")
        if q["category"] not in CATEGORIES:
            raise ValueError(f"Unknown category {q['category']!r}; allowed: {CATEGORIES}")
        counts[q["category"]] += 1
    for cat in CATEGORIES:
        if counts[cat] != 5:
            raise ValueError(f"Category {cat} must have 5 questions, got {counts[cat]}.")
    return questions


def main() -> int:
    args = parse_args()

    model = args.model or os.getenv("QA_MODEL") or app_config.QA_MODEL
    if not model:
        print("ERROR: QA_MODEL not set. Add to .env or pass --model.", file=sys.stderr)
        return 1

    print(f"== Generating questions with model: {model}", file=sys.stderr)
    samples = sample_chunks(args.per_source, args.seed)
    if not samples:
        print("ERROR: no chunks found in Qdrant. Run ingest first.", file=sys.stderr)
        return 1

    prompt = build_user_prompt(samples)
    print(f"  prompt size: {len(prompt)} chars", file=sys.stderr)

    payload = call_generator(prompt, model)
    questions = validate(payload)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(questions, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(questions)} questions to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
