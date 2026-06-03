"""
run_ragas.py -- Quantitative evaluation of the QA pipeline with RAGAS.

Reads data/ground_truth.json, ensures every question has a fresh answer + trace
on disk (regenerating via ask.answer_question if missing), assembles the RAGAS
dataset (question, ground_truth, answer, contexts), and writes
outputs/ragas/scores.csv + summary.md.

Usage:
    python src/run_ragas.py data/ground_truth.json --out outputs/ragas/
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ask import (  # noqa: E402  -- depends on src on path
    OUTPUT_DIR as ANSWERS_DIR,
    ensure_query_resources,
    generate_answer,
    slugify_question,
    write_json,
)
import config as app_config  # noqa: E402
from query_engine import QueryEngine  # noqa: E402

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_CONTEXTS_PER_QUESTION = 20


def load_ground_truth(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a JSON list of question objects.")
    missing = [r["id"] for r in rows if not r.get("ground_truth")]
    if missing:
        print(
            f"WARNING: {len(missing)} questions have empty ground_truth: {missing}.\n"
            "RAGAS context_recall and answer correctness will be meaningless for these rows.\n"
            "Paste the human-written answers into data/ground_truth.json before running.",
            file=sys.stderr,
        )
    return rows


def trace_path_for(question: str) -> str:
    return os.path.join(ANSWERS_DIR, f"{slugify_question(question)}_trace.json")


def answer_path_for(question: str) -> str:
    return os.path.join(ANSWERS_DIR, f"{slugify_question(question)}.json")


def regenerate_trace(question: str, engine: QueryEngine) -> dict:
    """Re-run retrieval + generation for one question and persist files like ask.py does."""
    context = engine.retrieve_context(
        question,
        top_k_chunks=getattr(app_config, "QA_TOP_K_CHUNKS", 5),
        top_k_entities=getattr(app_config, "QA_TOP_K_ENTITIES", 3),
        top_k_evidence=getattr(app_config, "QA_TOP_K_EVIDENCE", 5),
        hops=getattr(app_config, "QA_GRAPH_HOPS", 2),
    )
    result = generate_answer(question, context)
    trace_payload = {
        "question": question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_context": context,
        "result": result,
    }
    write_json(answer_path_for(question), result)
    write_json(trace_path_for(question), trace_payload)
    return trace_payload


def load_or_regenerate(question: str, engine: QueryEngine, force: bool) -> dict:
    tp = trace_path_for(question)
    ap = answer_path_for(question)
    if not force and os.path.exists(tp) and os.path.exists(ap):
        with open(tp, "r", encoding="utf-8") as handle:
            return json.load(handle)
    print(f"  regenerating trace for: {question[:80]}", file=sys.stderr)
    return regenerate_trace(question, engine)


def extract_contexts(trace: dict) -> list[str]:
    """Build the contexts list RAGAS expects: text chunks first, then evidence sentences."""
    rc = trace.get("retrieval_context", {})
    vector_hits = rc.get("vector_hits", {})

    contexts: list[str] = []
    seen: set[str] = set()

    for chunk in vector_hits.get("chunks", []):
        text = (chunk.get("text") or "").strip()
        if text and text not in seen:
            contexts.append(text)
            seen.add(text)

    for ev in vector_hits.get("evidence", []):
        text = (ev.get("evidence") or "").strip()
        if text and text not in seen:
            contexts.append(text)
            seen.add(text)

    for edge in rc.get("graph_trace", {}).get("traversed_edges", []):
        text = (edge.get("evidence") or "").strip()
        if text and text not in seen:
            contexts.append(text)
            seen.add(text)

    return contexts[:MAX_CONTEXTS_PER_QUESTION]


def build_ragas_dataset(rows: list[dict], traces: dict[str, dict]):
    from datasets import Dataset

    samples = {
        "question": [],
        "ground_truth": [],
        "answer": [],
        "contexts": [],
        "id": [],
    }
    for row in rows:
        question = row["question"]
        trace = traces[row["id"]]
        answer = (trace.get("result") or {}).get("answer", "")
        contexts = extract_contexts(trace)
        if not contexts:
            contexts = [""]  # ragas crashes on empty lists
        samples["id"].append(row["id"])
        samples["question"].append(question)
        samples["ground_truth"].append(row.get("ground_truth") or "")
        samples["answer"].append(answer)
        samples["contexts"].append(contexts)
    return Dataset.from_dict(samples)


def configure_ragas_judge():
    """Wrap OpenRouter chat + BGE-large embeddings as RAGAS LLM/embedding adapters."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env -- cannot run RAGAS judge.")

    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_model = (
        os.getenv("RAGAS_JUDGE_MODEL")
        or getattr(app_config, "QA_MODEL", None)
        or os.getenv("MODEL_NAME")
    )
    if not judge_model:
        raise RuntimeError("Set RAGAS_JUDGE_MODEL or QA_MODEL in .env.")

    chat = ChatOpenAI(
        model=judge_model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        default_headers={"HTTP-Referer": "https://github.com/alaap-nair/adobe-deep-research"},
    )
    embeddings = HuggingFaceEmbeddings(model_name=getattr(app_config, "EMBEDDING_MODEL"))

    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def run_evaluation(dataset):
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )

    judge_llm, judge_embeddings = configure_ragas_judge()
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    return evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )


def write_csv(out_dir: str, rows: list[dict], scored: list[dict]) -> str:
    path = os.path.join(out_dir, "scores.csv")
    fieldnames = [
        "id",
        "question",
        "difficulty",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, scores in zip(rows, scored):
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "difficulty": row.get("difficulty", ""),
                    **{k: scores.get(k, "") for k in fieldnames[3:]},
                }
            )
    return path


def write_summary(out_dir: str, rows: list[dict], scored: list[dict]) -> str:
    path = os.path.join(out_dir, "summary.md")
    metric_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    def fmt(v):
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "-"

    lines: list[str] = []
    lines.append("# RAGAS Evaluation Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Questions: {len(rows)}")
    lines.append("")
    lines.append("| ID | Difficulty | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Question |")
    lines.append("|----|-----------|--------------|------------------|----------------|-------------------|----------|")
    for row, scores in zip(rows, scored):
        lines.append(
            f"| {row['id']} "
            f"| {row.get('difficulty', '')} "
            f"| {fmt(scores.get('faithfulness'))} "
            f"| {fmt(scores.get('answer_relevancy'))} "
            f"| {fmt(scores.get('context_recall'))} "
            f"| {fmt(scores.get('context_precision'))} "
            f"| {row['question']} |"
        )
    lines.append("")

    # Aggregate row
    avgs = {}
    for k in metric_keys:
        vals = [s.get(k) for s in scored if isinstance(s.get(k), (int, float))]
        avgs[k] = sum(vals) / len(vals) if vals else None
    lines.append("## Aggregate")
    lines.append("")
    for k, v in avgs.items():
        lines.append(f"- **{k}**: {fmt(v)}")
    lines.append("")

    # Top failures
    failures: list[tuple[str, str, float]] = []
    for row, scores in zip(rows, scored):
        for k in metric_keys:
            v = scores.get(k)
            if isinstance(v, (int, float)):
                failures.append((row["id"], k, float(v)))
    failures.sort(key=lambda x: x[2])
    lines.append("## Top failures (lowest scores)")
    lines.append("")
    for qid, metric, score in failures[:5]:
        lines.append(f"- **{qid}** / {metric}: {score:.3f}")
    lines.append("")

    lines.append("## Interpretation (2-3 sentences)")
    lines.append("")
    lines.append("_Fill in: what do these numbers reveal about where this implementation breaks down?_")
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("ground_truth", help="Path to data/ground_truth.json")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "ragas"))
    p.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Re-run ask.py for every question even if a trace already exists.",
    )
    p.add_argument(
        "--traces-dir",
        help=(
            "Override the default outputs/answers/ directory when looking up "
            "answer + trace JSONs. Used by scripts/compare_models.py to score "
            "per-model runs in outputs/model_runs/<label>/."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.traces_dir:
        global ANSWERS_DIR
        ANSWERS_DIR = os.path.abspath(args.traces_dir)

    rows = load_ground_truth(args.ground_truth)

    needs_regen = args.force_regenerate or any(
        not (os.path.exists(trace_path_for(r["question"])) and os.path.exists(answer_path_for(r["question"])))
        for r in rows
    )

    traces: dict[str, dict] = {}
    engine = None
    try:
        if needs_regen:
            engine = QueryEngine()
            ensure_query_resources(engine)
        for row in rows:
            traces[row["id"]] = load_or_regenerate(row["question"], engine, args.force_regenerate)
    finally:
        if engine is not None:
            engine.close()

    print("Building RAGAS dataset...", file=sys.stderr)
    dataset = build_ragas_dataset(rows, traces)

    print("Running RAGAS evaluation (this takes a few minutes)...", file=sys.stderr)
    result = run_evaluation(dataset)

    # RAGAS >=0.2 returns an EvaluationResult; .scores is a list[dict].
    scored: list[dict] = list(result.scores) if hasattr(result, "scores") else result["scores"]

    csv_path = write_csv(args.out, rows, scored)
    md_path = write_summary(args.out, rows, scored)

    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)

    metric_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    failures = [
        (r["id"], k, float(s.get(k)))
        for r, s in zip(rows, scored)
        for k in metric_keys
        if isinstance(s.get(k), (int, float))
    ]
    failures.sort(key=lambda x: x[2])
    print("\nLowest 3 (id / metric / score):", file=sys.stderr)
    for qid, metric, score in failures[:3]:
        print(f"  {qid} / {metric}: {score:.3f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
