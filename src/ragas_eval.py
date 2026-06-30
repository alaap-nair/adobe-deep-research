"""
ragas_eval.py -- RAGAS evaluation for the hybrid KG/RAG pipeline.

Runs Faithfulness, Answer Relevancy, Context Recall, and Context Precision
metrics using OpenRouter as the LLM backend. Reads ground truth from
data/ground_truth.json and retrieves contexts + answers from the live pipeline.

Usage:
    python src/ragas_eval.py                          # run on all ground truth questions
    python src/ragas_eval.py --from-traces            # use existing trace files (no live query)
    python src/ragas_eval.py --output results.json    # save raw scores to file
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(ROOT, "data", "ground_truth.json")
ANSWERS_DIR = os.path.join(ROOT, "outputs", "answers")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RAGAS_MODEL = os.getenv("RAGAS_MODEL", os.getenv("MODEL_NAME", "mistralai/mistral-small-3.1-24b-instruct:free"))


def slugify(question: str) -> str:
    slug = question.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")[:80].rstrip("_")


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def extract_contexts_from_trace(trace: dict) -> list[str]:
    """Pull text contexts from a trace file for RAGAS evaluation."""
    contexts = []
    retrieval = trace.get("retrieval_context", {})

    # Chunk texts (primary retrieval contexts)
    for chunk in retrieval.get("vector_hits", {}).get("chunks", []):
        text = chunk.get("text") or chunk.get("snippet", "")
        if text and text.strip():
            contexts.append(text.strip())

    # Evidence sentences from triples
    for ev in retrieval.get("vector_hits", {}).get("evidence", []):
        text = ev.get("evidence", "")
        if text and text.strip():
            contexts.append(text.strip())

    # Graph traversal evidence
    for edge in retrieval.get("graph_trace", {}).get("traversed_edges", []):
        text = edge.get("evidence", "")
        if text and text.strip() and text.strip() not in contexts:
            contexts.append(text.strip())

    return contexts


def load_from_traces(ground_truth: list[dict]) -> list[dict]:
    """Load system answers and contexts from existing trace files."""
    samples = []
    for gt in ground_truth:
        question = gt["question"]
        slug = slugify(question)
        trace_path = os.path.join(ANSWERS_DIR, f"{slug}_trace.json")
        answer_path = os.path.join(ANSWERS_DIR, f"{slug}.json")

        if not os.path.exists(trace_path):
            print(f"  SKIP (no trace): {question}")
            continue

        with open(trace_path) as f:
            trace = json.load(f)

        contexts = extract_contexts_from_trace(trace)

        # Get the system answer
        answer = ""
        if os.path.exists(answer_path):
            with open(answer_path) as f:
                answer_data = json.load(f)
                answer = answer_data.get("answer", "")
        if not answer:
            result = trace.get("result", {})
            answer = result.get("answer", "I don't know based on the provided context.")

        samples.append({
            "question": question,
            "ground_truth": gt["ground_truth"],
            "answer": answer,
            "contexts": contexts if contexts else ["No context retrieved."],
        })

    return samples


def run_live_queries(ground_truth: list[dict]) -> list[dict]:
    """Run each question through the live pipeline to get fresh contexts + answers."""
    from query_engine import QueryEngine
    from qa_client import answer_question
    import config as app_config

    engine = QueryEngine()
    model_name = getattr(app_config, "QA_MODEL", "") or getattr(app_config, "MODEL_NAME", "")
    samples = []

    try:
        for gt in ground_truth:
            question = gt["question"]
            print(f"  Querying: {question}")
            try:
                context = engine.retrieve_context(question)
                result = answer_question(question, context, model_name)

                contexts = extract_contexts_from_trace({"retrieval_context": context})
                answer = result.get("answer", "I don't know based on the provided context.")

                # Save trace for future use
                slug = slugify(question)
                trace_path = os.path.join(ANSWERS_DIR, f"{slug}_trace.json")
                answer_path = os.path.join(ANSWERS_DIR, f"{slug}.json")
                os.makedirs(ANSWERS_DIR, exist_ok=True)

                with open(trace_path, "w") as f:
                    json.dump({"question": question, "retrieval_context": context, "result": result}, f, indent=2)
                with open(answer_path, "w") as f:
                    json.dump(result, f, indent=2)

                samples.append({
                    "question": question,
                    "ground_truth": gt["ground_truth"],
                    "answer": answer,
                    "contexts": contexts if contexts else ["No context retrieved."],
                })
            except Exception as e:
                print(f"  ERROR on '{question}': {e}")
                samples.append({
                    "question": question,
                    "ground_truth": gt["ground_truth"],
                    "answer": "Error: could not generate answer.",
                    "contexts": ["No context retrieved."],
                })
    finally:
        engine.close()

    return samples


def run_ragas(samples: list[dict], output_path: str | None = None):
    """Run RAGAS metrics on the collected samples and print a summary table."""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextRecall,
        LLMContextPrecisionWithoutReference,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHFEmbeddings

    # Configure LLM via OpenRouter
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=RAGAS_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/alaap-nair/adobe-deep-research",
        },
    ))

    # Use local sentence-transformers for embeddings (same model as pipeline)
    embeddings = LangchainEmbeddingsWrapper(LCHFEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
    ))

    # Build RAGAS dataset
    ragas_samples = []
    for s in samples:
        ragas_samples.append(SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            reference=s["ground_truth"],
            retrieved_contexts=s["contexts"],
        ))

    dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithoutReference(),
    ]

    print(f"\nRunning RAGAS evaluation on {len(samples)} questions...")
    print(f"LLM: {RAGAS_MODEL}\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )

    # Extract per-question scores
    df = result.to_pandas()

    # Print summary table
    print("\n" + "=" * 100)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 100)

    # Map metric names to display columns
    col_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_recall": "context_recall",
        "llm_context_recall": "context_recall",
        "context_precision": "context_precision",
        "llm_context_precision_without_reference": "context_precision",
    }

    # Normalize column names
    rename_map = {}
    for col in df.columns:
        if col in col_map:
            rename_map[col] = col_map[col]
    df = df.rename(columns=rename_map)

    header = f"{'Question':<55} {'Faith':>7} {'Relev':>7} {'Recall':>7} {'Prec':>7}"
    print(header)
    print("-" * 100)

    for _, row in df.iterrows():
        q = row.get("user_input", "")[:53]
        faith = row.get("faithfulness", float("nan"))
        relev = row.get("answer_relevancy", float("nan"))
        recall = row.get("context_recall", float("nan"))
        prec = row.get("context_precision", float("nan"))
        print(f"{q:<55} {faith:>7.3f} {relev:>7.3f} {recall:>7.3f} {prec:>7.3f}")

    print("-" * 100)

    # Averages
    avg_faith = df.get("faithfulness", df.iloc[:, 0]).mean() if "faithfulness" in df else 0
    avg_relev = df.get("answer_relevancy", df.iloc[:, 0]).mean() if "answer_relevancy" in df else 0
    avg_recall = df.get("context_recall", df.iloc[:, 0]).mean() if "context_recall" in df else 0
    avg_prec = df.get("context_precision", df.iloc[:, 0]).mean() if "context_precision" in df else 0
    print(f"{'AVERAGE':<55} {avg_faith:>7.3f} {avg_relev:>7.3f} {avg_recall:>7.3f} {avg_prec:>7.3f}")
    print("=" * 100)

    # Save results
    results_payload = {
        "model": RAGAS_MODEL,
        "num_questions": len(samples),
        "averages": {
            "faithfulness": round(float(avg_faith), 4),
            "answer_relevancy": round(float(avg_relev), 4),
            "context_recall": round(float(avg_recall), 4),
            "context_precision": round(float(avg_prec), 4),
        },
        "per_question": [],
    }

    for _, row in df.iterrows():
        results_payload["per_question"].append({
            "question": row.get("user_input", ""),
            "faithfulness": round(float(row.get("faithfulness", 0)), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 4),
            "context_recall": round(float(row.get("context_recall", 0)), 4),
            "context_precision": round(float(row.get("context_precision", 0)), 4),
        })

    out = output_path or os.path.join(ROOT, "outputs", "ragas_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults saved to {out}")

    return results_payload


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the KG/RAG pipeline")
    parser.add_argument("--from-traces", action="store_true",
                        help="Use existing trace files instead of live queries")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth questions")

    if args.from_traces:
        print("Loading from existing trace files...")
        samples = load_from_traces(ground_truth)
    else:
        print("Running live queries...")
        samples = run_live_queries(ground_truth)

    if not samples:
        print("No samples to evaluate!")
        return

    print(f"Prepared {len(samples)} samples for RAGAS evaluation")
    run_ragas(samples, args.output)


if __name__ == "__main__":
    main()
