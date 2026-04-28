from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    answer_correctness,
    answer_relevancy,
    context_recall,
    faithfulness,
)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON object or list of objects.")
    return payload


def validate_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = [key for key in ("user_input", "response", "retrieved_contexts") if key not in row]
        if missing:
            raise ValueError(f"Row {index} is missing required fields: {', '.join(missing)}")


def build_metrics(rows: list[dict]) -> list:
    """Same metric set as before: faithfulness, answer relevancy, context precision (no ref)."""
    metrics = [
        faithfulness,
        answer_relevancy,
        LLMContextPrecisionWithoutReference(),
    ]
    if all(str(row.get("reference") or "").strip() for row in rows):
        metrics.append(context_recall)
        metrics.append(answer_correctness)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG outputs with RAGAS.")
    parser.add_argument("input_json", type=Path, help="Path to a JSON object or list of JSON objects")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument(
        "--write-metrics",
        type=Path,
        default=None,
        help="Write JSON of mean score per numeric metric (for batch comparisons)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for ragas_eval.py")
    base_url = os.getenv("OPENAI_BASE_URL")

    rows = load_rows(args.input_json)
    validate_rows(rows)

    llm = ChatOpenAI(model=args.model, api_key=api_key, base_url=base_url, temperature=0)
    embeddings = OpenAIEmbeddings(model=args.embedding_model, api_key=api_key, base_url=base_url)

    metrics = build_metrics(rows)
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    print(result)
    df = result.to_pandas()
    print(json.dumps(df.to_dict(orient="records"), indent=2))
    if args.write_metrics:
        means = df.mean(numeric_only=True).dropna()
        payload = {str(k): float(v) for k, v in means.items()}
        args.write_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.write_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
