from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)
from ragas.llms import LangchainLLMWrapper


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


def build_metrics(llm: ChatOpenAI) -> list:
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm),
        ContextPrecisionWithoutReference(llm=llm),
    ]


def maybe_add_reference_metrics(metrics: list, llm: ChatOpenAI, rows: list[dict]) -> list:
    if all(row.get("reference") for row in rows):
        metrics.append(ContextRecall(llm=llm))
        metrics.append(AnswerCorrectness(llm=llm))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG outputs with RAGAS.")
    parser.add_argument("input_json", type=Path, help="Path to a JSON object or list of JSON objects")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for ragas_eval.py")
    base_url = os.getenv("OPENAI_BASE_URL")

    rows = load_rows(args.input_json)
    validate_rows(rows)

    llm = ChatOpenAI(model=args.model, api_key=api_key, base_url=base_url, temperature=0)
    embeddings = OpenAIEmbeddings(model=args.embedding_model, api_key=api_key, base_url=base_url)
    ragas_llm = LangchainLLMWrapper(
        llm,
        client=AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key),
    )

    metrics = maybe_add_reference_metrics(build_metrics(ragas_llm), ragas_llm, rows)
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm, embeddings=embeddings)
    print(result)
    print(json.dumps(result.to_pandas().to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
