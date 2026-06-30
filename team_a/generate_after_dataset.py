#!/usr/bin/env python3
"""Build RAGAS-compatible AFTER rows using the Neo4j (Workstream 3) RAG pipeline."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

from dotenv import load_dotenv

from bio_qa import (
    answer_question_neo4j,
    parse_bool_env,
    require_neo4j_settings,
    synthesize_answer_with_llm,
)
from neo4j_bio_graph import Neo4jBioGraph

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "data" / "before_data.json"
FALLBACK_INPUT = REPO_ROOT / "before_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "after.json"


def resolve_input_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Input not found: {explicit}")
        return explicit
    if DEFAULT_INPUT.is_file():
        return DEFAULT_INPUT
    if FALLBACK_INPUT.is_file():
        return FALLBACK_INPUT
    raise FileNotFoundError(
        f"No before dataset found. Add {DEFAULT_INPUT} or {FALLBACK_INPUT}."
    )


def neo4j_settings_namespace(args: argparse.Namespace) -> Namespace:
    trust = args.neo4j_trust_all_certificates or parse_bool_env(os.getenv("NEO4J_TRUST_ALL_CERTIFICATES"))
    return Namespace(
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        neo4j_trust_all_certificates=trust,
    )


def row_question(entry: dict) -> str:
    if entry.get("question"):
        return str(entry["question"]).strip()
    if entry.get("user_input"):
        return str(entry["user_input"]).strip()
    raise KeyError("Each entry needs 'question' or 'user_input'")


def row_reference(entry: dict) -> str:
    ref = entry.get("ground_truth")
    if ref is None:
        ref = entry.get("reference")
    return "" if ref is None else str(ref)


def contexts_as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value)]
    return [item if isinstance(item, str) else str(item) for item in value]


def validate_output_row(row: dict, index: int) -> None:
    for key in ("user_input", "response", "retrieved_contexts", "reference"):
        if key not in row:
            raise ValueError(f"Row {index}: missing field {key!r}")
    ctx = row["retrieved_contexts"]
    if not isinstance(ctx, list):
        raise ValueError(f"Row {index}: retrieved_contexts must be a list")
    for j, item in enumerate(ctx):
        if not isinstance(item, str):
            raise ValueError(f"Row {index}: retrieved_contexts[{j}] must be str")


def run_improved_query(
    question: str,
    graph: Neo4jBioGraph,
    *,
    answer_mode: str,
    answer_model: str,
    top_k: int,
) -> tuple[str, list[str]]:
    """Neo4j graph retrieval + extractive or LLM answer (same paths as bio_qa.py)."""
    result = answer_question_neo4j(question, graph, top_k=top_k)
    if answer_mode == "llm":
        result["response"] = synthesize_answer_with_llm(question, result, answer_model)
    elif answer_mode != "extractive":
        raise ValueError(f"Unsupported answer_mode: {answer_mode}")
    return str(result.get("response", "")), contexts_as_str_list(result.get("retrieved_contexts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data/after.json for RAGAS using the improved Neo4j RAG pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Before dataset JSON (default: {DEFAULT_INPUT} or {FALLBACK_INPUT.name})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="RAGAS JSON output path")
    parser.add_argument(
        "--answer-mode",
        choices=("extractive", "llm"),
        default=os.getenv("BIO_QA_ANSWER_MODE", "llm"),
        help="Match bio_qa.py: extractive sentences vs LLM over retrieved context",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model for --answer-mode llm",
    )
    parser.add_argument("--top-k", type=int, default=int(os.getenv("BIO_QA_TOP_K", "3")))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-username", default=os.getenv("NEO4J_USERNAME"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument(
        "--neo4j-trust-all-certificates",
        action="store_true",
        help="Relax TLS verification for Neo4j (or set NEO4J_TRUST_ALL_CERTIFICATES)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    input_path = resolve_input_path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Before dataset must be a JSON array of objects.")

    uri, username, password, database, trust_all = require_neo4j_settings(neo4j_settings_namespace(args))
    graph = Neo4jBioGraph(
        uri=uri,
        username=username,
        password=password,
        database=database,
        trust_all_certificates=trust_all,
    )

    results: list[dict] = []
    total = len(payload)
    try:
        for i, entry in enumerate(payload, start=1):
            question = row_question(entry)
            reference = row_reference(entry)
            print(f"Running Q {i}/{total}: {question}")

            try:
                response, contexts = run_improved_query(
                    question,
                    graph,
                    answer_mode=args.answer_mode,
                    answer_model=args.answer_model,
                    top_k=args.top_k,
                )
            except Exception as exc:  # noqa: BLE001 — keep batch going; row still valid for RAGAS shape
                response = f"Pipeline error: {exc}"
                contexts = []

            row = {
                "user_input": question,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
            validate_output_row(row, i)
            results.append(row)
    finally:
        graph.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    json.loads(args.output.read_text(encoding="utf-8"))  # sanity check: round-trip parse

    print(f"\nSaved AFTER dataset to {args.output}")


if __name__ == "__main__":
    main()
