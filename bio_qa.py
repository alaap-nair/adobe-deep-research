from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from neo4j_bio_graph import Neo4jBioGraph


DATA_DIR = Path("data/bio")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "during",
    "for",
    "from",
    "happens",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
}
QUESTION_FILLER_WORDS = STOP_WORDS | {
    "all",
    "any",
    "best",
    "can",
    "despite",
    "explain",
    "give",
    "good",
    "possible",
    "please",
    "preference",
    "tell",
    "thermodynamic",
    "why",
    "would",
}


@dataclass
class Chunk:
    source: str
    text: str


@dataclass
class SearchResult:
    score: float
    chunk: Chunk


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in WORD_RE.findall(text)
        if token.lower() not in STOP_WORDS and len(token) > 1
    ]


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def split_text(text: str, chunk_size: int = 900, overlap: int = 180) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def normalize_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"([A-Za-z])\s{2,}([A-Za-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(path)
        pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_extracted_text(page.extract_text() or "")
            if text:
                pages.append(f"[Page {index}] {text}")
        return "\n\n".join(pages)
    raise ValueError(f"Unsupported file type: {path}")


def load_chunks(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"} or not path.is_file():
            continue
        for piece in split_text(read_source_text(path)):
            chunks.append(Chunk(source=path.name, text=piece))
    return chunks


def iter_source_chunks(data_dir: Path) -> Iterable[tuple[Path, list[str]]]:
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"} or not path.is_file():
            continue
        yield path, split_text(read_source_text(path))


def score_chunk(question_terms: set[str], chunk: Chunk) -> float:
    chunk_terms = tokenize(chunk.text)
    if not chunk_terms:
        return 0.0

    overlap = sum(1 for term in chunk_terms if term in question_terms)
    unique_overlap = len(set(chunk_terms) & question_terms)
    density = overlap / len(chunk_terms)
    return unique_overlap * 2.0 + density * 10.0


def best_sentences(question: str, chunks: Iterable[Chunk], limit: int = 4) -> list[str]:
    question_terms = set(tokenize(question))
    ranked: list[tuple[float, str]] = []
    for chunk in chunks:
        for sentence in SENTENCE_SPLIT_RE.split(chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = tokenize(sentence)
            if not terms:
                continue
            overlap = len(set(terms) & question_terms)
            if overlap:
                ranked.append((overlap + (overlap / len(terms)), sentence))

    ranked.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    selected: list[str] = []
    for _, sentence in ranked:
        if sentence in seen:
            continue
        selected.append(sentence)
        seen.add(sentence)
        if len(selected) >= limit:
            break
    return selected


def contiguous_sentences(question: str, chunk: Chunk, window: int = 4) -> list[str]:
    question_terms = set(tokenize(question))
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(chunk.text) if sentence.strip()]
    if not sentences:
        return []

    best_index = 0
    best_score = -1.0
    for index, sentence in enumerate(sentences):
        terms = tokenize(sentence)
        if not terms:
            continue
        score = len(set(terms) & question_terms)
        if score > best_score:
            best_index = index
            best_score = score

    end = min(best_index + window, len(sentences))
    return sentences[best_index:end]


def summarize_context(chunk: Chunk, limit: int = 500) -> str:
    text = chunk.text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def rank_chunks(question: str, chunks: list[Chunk], top_k: int) -> list[SearchResult]:
    question_terms = set(tokenize(question))
    ranked = sorted(
        (SearchResult(score_chunk(question_terms, chunk), chunk) for chunk in chunks),
        key=lambda item: item.score,
        reverse=True,
    )
    positives = [item for item in ranked if item.score > 0][:top_k]
    if positives:
        return positives
    return ranked[:top_k]


def answer_question(question: str, data_dir: Path) -> dict:
    chunks = load_chunks(data_dir)
    if not chunks:
        raise FileNotFoundError(f"No .txt, .md, or .pdf files found under {data_dir}")

    top_results = rank_chunks(question, chunks, top_k=3)
    top_chunks = [result.chunk for result in top_results]

    selected_sentences = best_sentences(question, top_chunks, limit=3)
    if not selected_sentences:
        selected_sentences = contiguous_sentences(question, top_chunks[0], window=2)
    if selected_sentences:
        response = " ".join(selected_sentences)
    else:
        response = summarize_context(top_chunks[0])

    return {
        "user_input": question,
        "response": response,
        "retrieved_contexts": [chunk.text for chunk in top_chunks],
        "contexts_metadata": [
            {"source": result.chunk.source, "score": round(result.score, 3)}
            for result in top_results
        ],
    }


def parse_bool_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def require_neo4j_settings(args: argparse.Namespace) -> tuple[str, str, str, str, bool]:
    uri = args.neo4j_uri or os.getenv("NEO4J_URI")
    username = args.neo4j_username or os.getenv("NEO4J_USERNAME")
    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    database = args.neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
    trust_all_certificates = args.neo4j_trust_all_certificates or parse_bool_env(
        os.getenv("NEO4J_TRUST_ALL_CERTIFICATES")
    )
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise EnvironmentError(f"Missing Neo4j settings: {', '.join(missing)}")
    return uri, username, password, database, trust_all_certificates


def chunk_terms(text: str) -> list[dict[str, int | str]]:
    counts = Counter(tokenize(text))
    return [{"value": token, "count": count} for token, count in sorted(counts.items())]


def extract_concepts(text: str, min_token_length: int = 4) -> list[str]:
    words = [token.lower() for token in WORD_RE.findall(text)]
    filtered = [
        word
        for word in words
        if len(word) >= min_token_length and word not in QUESTION_FILLER_WORDS and not word.isdigit()
    ]
    concepts: list[str] = []
    for size in (3, 2):
        for index in range(len(filtered) - size + 1):
            phrase = " ".join(filtered[index : index + size])
            concepts.append(phrase)
    concepts.extend(filtered)
    return dedupe_preserve_order(concepts[:24])


def concept_pairs(concepts: list[str], limit: int = 20) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for index, left in enumerate(concepts):
        for right in concepts[index + 1 : index + 4]:
            if left == right:
                continue
            pairs.append({"left": left, "right": right})
            if len(pairs) >= limit:
                return pairs
    return pairs


def ingest_into_neo4j(data_dir: Path, graph: Neo4jBioGraph, reset: bool = False) -> int:
    graph.create_schema()
    if reset:
        graph.clear_graph()

    source_count = 0
    for path, pieces in iter_source_chunks(data_dir):
        rows = []
        for index, piece in enumerate(pieces):
            concepts = extract_concepts(piece)
            rows.append(
                {
                    "chunk_id": f"{path.as_posix()}::{index}",
                    "text": piece,
                    "ordinal": index,
                    "terms": chunk_terms(piece),
                    "concepts": concepts,
                    "concept_pairs": concept_pairs(concepts),
                }
            )
        if not rows:
            continue
        graph.ingest_source(path.as_posix(), path.name, rows)
        source_count += 1
    return source_count


def answer_question_neo4j(question: str, graph: Neo4jBioGraph, top_k: int = 3) -> dict:
    top_results = graph.search_chunks(tokenize(question), extract_concepts(question), top_k=top_k)
    if not top_results:
        raise FileNotFoundError("No graph-backed chunks found in Neo4j. Run with --ingest-neo4j first.")

    top_chunks = [Chunk(source=result.source, text=result.text) for result in top_results]
    selected_sentences = best_sentences(question, top_chunks, limit=3)
    if not selected_sentences:
        selected_sentences = contiguous_sentences(question, top_chunks[0], window=2)
    response = " ".join(selected_sentences) if selected_sentences else summarize_context(top_chunks[0])

    return {
        "user_input": question,
        "response": response,
        "retrieved_contexts": [chunk.text for chunk in top_chunks],
        "contexts_metadata": [
            {
                "source": result.source,
                "score": round(result.score, 3),
                "concepts": result.concepts,
            }
            for result in top_results
        ],
    }


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def synthesize_answer_with_llm(question: str, result: dict, model: str) -> str:
    client = get_openai_client()
    if client is None:
        raise EnvironmentError("OPENAI_API_KEY is required for LLM answer synthesis.")

    contexts = []
    for index, context in enumerate(result["retrieved_contexts"], start=1):
        metadata = result["contexts_metadata"][index - 1]
        concepts = ", ".join(metadata.get("concepts", [])) or "none"
        contexts.append(
            f"Context {index} | source={metadata['source']} | score={metadata['score']} | concepts={concepts}\n{context}"
        )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You answer biology questions using only the provided context. "
                    "Give a direct answer first, then a short explanation. "
                    "If the context is insufficient, say that explicitly."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n\n" + "\n\n".join(contexts),
            },
        ],
    )
    return response.output_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer biology questions from local documents.")
    parser.add_argument("question", nargs="?", help="Biology question to answer")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder containing .txt, .md, or .pdf source material",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the answer JSON for later RAGAS evaluation",
    )
    parser.add_argument(
        "--reference",
        help="Optional ground-truth answer to include in the saved JSON row",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print only the extracted answer instead of full JSON",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "neo4j"),
        default="local",
        help="Retrieval backend to use",
    )
    parser.add_argument(
        "--ingest-neo4j",
        action="store_true",
        help="Load all source files under --data-dir into Neo4j before answering",
    )
    parser.add_argument(
        "--reset-graph",
        action="store_true",
        help="Delete existing biology graph nodes before Neo4j ingestion",
    )
    parser.add_argument("--neo4j-uri", help="Neo4j connection URI")
    parser.add_argument("--neo4j-username", help="Neo4j username")
    parser.add_argument("--neo4j-password", help="Neo4j password")
    parser.add_argument("--neo4j-database", help="Neo4j database name")
    parser.add_argument(
        "--neo4j-trust-all-certificates",
        action="store_true",
        help="Disable strict TLS certificate verification for Neo4j connections",
    )
    parser.add_argument(
        "--answer-mode",
        choices=("extractive", "llm"),
        default="extractive",
        help="How to turn retrieved context into the final answer",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model for --answer-mode llm",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.question and not args.ingest_neo4j:
        raise ValueError("Provide a question or run with --ingest-neo4j.")

    result: dict | None = None
    if args.backend == "neo4j" or args.ingest_neo4j:
        uri, username, password, database, trust_all_certificates = require_neo4j_settings(args)
        graph = Neo4jBioGraph(
            uri=uri,
            username=username,
            password=password,
            database=database,
            trust_all_certificates=trust_all_certificates,
        )
        try:
            if args.ingest_neo4j:
                count = ingest_into_neo4j(args.data_dir, graph, reset=args.reset_graph)
                print(f"Ingested {count} source files into Neo4j.")
            if args.question:
                result = answer_question_neo4j(args.question, graph)
        finally:
            graph.close()
    elif args.question:
        result = answer_question(args.question, args.data_dir)

    if result is None:
        return

    if args.reference:
        result["reference"] = args.reference

    if args.answer_mode == "llm":
        result["response"] = synthesize_answer_with_llm(args.question, result, args.answer_model)

    if args.plain:
        print(result["response"])
    else:
        print(json.dumps(result, indent=2))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps([result], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
