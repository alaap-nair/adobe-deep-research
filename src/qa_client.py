"""
qa_client.py -- LLM answer synthesis for hybrid retrieval results.

Builds a bounded prompt from retrieval context, calls OpenRouter, and validates
that the returned payload matches the required assignment schema.
"""

import json
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_UNSUPPORTED_ANSWER = "I don't know based on the provided context."

SYSTEM_PROMPT = """You are answering a biology question using only the supplied retrieval context.

Return ONLY valid JSON matching this schema:
{
  "question": "...",
  "answer": "...",
  "citations": ["..."],
  "reasoning": "..."
}

Rules:
- Use only facts present in the supplied context.
- If the context is insufficient, answer exactly "I don't know based on the provided context."
- Keep the answer concise. If the answer is a single term, return only that term.
- Copy citations exactly from the allowed citation strings.
- Reasoning should be brief and grounded in the provided evidence.
"""


class QAResponse(BaseModel):
    """Validated QA payload returned to ask.py."""

    question: str
    answer: str = Field(..., min_length=1)
    citations: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., min_length=1)


def parse_llm_json(raw_content: str) -> dict[str, Any]:
    """Parse JSON from an LLM response, stripping markdown fences if needed."""
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return json.loads(content)


def truncate_text(text: str, limit: int = 120) -> str:
    """Collapse whitespace and trim long snippets for citations."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3].rstrip()}..."


def ensure_prefixed_id(prefix: str, raw_id: str) -> str:
    """Avoid double-prefixing ids that already include their namespace."""
    value = (raw_id or "unknown").strip()
    if value.startswith(f"{prefix}:"):
        return value
    return f"{prefix}:{value}"


def chunk_citation(chunk_hit: dict[str, Any]) -> str:
    """Create a stable citation string for a retrieved chunk."""
    chunk_id = ensure_prefixed_id("chunk", chunk_hit.get("chunk_id", "unknown"))
    source_name = chunk_hit.get("source_name", "unknown")
    snippet = truncate_text(chunk_hit.get("snippet") or chunk_hit.get("text", ""))
    return f"{chunk_id} | {source_name} | {snippet}"


def triple_citation(edge: dict[str, Any]) -> str:
    """Create a stable citation string for a retrieved triple."""
    triple_id = ensure_prefixed_id("triple", edge.get("triple_id", "unknown"))
    source_name = (
        edge.get("source_name")
        or edge.get("head_name")
        or edge.get("source_entity_id")
        or edge.get("head_entity_id")
        or "unknown"
    )
    relation = edge.get("relation", "related_to")
    target_name = (
        edge.get("target_name")
        or edge.get("tail_name")
        or edge.get("target_entity_id")
        or edge.get("tail_entity_id")
        or "unknown"
    )
    return f"{triple_id} | {source_name} --[{relation}]--> {target_name}"


def build_allowed_citations(context: dict[str, Any]) -> list[str]:
    """Collect and deduplicate all citation candidates present in the context."""
    citations: list[str] = []
    seen: set[str] = set()

    for chunk_hit in context.get("vector_hits", {}).get("chunks", []):
        citation = chunk_citation(chunk_hit)
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    triple_sources = []
    triple_sources.extend(context.get("vector_hits", {}).get("evidence", []))
    triple_sources.extend(context.get("graph_trace", {}).get("traversed_edges", []))
    for edge in triple_sources:
        citation = triple_citation(edge)
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    return citations


def build_prompt(question: str, context: dict[str, Any], allowed_citations: list[str]) -> str:
    """Format the bounded retrieval context into a prompt payload."""
    prompt_payload = {
        "question": question,
        "query_analysis": context.get("query_analysis", {}),
        "retrieved_chunks": context.get("vector_hits", {}).get("chunks", []),
        "evidence_hits": context.get("vector_hits", {}).get("evidence", []),
        "graph_trace": context.get("graph_trace", {}),
        "allowed_citations": allowed_citations,
    }
    return (
        "Use the following hybrid retrieval context to answer the question.\n\n"
        f"{json.dumps(prompt_payload, indent=2, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )


def call_openrouter(
    prompt: str,
    model: str,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call OpenRouter and return the parsed API response."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    for attempt in range(max_retries):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/alaap-nair/adobe-deep-research",
            },
            json={
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=90,
        )
        if response.status_code == 429 and attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("OpenRouter request failed after retries")


def build_graph_proof(
    context: dict[str, Any],
    selected_citations: list[str],
    model_reasoning: str,
) -> str:
    """Compose a compact KG proof and append the model's reasoning."""
    resolved_entities = context.get("query_analysis", {}).get("resolved_entities", [])
    preferred_entities = [
        entity for entity in resolved_entities if entity.get("match_type") in {"keyword", "fuzzy"}
    ]
    entity_pool = preferred_entities or resolved_entities

    entity_names = []
    for entity in entity_pool:
        name = entity.get("name") or entity.get("matched_name") or entity.get("entity_name")
        if name and name not in entity_names:
            entity_names.append(name)

    traversed_edges = context.get("graph_trace", {}).get("traversed_edges", [])
    citation_to_edge = {triple_citation(edge): edge for edge in traversed_edges}

    supporting_edge = None
    for citation in selected_citations:
        supporting_edge = citation_to_edge.get(citation)
        if supporting_edge:
            break
    if supporting_edge is None and traversed_edges:
        supporting_edge = traversed_edges[0]

    edge_labels = []
    ordered_edges = []
    if supporting_edge:
        ordered_edges.append(supporting_edge)
    for edge in traversed_edges:
        if edge not in ordered_edges:
            ordered_edges.append(edge)

    for edge in ordered_edges[:3]:
        label = (
            f"{edge.get('source_name', edge.get('source_entity_id', 'unknown'))} "
            f"--[{edge.get('relation', 'related_to')}]--> "
            f"{edge.get('target_name', edge.get('target_entity_id', 'unknown'))}"
        )
        edge_labels.append(label)

    proof_parts = []
    if entity_names:
        proof_parts.append(f"Matched entities: {', '.join(entity_names)}.")
    if edge_labels:
        proof_parts.append(f"Traversed graph edges: {'; '.join(edge_labels)}.")
    if supporting_edge and supporting_edge.get("evidence"):
        proof_parts.append(f"Graph evidence: \"{supporting_edge['evidence']}\".")

    cleaned_reasoning = re.sub(r"\s+", " ", model_reasoning).strip()
    if cleaned_reasoning:
        proof_parts.append(cleaned_reasoning)

    return " ".join(proof_parts).strip()


def validate_citations(
    candidate_citations: list[str],
    allowed_citations: list[str],
) -> list[str]:
    """Keep only citations present in the bounded retrieval context."""
    allowed = set(allowed_citations)
    valid = [citation for citation in candidate_citations if citation in allowed]
    if valid:
        return valid
    return allowed_citations[:2]


def answer_question(question: str, context: dict[str, Any], model: str) -> dict[str, Any]:
    """Run answer synthesis and return a validated QA payload."""
    allowed_citations = build_allowed_citations(context)
    prompt = build_prompt(question, context, allowed_citations)
    response = call_openrouter(prompt, model)

    if "choices" not in response:
        raise ValueError(f"Unexpected OpenRouter response: {response}")

    raw_content = response["choices"][0]["message"]["content"]
    raw_payload = parse_llm_json(raw_content)
    qa_response = QAResponse(**raw_payload)

    citations = validate_citations(qa_response.citations, allowed_citations)
    reasoning = build_graph_proof(context, citations, qa_response.reasoning)

    final_payload = QAResponse(
        question=question,
        answer=qa_response.answer.strip(),
        citations=citations,
        reasoning=reasoning or DEFAULT_UNSUPPORTED_ANSWER,
    )
    return final_payload.model_dump()
