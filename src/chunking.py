"""
chunking.py -- Lightweight source-text chunking helpers.

Chunks are paragraph-first, then sentence-fallback when a paragraph is too long.
The resulting records stay small enough for vector retrieval while keeping a
stable, human-readable chunk_id for citations.
"""

from __future__ import annotations

import os
import re


def normalize_source_name(source_name: str) -> str:
    """Normalize a source label into a stable chunk-id prefix."""
    base = os.path.basename(source_name.strip()) if source_name else "source"
    base = re.sub(r"\.[^.]+$", "", base)
    base = re.sub(r"[^a-zA-Z0-9]+", "_", base.strip().lower())
    return base.strip("_") or "source"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a simple punctuation-based heuristic."""
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _clean_overlap_tail(text: str, tail_len: int) -> str:
    """Keep overlap context without starting the next chunk mid-word."""
    if tail_len <= 0:
        return ""

    tail = text[-tail_len:]
    if not tail:
        return ""

    if tail[0].isalnum():
        boundary = re.search(r"\s+", tail)
        if boundary:
            tail = tail[boundary.end() :]
        else:
            return ""

    return tail.strip()


def _split_long_unit(unit: str, max_chars: int) -> list[str]:
    """Split a unit that is still too long after sentence splitting."""
    if len(unit) <= max_chars:
        return [unit.strip()]

    sentences = split_sentences(unit)
    if len(sentences) > 1:
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())
        return chunks

    words = unit.split()
    if not words:
        return []

    chunks = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join(current_words + [word]).strip()
        if current_words and len(candidate) > max_chars:
            chunks.append(" ".join(current_words).strip())
            current_words = [word]
        else:
            current_words.append(word)
    if current_words:
        chunks.append(" ".join(current_words).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    """
    Chunk text paragraph-first, then sentence fallback.

    A small character overlap is preserved between adjacent chunks so the
    retrieval side keeps enough local context for answer synthesis.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(_split_long_unit(paragraph, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit.strip()
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            if overlap > 0:
                tail_budget = max_chars - len(unit.strip()) - 2
                tail_len = max(0, min(overlap, tail_budget))
                tail = _clean_overlap_tail(current, tail_len)
                current = f"{tail}\n\n{unit}".strip() if tail else unit.strip()
            else:
                current = unit.strip()
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def build_chunk_records(
    text: str,
    source_name: str = "source",
    max_chars: int = 700,
    overlap: int = 100,
) -> list[dict]:
    """Convert source text into chunk records with stable ids and metadata."""
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    source_prefix = normalize_source_name(source_name)
    return [
        {
            "chunk_id": f"chunk:{source_prefix}:{index}",
            "text": chunk,
            "source_name": source_name,
            "chunk_index": index,
        }
        for index, chunk in enumerate(chunks)
    ]
