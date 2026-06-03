"""
reranker.py -- Cross-encoder re-ranking of retrieved chunks.

Implements the second stage of a retrieve-then-rerank pipeline
(Nogueira & Cho 2019, "Passage Re-Ranking with BERT"). After Qdrant returns
a candidate set, we score each (query, chunk_text) pair with a cross-encoder
and return the top-k by relevance.

Activated by setting USE_RERANKER=1 in the environment. The model defaults
to cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, CPU-fast); override with
RERANKER_MODEL.
"""

import os

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MODEL = None


def is_enabled() -> bool:
    return os.getenv("USE_RERANKER", "0") == "1"


def _load_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder

        model_name = os.getenv("RERANKER_MODEL", _DEFAULT_MODEL)
        _MODEL = CrossEncoder(model_name)
    return _MODEL


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Score (query, chunk_text) pairs with a cross-encoder; return top-k by score.

    Each chunk dict gains a ``rerank_score`` field. Original ``score`` is preserved.
    Returns the top-k chunks (or all chunks if fewer than top_k).

    No-op if USE_RERANKER != 1 or chunks is empty -- returns chunks[:top_k].
    """
    if not chunks or top_k <= 0:
        return chunks[:top_k] if chunks else []
    if not is_enabled():
        return chunks[:top_k]

    model = _load_model()
    pairs = [(query, ch.get("text") or "") for ch in chunks]
    scores = model.predict(pairs)

    for ch, s in zip(chunks, scores):
        ch["rerank_score"] = float(s)

    chunks_sorted = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    return chunks_sorted[:top_k]
