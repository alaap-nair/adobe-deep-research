import os
import time
from typing import Iterable

import requests


def embed_texts_openrouter(
    texts: list[str],
    *,
    model: str | None = None,
    api_key: str | None = None,
    batch_size: int = 16,
    timeout_s: int = 60,
) -> list[list[float]]:
    """
    Create embeddings for text chunks using OpenRouter's OpenAI-compatible embeddings endpoint.
    Docs: https://openrouter.ai/docs/api/reference/embeddings
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENROUTER_API_KEY (needed for embeddings).")
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Basic retry on rate limits
        for attempt in range(3):
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": batch},
                timeout=timeout_s,
            )
            if resp.status_code in (429, 503) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data", [])
            # rows are sorted by index
            for r in rows:
                out.append(r["embedding"])
            break
    return out


def embedding_dim(vectors: Iterable[list[float]]) -> int:
    for v in vectors:
        return len(v)
    return 0

