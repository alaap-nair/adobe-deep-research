"""
graphiti_setup.py -- Build a live Graphiti client on the project's existing Neo4j.

Design choices (see demo plan):
- LLM client is env-driven. Preferred: Anthropic (claude-haiku-4-5) for fast,
  reliable structured extraction. Fallback: any OpenAI-compatible endpoint
  (e.g. OpenRouter) via OpenAIGenericClient. Whichever key is present wins.
- Embedder is the project's local BAAI/bge-large-en-v1.5 via sentence-transformers
  (LocalEmbedder below) -- no embedding-API key, consistent with the rest of the
  hybrid system, zero per-call cost.
- group_id = "maya" partitions everything into Maya's personal subgraph.

Run `python -m demo.backend.seed_maya` first to populate, then serve `server.py`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.config import LLMConfig

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "maya")

# Local embedding model -- reuse the project's bge-large (already downloaded).
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


class LocalEmbedder(EmbedderClient):
    """Graphiti embedder backed by a local sentence-transformers model.

    Lets Graphiti run with zero embedding-API cost and the same bge-large model
    the rest of the dual-store system uses. Loaded lazily so importing this
    module is cheap.
    """

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def create(self, input_data) -> list[float]:
        text = input_data if isinstance(input_data, str) else " ".join(input_data)
        vec = self._ensure().encode(text, normalize_embeddings=True)
        return vec.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vecs = self._ensure().encode(list(input_data_list), normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def _build_llm_and_reranker():
    """Pick the LLM provider from the environment.

    Returns (llm_client, cross_encoder). Anthropic is preferred; otherwise any
    OpenAI-compatible endpoint (OpenRouter by default).
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if anthropic_key:
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        model = os.getenv("GRAPHITI_LLM_MODEL", "claude-haiku-4-5")
        llm = AnthropicClient(
            config=LLMConfig(api_key=anthropic_key, model=model, small_model=model)
        )
        # Local BGE cross-encoder so reranking needs no extra API key.
        return llm, BGERerankerClient()

    if openrouter_key:
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        model = os.getenv("GRAPHITI_LLM_MODEL", "openai/gpt-4o-mini")
        cfg = LLMConfig(
            api_key=openrouter_key,
            model=model,
            small_model=model,
            base_url=os.getenv("GRAPHITI_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        llm = OpenAIGenericClient(config=cfg)
        return llm, OpenAIRerankerClient(client=llm, config=cfg)

    raise RuntimeError(
        "No LLM credential found. Set ANTHROPIC_API_KEY (preferred) or "
        "OPENROUTER_API_KEY in .env for live Graphiti ingestion."
    )


def build_graphiti() -> Graphiti:
    """Construct a Graphiti client wired to the existing Neo4j + local embedder."""
    llm_client, cross_encoder = _build_llm_and_reranker()
    return Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=LocalEmbedder(),
        cross_encoder=cross_encoder,
    )
