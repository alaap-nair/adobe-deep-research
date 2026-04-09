"""
build_vectorstore.py -- Qdrant Vector Store Assembly.

Embeds entities and evidence text, then upserts into Qdrant collections.
Point IDs are deterministic UUIDs derived from the same IDs used in Neo4j,
so you can jump between the two databases using a common identifier.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config import (
    QDRANT_PATH,
    QDRANT_URL,
    QDRANT_PORT,
    QDRANT_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    ENTITY_COLLECTION,
    EVIDENCE_COLLECTION,
    CHUNK_COLLECTION,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP,
)
from graph_schema import (
    GraphEntity,
    GraphRelation,
    build_graph_objects,
    to_qdrant_id,
)
from chunking import build_chunk_records

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Module-level model cache
_model = None


def get_model_dimension(model: SentenceTransformer) -> int:
    """Return the embedding dimension across sentence-transformers versions."""
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], model: SentenceTransformer = None) -> list[list[float]]:
    """Batch-encode texts into embeddings."""
    if model is None:
        model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def get_client(url=None, port=None, api_key=None, path=None) -> QdrantClient:
    """
    Create a Qdrant client.

    By default uses local file storage (no server needed).
    Set QDRANT_URL in .env to use a remote server instead.
    """
    remote_url = url or QDRANT_URL
    if remote_url:
        return QdrantClient(
            url=remote_url,
            port=port or QDRANT_PORT,
            api_key=api_key or QDRANT_API_KEY,
        )
    # Local file-based storage (no server required)
    local_path = path or QDRANT_PATH
    os.makedirs(local_path, exist_ok=True)
    return QdrantClient(path=local_path)


def create_collections(client: QdrantClient, dim: int = None):
    """Create the entity, evidence, and chunk collections if they don't exist."""
    dim = dim or EMBEDDING_DIM
    for name in [ENTITY_COLLECTION, EVIDENCE_COLLECTION, CHUNK_COLLECTION]:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )


def upsert_entity_vectors(
    client: QdrantClient,
    entities: list[GraphEntity],
    model: SentenceTransformer = None,
):
    """Embed entity names and upsert into the entities collection."""
    if not entities:
        return
    texts = [e.name for e in entities]
    vectors = embed_texts(texts, model)
    points = [
        PointStruct(
            id=to_qdrant_id(e.entity_id),
            vector=vec,
            payload={
                "entity_id": e.entity_id,
                "name": e.name,
                "original_names": e.original_names,
            },
        )
        for e, vec in zip(entities, vectors)
    ]
    client.upsert(collection_name=ENTITY_COLLECTION, points=points)


def upsert_evidence_vectors(
    client: QdrantClient,
    relations: list[GraphRelation],
    model: SentenceTransformer = None,
):
    """Embed evidence text and upsert into the evidence collection."""
    if not relations:
        return
    texts = [r.evidence for r in relations]
    vectors = embed_texts(texts, model)
    points = [
        PointStruct(
            id=to_qdrant_id(r.triple_id),
            vector=vec,
            payload={
                "triple_id": r.triple_id,
                "head_entity_id": r.head_entity_id,
                "tail_entity_id": r.tail_entity_id,
                "relation": r.relation,
                "evidence": r.evidence,
            },
        )
        for r, vec in zip(relations, vectors)
    ]
    client.upsert(collection_name=EVIDENCE_COLLECTION, points=points)


def upsert_chunk_vectors(
    client: QdrantClient,
    chunks: list[dict],
    model: SentenceTransformer = None,
):
    """Embed chunk text and upsert into the chunks collection."""
    if not chunks:
        return
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts, model)
    points = [
        PointStruct(
            id=to_qdrant_id(chunk["chunk_id"]),
            vector=vec,
            payload={
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source_name": chunk["source_name"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=CHUNK_COLLECTION, points=points)


def build_vectorstore(
    entities: list[GraphEntity],
    relations: list[GraphRelation],
    client: QdrantClient = None,
    model: SentenceTransformer = None,
    chunks: list[dict] = None,
    source_text: str = None,
    source_name: str = "source",
):
    """
    Main entry point: embed and upsert entities + evidence into Qdrant.

    Args:
        entities: Structured graph entities (from build_graph_objects or build_graph).
        relations: Structured graph relations.
        client: Optional Qdrant client (uses default from config if None).
        model: Optional pre-loaded SentenceTransformer model.
        chunks: Optional pre-built chunk records to index.
        source_text: Optional raw source text to chunk and index if chunks are not passed.
        source_name: Label used when generating chunk ids from source_text.
    """
    close_client = False
    if client is None:
        client = get_client()
        close_client = True

    if model is None:
        model = get_embedding_model()

    try:
        create_collections(client, get_model_dimension(model))
        upsert_entity_vectors(client, entities, model)
        upsert_evidence_vectors(client, relations, model)
        if chunks is None and source_text:
            chunks = build_chunk_records(
                source_text,
                source_name=source_name,
                max_chars=CHUNK_MAX_CHARS,
                overlap=CHUNK_OVERLAP,
            )
        upsert_chunk_vectors(client, chunks or [], model)

        ent_count = client.count(ENTITY_COLLECTION).count
        ev_count = client.count(EVIDENCE_COLLECTION).count
        chunk_count = client.count(CHUNK_COLLECTION).count
        print(
            f"Qdrant: {ent_count} entity vectors, {ev_count} evidence vectors, "
            f"{chunk_count} chunk vectors"
        )
    finally:
        if close_client:
            client.close()


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs", "triples.json")
    with open(json_path) as f:
        data = json.load(f)

    print(f"Loading {len(data['triples'])} triples from {json_path}")
    entities, relations = build_graph_objects(data["triples"])
    build_vectorstore(entities, relations)
    print("Done.")
