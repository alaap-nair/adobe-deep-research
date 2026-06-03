"""
test_integration.py -- End-to-end integration tests.

Tests the full flow: JSON -> graph objects -> Qdrant -> verify ID linkage.
Neo4j tests are skipped if unavailable.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from sentence_transformers import SentenceTransformer

from graph_schema import build_graph_objects, to_qdrant_id
from build_vectorstore import (
    build_vectorstore,
    embed_texts,
)
from config import ENTITY_COLLECTION, EVIDENCE_COLLECTION

ROOT = os.path.dirname(os.path.dirname(__file__))
TRIPLES_PATH = os.path.join(ROOT, "outputs", "triples.json")


@pytest.fixture(scope="module")
def small_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def real_data():
    """Load the actual extraction output."""
    if not os.path.exists(TRIPLES_PATH):
        pytest.skip(f"Output file not found: {TRIPLES_PATH}")
    with open(TRIPLES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_graph_objects(real_data):
    return build_graph_objects(real_data["triples"])


class TestRealDataIngestion:
    def test_all_entities_from_triples(self, real_data, real_graph_objects):
        """Every entity in the extraction output should appear in graph objects."""
        entities, _ = real_graph_objects
        graph_names = {e.name for e in entities}
        for raw_entity in real_data["entities"]:
            normalized = raw_entity.strip().lower().replace(" ", " ")
            assert any(normalized in n or n in normalized for n in graph_names), \
                f"Entity '{raw_entity}' not found in graph objects"

    def test_triple_count_matches(self, real_data, real_graph_objects):
        _, relations = real_graph_objects
        assert len(relations) == len(real_data["triples"])


class TestVectorstoreWithRealData:
    def test_ingest_and_search(self, qdrant_memory_client, real_graph_objects, small_model):
        entities, relations = real_graph_objects
        build_vectorstore(entities, relations, qdrant_memory_client, small_model)

        # Verify counts
        ent_count = qdrant_memory_client.count(ENTITY_COLLECTION).count
        ev_count = qdrant_memory_client.count(EVIDENCE_COLLECTION).count
        assert ent_count == len(entities)
        assert ev_count == len(relations)

    def test_glycolysis_search(self, qdrant_memory_client, real_graph_objects, small_model):
        entities, relations = real_graph_objects
        build_vectorstore(entities, relations, qdrant_memory_client, small_model)

        query_vec = embed_texts(["What does glycolysis produce?"], small_model)[0]
        results = qdrant_memory_client.query_points(
            collection_name=EVIDENCE_COLLECTION,
            query=query_vec,
            limit=5,
            with_payload=True,
        ).points

        # Should find evidence about glycolysis producing ATP or pyruvate
        all_evidence = " ".join(r.payload["evidence"] for r in results)
        assert "ATP" in all_evidence or "pyruvate" in all_evidence

    def test_id_consistency_across_dbs(self, qdrant_memory_client, real_graph_objects, small_model):
        """Every entity/triple ID in Qdrant should be fetchable by its deterministic UUID."""
        entities, relations = real_graph_objects
        build_vectorstore(entities, relations, qdrant_memory_client, small_model)

        # Check all entities
        for entity in entities:
            qdrant_uuid = to_qdrant_id(entity.entity_id)
            points = qdrant_memory_client.retrieve(
                collection_name=ENTITY_COLLECTION,
                ids=[qdrant_uuid],
                with_payload=True,
            )
            assert len(points) == 1, f"Entity {entity.entity_id} not found in Qdrant"
            assert points[0].payload["entity_id"] == entity.entity_id

        # Check all triples
        for rel in relations:
            qdrant_uuid = to_qdrant_id(rel.triple_id)
            points = qdrant_memory_client.retrieve(
                collection_name=EVIDENCE_COLLECTION,
                ids=[qdrant_uuid],
                with_payload=True,
            )
            assert len(points) == 1, f"Triple {rel.triple_id} not found in Qdrant"
            assert points[0].payload["triple_id"] == rel.triple_id
