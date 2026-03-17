"""
test_build_vectorstore.py -- Tests for Qdrant vector store ingestion.

Uses Qdrant in-memory mode (no server required).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from sentence_transformers import SentenceTransformer

from graph_schema import build_graph_objects, to_qdrant_id
from build_vectorstore import (
    create_collections,
    upsert_entity_vectors,
    upsert_evidence_vectors,
    embed_texts,
    build_vectorstore,
)
from config import ENTITY_COLLECTION, EVIDENCE_COLLECTION


# Use a small model for fast tests
@pytest.fixture(scope="module")
def small_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def dim(small_model):
    return small_model.get_sentence_embedding_dimension()


class TestEmbedTexts:
    def test_returns_list(self, small_model):
        vecs = embed_texts(["hello world"], small_model)
        assert isinstance(vecs, list)
        assert isinstance(vecs[0], list)

    def test_correct_dimension(self, small_model):
        vecs = embed_texts(["test"], small_model)
        assert len(vecs[0]) == small_model.get_sentence_embedding_dimension()

    def test_batch(self, small_model):
        vecs = embed_texts(["a", "b", "c"], small_model)
        assert len(vecs) == 3


class TestCreateCollections:
    def test_collections_created(self, qdrant_memory_client, dim):
        create_collections(qdrant_memory_client, dim)
        assert qdrant_memory_client.collection_exists(ENTITY_COLLECTION)
        assert qdrant_memory_client.collection_exists(EVIDENCE_COLLECTION)

    def test_idempotent(self, qdrant_memory_client, dim):
        create_collections(qdrant_memory_client, dim)
        create_collections(qdrant_memory_client, dim)  # should not raise


class TestUpsertEntityVectors:
    def test_upserts(self, qdrant_memory_client, sample_graph_objects, small_model, dim):
        entities, _ = sample_graph_objects
        create_collections(qdrant_memory_client, dim)
        upsert_entity_vectors(qdrant_memory_client, entities, small_model)
        count = qdrant_memory_client.count(ENTITY_COLLECTION).count
        assert count == len(entities)

    def test_idempotent(self, qdrant_memory_client, sample_graph_objects, small_model, dim):
        entities, _ = sample_graph_objects
        create_collections(qdrant_memory_client, dim)
        upsert_entity_vectors(qdrant_memory_client, entities, small_model)
        upsert_entity_vectors(qdrant_memory_client, entities, small_model)
        count = qdrant_memory_client.count(ENTITY_COLLECTION).count
        assert count == len(entities)  # no duplicates


class TestUpsertEvidenceVectors:
    def test_upserts(self, qdrant_memory_client, sample_graph_objects, small_model, dim):
        _, relations = sample_graph_objects
        create_collections(qdrant_memory_client, dim)
        upsert_evidence_vectors(qdrant_memory_client, relations, small_model)
        count = qdrant_memory_client.count(EVIDENCE_COLLECTION).count
        assert count == len(relations)


class TestBuildVectorstore:
    def test_end_to_end(self, qdrant_memory_client, sample_graph_objects, small_model):
        entities, relations = sample_graph_objects
        build_vectorstore(entities, relations, qdrant_memory_client, small_model)
        ent_count = qdrant_memory_client.count(ENTITY_COLLECTION).count
        ev_count = qdrant_memory_client.count(EVIDENCE_COLLECTION).count
        assert ent_count == len(entities)
        assert ev_count == len(relations)


class TestIdLinkage:
    def test_entity_ids_match(self, qdrant_memory_client, sample_graph_objects, small_model, dim):
        """Verify that Qdrant point payloads contain entity_ids matching graph_schema."""
        entities, relations = sample_graph_objects
        create_collections(qdrant_memory_client, dim)
        upsert_entity_vectors(qdrant_memory_client, entities, small_model)

        for entity in entities:
            qdrant_uuid = to_qdrant_id(entity.entity_id)
            points = qdrant_memory_client.retrieve(
                collection_name=ENTITY_COLLECTION,
                ids=[qdrant_uuid],
                with_payload=True,
            )
            assert len(points) == 1
            assert points[0].payload["entity_id"] == entity.entity_id

    def test_triple_ids_match(self, qdrant_memory_client, sample_graph_objects, small_model, dim):
        """Verify that Qdrant evidence payloads contain triple_ids matching graph_schema."""
        entities, relations = sample_graph_objects
        create_collections(qdrant_memory_client, dim)
        upsert_evidence_vectors(qdrant_memory_client, relations, small_model)

        for rel in relations:
            qdrant_uuid = to_qdrant_id(rel.triple_id)
            points = qdrant_memory_client.retrieve(
                collection_name=EVIDENCE_COLLECTION,
                ids=[qdrant_uuid],
                with_payload=True,
            )
            assert len(points) == 1
            assert points[0].payload["triple_id"] == rel.triple_id
            assert points[0].payload["head_entity_id"] == rel.head_entity_id
            assert points[0].payload["tail_entity_id"] == rel.tail_entity_id


class TestSemanticSearch:
    def test_search_finds_relevant_entity(self, qdrant_memory_client, sample_graph_objects, small_model, dim):
        """Search for 'energy molecule' should return ATP in top results."""
        entities, relations = sample_graph_objects
        create_collections(qdrant_memory_client, dim)
        upsert_entity_vectors(qdrant_memory_client, entities, small_model)

        query_vec = embed_texts(["energy molecule"], small_model)[0]
        results = qdrant_memory_client.query_points(
            collection_name=ENTITY_COLLECTION,
            query=query_vec,
            limit=3,
            with_payload=True,
        ).points
        names = [r.payload["name"] for r in results]
        # ATP should be among top results for "energy molecule"
        assert any("atp" in n.lower() for n in names)

    def test_search_finds_relevant_evidence(self, qdrant_memory_client, sample_graph_objects, small_model, dim):
        """Search should return results with valid payloads from the evidence collection."""
        entities, relations = sample_graph_objects
        create_collections(qdrant_memory_client, dim)
        upsert_evidence_vectors(qdrant_memory_client, relations, small_model)

        query_vec = embed_texts(["glycolysis cytosol"], small_model)[0]
        results = qdrant_memory_client.query_points(
            collection_name=EVIDENCE_COLLECTION,
            query=query_vec,
            limit=5,
            with_payload=True,
        ).points
        # All 5 evidence entries should be returned
        assert len(results) == len(relations)
        # Each result should have valid payload fields
        for r in results:
            assert "triple_id" in r.payload
            assert "evidence" in r.payload
            assert "head_entity_id" in r.payload
