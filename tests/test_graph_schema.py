"""
test_graph_schema.py -- Tests for ID generation, normalization, and graph models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_schema import (
    normalize_name,
    entity_id,
    triple_id,
    to_qdrant_id,
    build_graph_objects,
)


class TestNormalizeName:
    def test_basic(self):
        assert normalize_name("ATP synthase") == "atp_synthase"

    def test_case_folding(self):
        assert normalize_name("Glycolysis") == "glycolysis"

    def test_whitespace_collapse(self):
        assert normalize_name("  electron  transport  chain  ") == "electron_transport_chain"

    def test_unicode_preserved(self):
        # NAD+ (superscript plus) and FADH2 (subscript 2) should be preserved
        assert normalize_name("NAD\u207a") == "nad\u207a"
        assert normalize_name("FADH\u2082") == "fadh\u2082"


class TestEntityId:
    def test_deterministic(self):
        assert entity_id("ATP") == entity_id("ATP")

    def test_case_insensitive(self):
        assert entity_id("Glycolysis") == entity_id("glycolysis")

    def test_different_names_different_ids(self):
        assert entity_id("ATP") != entity_id("ADP")

    def test_prefix(self):
        assert entity_id("glucose").startswith("ent:")


class TestTripleId:
    def test_deterministic(self):
        id1 = triple_id("glycolysis", "produce", "ATP")
        id2 = triple_id("glycolysis", "produce", "ATP")
        assert id1 == id2

    def test_order_matters(self):
        id1 = triple_id("A", "rel", "B")
        id2 = triple_id("B", "rel", "A")
        assert id1 != id2

    def test_different_relations(self):
        id1 = triple_id("A", "produce", "B")
        id2 = triple_id("A", "consume", "B")
        assert id1 != id2

    def test_prefix(self):
        assert triple_id("A", "rel", "B").startswith("triple:")


class TestQdrantId:
    def test_deterministic(self):
        assert to_qdrant_id("ent:atp") == to_qdrant_id("ent:atp")

    def test_different_inputs(self):
        assert to_qdrant_id("ent:atp") != to_qdrant_id("ent:adp")

    def test_uuid_format(self):
        qid = to_qdrant_id("ent:test")
        parts = qid.split("-")
        assert len(parts) == 5  # UUID format: 8-4-4-4-12


class TestBuildGraphObjects:
    def test_entity_count(self, sample_triples):
        entities, _ = build_graph_objects(sample_triples)
        # glycolysis, cytosol, ATP, glucose, pyruvate, mitochondrion,
        # electron transport chain, inner mitochondrial membrane = 8
        names = {e.name for e in entities}
        assert len(names) == 8

    def test_relation_count(self, sample_triples):
        _, relations = build_graph_objects(sample_triples)
        assert len(relations) == 5

    def test_entity_deduplication(self, sample_triples):
        entities, _ = build_graph_objects(sample_triples)
        entity_ids = [e.entity_id for e in entities]
        assert len(entity_ids) == len(set(entity_ids))

    def test_surface_forms_collected(self):
        triples = [
            {"head": "Glycolysis", "relation": "occur in", "tail": "cytosol", "evidence": "..."},
            {"head": "glycolysis", "relation": "produce", "tail": "ATP", "evidence": "..."},
        ]
        entities, _ = build_graph_objects(triples)
        glyc = [e for e in entities if e.entity_id == "ent:glycolysis"][0]
        assert "Glycolysis" in glyc.original_names
        assert "glycolysis" in glyc.original_names

    def test_ids_link_correctly(self, sample_triples):
        entities, relations = build_graph_objects(sample_triples)
        entity_ids = {e.entity_id for e in entities}
        for r in relations:
            assert r.head_entity_id in entity_ids
            assert r.tail_entity_id in entity_ids
