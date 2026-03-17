"""
test_build_graph.py -- Tests for Neo4j ingestion.

These tests require a running Neo4j instance. They are automatically skipped
if NEO4J_URI is not set or if the database is unreachable.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

try:
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from build_graph import (
        get_driver,
        create_constraints,
        upsert_entities,
        upsert_relations,
        clear_graph,
        get_graph_stats,
        build_graph,
    )
    from graph_schema import build_graph_objects

    # Try to connect
    _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _driver.verify_connectivity()
    _driver.close()
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")


@pytest.fixture
def neo4j_driver():
    driver = get_driver()
    clear_graph(driver)
    yield driver
    clear_graph(driver)
    driver.close()


class TestUpsertEntities:
    def test_creates_nodes(self, neo4j_driver, sample_graph_objects):
        entities, _ = sample_graph_objects
        create_constraints(neo4j_driver)
        upsert_entities(neo4j_driver, entities)
        stats = get_graph_stats(neo4j_driver)
        assert stats["nodes"] == len(entities)

    def test_idempotent(self, neo4j_driver, sample_graph_objects):
        entities, _ = sample_graph_objects
        create_constraints(neo4j_driver)
        upsert_entities(neo4j_driver, entities)
        upsert_entities(neo4j_driver, entities)
        stats = get_graph_stats(neo4j_driver)
        assert stats["nodes"] == len(entities)


class TestUpsertRelations:
    def test_creates_edges(self, neo4j_driver, sample_graph_objects):
        entities, relations = sample_graph_objects
        create_constraints(neo4j_driver)
        upsert_entities(neo4j_driver, entities)
        upsert_relations(neo4j_driver, relations)
        stats = get_graph_stats(neo4j_driver)
        assert stats["relationships"] == len(relations)

    def test_idempotent(self, neo4j_driver, sample_graph_objects):
        entities, relations = sample_graph_objects
        create_constraints(neo4j_driver)
        upsert_entities(neo4j_driver, entities)
        upsert_relations(neo4j_driver, relations)
        upsert_relations(neo4j_driver, relations)
        stats = get_graph_stats(neo4j_driver)
        assert stats["relationships"] == len(relations)


class TestBuildGraph:
    def test_end_to_end(self, neo4j_driver, sample_triples):
        entities, relations = build_graph(sample_triples, neo4j_driver)
        stats = get_graph_stats(neo4j_driver)
        assert stats["nodes"] == len(entities)
        assert stats["relationships"] == len(relations)

    def test_graph_has_correct_entities(self, neo4j_driver, sample_triples):
        build_graph(sample_triples, neo4j_driver)
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) RETURN n.name AS name ORDER BY n.name"
            )
            names = [r["name"] for r in result]
        assert "glycolysis" in names
        assert "atp" in names
        assert "cytosol" in names

    def test_graph_has_correct_relations(self, neo4j_driver, sample_triples):
        build_graph(sample_triples, neo4j_driver)
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH ()-[r:RELATES_TO]->() RETURN r.relation AS relation"
            )
            relations = [r["relation"] for r in result]
        assert "occur in" in relations
        assert "produce" in relations
