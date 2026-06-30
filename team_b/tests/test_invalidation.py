"""
test_invalidation.py -- Phase-3 temporal conflict resolution.

Unit tests for the functional-relation predicate plus live Neo4j tests that a
new episode asserting a different tail for a functional relation invalidates the
older edge, while multi-valued relations are left intact. Live tests skip when
Neo4j is unavailable.
"""

import os
import sys
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domain_schema import is_functional_relation, FUNCTIONAL_RELATIONS


class TestFunctionalRelation:
    def test_default_set(self):
        assert is_functional_relation("LOCATED_IN")
        assert is_functional_relation("BELONGS_TO")

    def test_multivalued_not_functional(self):
        assert not is_functional_relation("PRODUCES")
        assert not is_functional_relation("CONSUMES")

    def test_none_not_functional(self):
        assert not is_functional_relation(None)


# --- Live Neo4j tests ---

try:
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from build_graph import get_driver, create_constraints, clear_graph, build_graph
    from graph_schema import build_episode, triple_id

    _d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _d.verify_connectivity()
    _d.close()
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False

live = pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")


@pytest.fixture
def driver():
    d = get_driver()
    clear_graph(d)
    create_constraints(d)
    yield d
    clear_graph(d)
    d.close()


def _edge_state(driver, head, relation, tail):
    tid = triple_id(head, relation, tail)
    with driver.session() as s:
        rec = s.run(
            "MATCH ()-[r:RELATES_TO {triple_id: $tid}]->() "
            "RETURN r.invalid_at AS invalid",
            tid=tid,
        ).single()
    return rec


@live
def test_functional_conflict_invalidates_old_edge(driver):
    # Episode 1: the ETC is located in the inner mitochondrial membrane.
    ep1 = build_episode("doc_v1.txt", "etc located in inner membrane")
    build_graph(
        [{"head": "ETC", "relation": "located in", "tail": "inner mitochondrial membrane",
          "relation_type": "LOCATED_IN", "evidence": "..."}],
        driver, episode=ep1,
    )
    # Episode 2 (later) relocates it -- a contradiction for a functional relation.
    ep2 = build_episode("doc_v2.txt", "etc located in cytosol")
    build_graph(
        [{"head": "ETC", "relation": "located in", "tail": "cytosol",
          "relation_type": "LOCATED_IN", "evidence": "..."}],
        driver, episode=ep2,
    )

    old = _edge_state(driver, "ETC", "located in", "inner mitochondrial membrane")
    new = _edge_state(driver, "ETC", "located in", "cytosol")
    assert old["invalid"] is not None   # superseded
    assert new["invalid"] is None       # current truth


@live
def test_multivalued_relation_not_invalidated(driver):
    # Glycolysis PRODUCES ATP, then PRODUCES pyruvate -- both remain true.
    ep1 = build_episode("a.txt", "glycolysis produces atp")
    build_graph(
        [{"head": "glycolysis", "relation": "produce", "tail": "ATP",
          "relation_type": "PRODUCES", "evidence": "..."}],
        driver, episode=ep1,
    )
    ep2 = build_episode("b.txt", "glycolysis produces pyruvate")
    build_graph(
        [{"head": "glycolysis", "relation": "produce", "tail": "pyruvate",
          "relation_type": "PRODUCES", "evidence": "..."}],
        driver, episode=ep2,
    )

    assert _edge_state(driver, "glycolysis", "produce", "ATP")["invalid"] is None
    assert _edge_state(driver, "glycolysis", "produce", "pyruvate")["invalid"] is None


@live
def test_same_tail_reassertion_is_idempotent(driver):
    # Re-asserting the SAME functional fact must not invalidate it.
    ep1 = build_episode("a.txt", "x located in y")
    build_graph(
        [{"head": "X", "relation": "located in", "tail": "Y",
          "relation_type": "LOCATED_IN", "evidence": "..."}],
        driver, episode=ep1,
    )
    ep2 = build_episode("b.txt", "x located in y again")
    build_graph(
        [{"head": "X", "relation": "located in", "tail": "Y",
          "relation_type": "LOCATED_IN", "evidence": "..."}],
        driver, episode=ep2,
    )
    assert _edge_state(driver, "X", "located in", "Y")["invalid"] is None
