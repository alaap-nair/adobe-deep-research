"""
test_temporal_query.py -- Live Neo4j tests for phase-2 temporal traversal.

Exercises get_entity_neighborhood's temporal filter and the :RELATES_TO
traversal scoping (so :MENTIONS edges to :Episodic provenance nodes never
create spurious entity-to-entity hops). Skipped automatically when Neo4j is
unavailable. The embedding model / Qdrant are stubbed -- only graph traversal
is under test here.
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from build_graph import get_driver, create_constraints, clear_graph, build_graph
    from graph_schema import build_episode, triple_id
    from query_engine import QueryEngine

    _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _driver.verify_connectivity()
    _driver.close()
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")


# Two facts sharing a head, plus a disconnected pair co-mentioned in the episode.
TRIPLES = [
    {"head": "glycolysis", "relation": "produce", "tail": "ATP", "evidence": "Glycolysis produces ATP."},
    {"head": "glycolysis", "relation": "occur in", "tail": "cytosol", "evidence": "Glycolysis occurs in the cytosol."},
    {"head": "telomere", "relation": "shorten with", "tail": "cell division", "evidence": "Telomeres shorten with division."},
]


@pytest.fixture
def engine():
    driver = get_driver()
    clear_graph(driver)
    create_constraints(driver)
    episode = build_episode("biology.txt", "glycolysis + telomere passage")
    build_graph(TRIPLES, driver, episode=episode)
    eng = QueryEngine(neo4j_driver=driver, qdrant_client=SimpleNamespace(close=lambda: None),
                      model=SimpleNamespace())
    yield eng
    clear_graph(driver)
    eng.close()


def _tails(neighborhood):
    return {e["target_entity_id"] for e in neighborhood["edges"]}


def test_relates_to_scoping_excludes_episodic_co_mentions(engine):
    # glycolysis and telomere are unconnected by RELATES_TO but co-mentioned in
    # one episode. Pre-fix, a 2-hop untyped traversal would bridge them through
    # the :Episodic node. They must NOT appear in each other's neighborhood.
    nb = engine.get_entity_neighborhood("ent:glycolysis", hops=2)
    node_ids = {n["entity_id"] for n in nb["nodes"]}
    assert "ent:telomere" not in node_ids
    assert "ent:cell_division" not in node_ids
    assert {"ent:atp", "ent:cytosol"} <= node_ids


def test_currently_valid_excludes_invalidated_edge(engine):
    # Invalidate the glycolysis->ATP edge directly (phase 3 will automate this).
    tid = triple_id("glycolysis", "produce", "ATP")
    with engine.neo4j.session() as s:
        s.run(
            "MATCH ()-[r:RELATES_TO {triple_id: $tid}]->() SET r.invalid_at = $t",
            tid=tid, t=datetime.now(timezone.utc),
        )

    valid = engine.get_entity_neighborhood("ent:glycolysis", hops=1)
    assert "ent:atp" not in _tails(valid)        # invalidated -> hidden
    assert "ent:cytosol" in _tails(valid)         # still valid

    full = engine.get_entity_neighborhood("ent:glycolysis", hops=1, include_invalid=True)
    assert "ent:atp" in _tails(full)              # history still reachable


def test_as_of_point_in_time(engine):
    tid = triple_id("glycolysis", "produce", "ATP")
    cutoff = datetime.now(timezone.utc)
    # Edge becomes invalid AFTER the cutoff instant.
    with engine.neo4j.session() as s:
        s.run(
            "MATCH ()-[r:RELATES_TO {triple_id: $tid}]->() SET r.invalid_at = $t",
            tid=tid, t=cutoff + timedelta(days=1),
        )

    before = engine.get_entity_neighborhood("ent:glycolysis", hops=1, as_of=cutoff)
    assert "ent:atp" in _tails(before)            # still true as of cutoff

    after = engine.get_entity_neighborhood(
        "ent:glycolysis", hops=1, as_of=cutoff + timedelta(days=2)
    )
    assert "ent:atp" not in _tails(after)         # invalidated by then
