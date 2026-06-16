"""
test_communities.py -- Phase-5 Louvain community detection.

Pure tests for partition_edges (offline, no DB) plus a live Neo4j test that
:Community nodes and :IN_COMMUNITY edges are written for a two-cluster graph.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from communities import partition_edges


class TestPartitionEdges:
    def test_empty(self):
        assert partition_edges([]) == []

    def test_two_clusters_separate(self):
        # Two triangles joined by a single bridge edge -> two communities.
        edges = [
            ("a", "b"), ("b", "c"), ("a", "c"),          # cluster 1
            ("x", "y"), ("y", "z"), ("x", "z"),          # cluster 2
            ("c", "x"),                                   # weak bridge
        ]
        parts = partition_edges(edges, seed=42)
        assert len(parts) == 2
        # a/b/c land together; x/y/z land together
        cluster_of = {}
        for i, members in enumerate(parts):
            for m in members:
                cluster_of[m] = i
        assert cluster_of["a"] == cluster_of["b"] == cluster_of["c"]
        assert cluster_of["x"] == cluster_of["y"] == cluster_of["z"]
        assert cluster_of["a"] != cluster_of["x"]

    def test_sorted_largest_first(self):
        edges = [("a", "b"), ("b", "c"), ("a", "c"), ("a", "d"), ("b", "d"),  # big
                 ("p", "q")]                                                   # small
        parts = partition_edges(edges, seed=42)
        assert len(parts) >= 2
        assert len(parts[0]) >= len(parts[-1])

    def test_deterministic_with_seed(self):
        edges = [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y"), ("y", "z"), ("c", "x")]
        assert partition_edges(edges, seed=7) == partition_edges(edges, seed=7)


# --- Live Neo4j test ---

try:
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from build_graph import get_driver, create_constraints, clear_graph, build_graph
    from graph_schema import build_episode
    from communities import build_communities

    _d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _d.verify_connectivity()
    _d.close()
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False


@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
def test_build_communities_persists_nodes_and_edges():
    driver = get_driver()
    clear_graph(driver)
    create_constraints(driver)
    try:
        ep = build_episode("doc.txt", "two clusters")
        triples = [
            # cluster 1: glycolysis hub
            {"head": "glycolysis", "relation": "produce", "tail": "ATP", "evidence": "..."},
            {"head": "glycolysis", "relation": "produce", "tail": "pyruvate", "evidence": "..."},
            {"head": "ATP", "relation": "powers", "tail": "pyruvate", "evidence": "..."},
            # cluster 2: replication hub
            {"head": "DNA polymerase", "relation": "synthesizes", "tail": "DNA", "evidence": "..."},
            {"head": "DNA polymerase", "relation": "needs", "tail": "primer", "evidence": "..."},
            {"head": "DNA", "relation": "contains", "tail": "primer", "evidence": "..."},
        ]
        build_graph(triples, driver, episode=ep)

        records = build_communities(driver)
        assert len(records) >= 2

        with driver.session() as s:
            comm_nodes = s.run("MATCH (c:Community) RETURN count(c) AS c").single()["c"]
            in_comm = s.run(
                "MATCH (:Entity)-[m:IN_COMMUNITY]->(:Community) RETURN count(m) AS c"
            ).single()["c"]
        assert comm_nodes == len(records)
        assert in_comm >= 6  # every entity assigned to a community
        # summaries are non-empty
        assert all(r["summary"] for r in records)
    finally:
        clear_graph(driver)
        with driver.session() as s:
            s.run("MATCH (c:Community) DETACH DELETE c")
        driver.close()
