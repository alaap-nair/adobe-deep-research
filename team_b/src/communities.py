"""
communities.py -- GraphRAG-style community detection (phase 5).

Partitions the entity graph into communities with the Louvain algorithm
(networkx, no Neo4j GDS plugin required), then writes :Community nodes and
(:Entity)-[:IN_COMMUNITY]->(:Community) edges back to Neo4j with a deterministic
per-community summary (top members by intra-community degree).

Only temporally-valid :RELATES_TO edges are considered by default, so the
communities reflect the *current* state of the graph; pass `as_of` for a
point-in-time partition or `include_invalid=True` to include retracted facts.

Usage:
    python src/communities.py            # build from the live Neo4j graph
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from networkx.algorithms.community import louvain_communities

from build_graph import get_driver

LOUVAIN_SEED = int(os.getenv("COMMUNITY_SEED", "42"))


def _temporal_clause(as_of: Optional[datetime], include_invalid: bool) -> str:
    if include_invalid:
        return ""
    if as_of is None:
        return "WHERE r.invalid_at IS NULL AND r.expired_at IS NULL"
    return (
        "WHERE (r.valid_at IS NULL OR r.valid_at <= $as_of) "
        "AND (r.invalid_at IS NULL OR r.invalid_at > $as_of) "
        "AND (r.expired_at IS NULL OR r.expired_at > $as_of)"
    )


def load_valid_edges(
    driver, as_of: Optional[datetime] = None, include_invalid: bool = False
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Return (edges, names) where edges is a list of (head_id, tail_id) and
    names maps entity_id -> display name. Honors the temporal filter."""
    if as_of is not None and as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    query = f"""
    MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
    {_temporal_clause(as_of, include_invalid)}
    RETURN h.entity_id AS head, h.name AS head_name,
           t.entity_id AS tail, t.name AS tail_name
    """
    params = {"as_of": as_of} if (as_of is not None and not include_invalid) else {}
    edges: list[tuple[str, str]] = []
    names: dict[str, str] = {}
    with driver.session() as session:
        for rec in session.run(query, **params):
            edges.append((rec["head"], rec["tail"]))
            names[rec["head"]] = rec["head_name"] or rec["head"]
            names[rec["tail"]] = rec["tail_name"] or rec["tail"]
    return edges, names


def partition_edges(edges: list[tuple[str, str]], seed: int = LOUVAIN_SEED) -> list[set[str]]:
    """Louvain partition of an undirected, weighted-by-multiplicity graph.

    Pure (no DB). Parallel edges between the same pair raise the edge weight so
    densely-linked entities cluster together. Returns communities sorted largest
    first for stable community ids.
    """
    if not edges:
        return []
    graph = nx.Graph()
    for head, tail in edges:
        if head == tail:
            continue
        if graph.has_edge(head, tail):
            graph[head][tail]["weight"] += 1
        else:
            graph.add_edge(head, tail, weight=1)
    if graph.number_of_nodes() == 0:
        return []
    communities = louvain_communities(graph, weight="weight", seed=seed)
    return sorted((set(c) for c in communities), key=len, reverse=True)


def _summarize(
    members: set[str], edges: list[tuple[str, str]], names: dict[str, str], top_n: int = 5
) -> tuple[str, list[str]]:
    """Deterministic summary: the highest intra-community degree members."""
    degree: dict[str, int] = {m: 0 for m in members}
    for head, tail in edges:
        if head in members and tail in members:
            degree[head] += 1
            degree[tail] += 1
    ranked = sorted(members, key=lambda m: (-degree[m], names.get(m, m)))
    top = [names.get(m, m) for m in ranked[:top_n]]
    return ", ".join(top), ranked


def build_communities(
    driver=None, as_of: Optional[datetime] = None, include_invalid: bool = False
) -> list[dict]:
    """Detect communities over the current graph and persist them to Neo4j.

    Replaces any previously-stored communities (they are fully derived from the
    graph and cheap to recompute). Returns the community records.
    """
    close = False
    if driver is None:
        driver = get_driver()
        close = True
    try:
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT community_id_unique IF NOT EXISTS "
                "FOR (c:Community) REQUIRE c.community_id IS UNIQUE"
            )

        edges, names = load_valid_edges(driver, as_of=as_of, include_invalid=include_invalid)
        partitions = partition_edges(edges)

        records: list[dict] = []
        for idx, members in enumerate(partitions):
            summary, ranked = _summarize(members, edges, names)
            records.append(
                {
                    "community_id": f"comm:{idx}",
                    "size": len(members),
                    "summary": summary,
                    "member_ids": ranked,
                }
            )

        with driver.session() as session:
            # Communities are derived -- clear and rebuild for a clean partition.
            session.run("MATCH (c:Community) DETACH DELETE c")
            if records:
                session.run(
                    """
                    UNWIND $comms AS comm
                    MERGE (c:Community {community_id: comm.community_id})
                    SET c.size = comm.size, c.summary = comm.summary,
                        c.member_ids = comm.member_ids
                    WITH c, comm
                    UNWIND comm.member_ids AS mid
                    MATCH (n:Entity {entity_id: mid})
                    MERGE (n)-[:IN_COMMUNITY]->(c)
                    """,
                    comms=records,
                )
        print(f"Communities: {len(records)} detected over {len(names)} entities")
        return records
    finally:
        if close:
            driver.close()


if __name__ == "__main__":
    for comm in build_communities():
        print(f"  {comm['community_id']} ({comm['size']}): {comm['summary']}")
