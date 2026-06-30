"""
build_graph.py -- Neo4j Knowledge Graph Assembly.

Ingests extracted triples into Neo4j as a labeled property graph.
Entities become :Entity nodes, triples become :RELATES_TO edges.
All IDs are deterministic and match the Qdrant vector store.
"""

import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graph_schema import build_graph_objects, GraphEntity, GraphRelation, Episode

# Cypher label whitelist: alphanumerics + underscore. We pass schema-vetted
# strings from domain_schema.NODE_TYPES, but defense in depth.
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_driver(uri=None, user=None, password=None):
    """Create a Neo4j driver instance."""
    return GraphDatabase.driver(
        uri or NEO4J_URI,
        auth=(user or NEO4J_USER, password or NEO4J_PASSWORD),
    )


def create_constraints(driver):
    """Create uniqueness constraints and indexes for idempotent upserts."""
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
        )
        session.run(
            "CREATE INDEX triple_id_index IF NOT EXISTS "
            "FOR ()-[r:RELATES_TO]-() ON (r.triple_id)"
        )
        session.run(
            "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS "
            "FOR (e:Episodic) REQUIRE e.episode_id IS UNIQUE"
        )


def upsert_entities(driver, entities: list[GraphEntity]):
    """
    Batch-upsert entity nodes. Idempotent via MERGE on entity_id.

    W3B: entities with a `node_type` get a second label (e.g. :Entity:Molecule)
    so Cypher queries can filter by domain type. Entities without a type fall
    back to plain :Entity.
    """
    by_type: dict[str | None, list[GraphEntity]] = defaultdict(list)
    for ent in entities:
        by_type[getattr(ent, "node_type", None)].append(ent)

    with driver.session() as session:
        # Untyped (legacy) entities
        if by_type.get(None):
            session.run(
                """
                UNWIND $entities AS e
                MERGE (n:Entity {entity_id: e.entity_id})
                SET n.name = e.name, n.original_names = e.original_names
                """,
                entities=[e.model_dump() for e in by_type[None]],
            )
        # Typed entities -- one query per type so the label can be inlined safely
        for node_type, ents in by_type.items():
            if node_type is None:
                continue
            if not _LABEL_RE.match(node_type):
                raise ValueError(f"Invalid node label for Cypher: {node_type!r}")
            query = (
                "UNWIND $entities AS e "
                "MERGE (n:Entity {entity_id: e.entity_id}) "
                "SET n.name = e.name, n.original_names = e.original_names, "
                f"n.node_type = e.node_type, n:{node_type}"
            )
            session.run(query, entities=[e.model_dump() for e in ents])


def upsert_relations(driver, relations: list[GraphRelation]):
    """
    Batch-upsert relationship edges. Idempotent via MERGE on triple_id.

    Bi-temporal fields are stamped on each edge: `created_at` is set only ON
    CREATE so the original ingest time survives re-runs, while the event-time
    (`valid_at`) and retraction fields (`invalid_at`/`expired_at`) are kept
    current. `episode_ids` accumulates -- the same fact asserted by multiple
    episodes records every episode that supports it.
    """
    query = """
    UNWIND $relations AS r
    MATCH (h:Entity {entity_id: r.head_entity_id})
    MATCH (t:Entity {entity_id: r.tail_entity_id})
    MERGE (h)-[rel:RELATES_TO {triple_id: r.triple_id}]->(t)
    ON CREATE SET rel.created_at = r.created_at
    SET rel.relation = r.relation,
        rel.evidence = r.evidence,
        rel.relation_type = r.relation_type,
        rel.valid_at = r.valid_at,
        rel.invalid_at = r.invalid_at,
        rel.expired_at = r.expired_at,
        rel.episode_ids = CASE
            WHEN rel.episode_ids IS NULL THEN r.episode_ids
            ELSE rel.episode_ids + [x IN r.episode_ids WHERE NOT x IN rel.episode_ids]
        END
    """
    records = [r.model_dump() for r in relations]
    with driver.session() as session:
        session.run(query, relations=records)


def upsert_episode(driver, episode: Episode, entity_ids: list[str]):
    """
    Upsert the :Episodic provenance node and link it to every entity it
    mentioned via :MENTIONS. Idempotent via MERGE on episode_id.
    """
    with driver.session() as session:
        session.run(
            """
            MERGE (ep:Episodic {episode_id: $episode_id})
            ON CREATE SET ep.created_at = $created_at
            SET ep.source = $source,
                ep.content = $content,
                ep.valid_at = $valid_at
            """,
            episode_id=episode.episode_id,
            source=episode.source,
            content=episode.content,
            valid_at=episode.valid_at,
            created_at=episode.created_at,
        )
        if entity_ids:
            session.run(
                """
                MATCH (ep:Episodic {episode_id: $episode_id})
                UNWIND $entity_ids AS eid
                MATCH (n:Entity {entity_id: eid})
                MERGE (ep)-[:MENTIONS]->(n)
                """,
                episode_id=episode.episode_id,
                entity_ids=entity_ids,
            )


def invalidate_contradicted_edges(driver, relations: list[GraphRelation], episode: Episode) -> int:
    """
    Temporal conflict resolution (phase 3).

    For each new relation whose `relation_type` is *functional* (single-valued,
    see domain_schema.FUNCTIONAL_RELATIONS), find any existing currently-valid
    edge with the same head and relation_type but a different tail and mark it
    contradicted: `invalid_at` = the new fact's event time, `expired_at` = the
    ingest time. The superseding edge itself is untouched.

    Returns the number of edges invalidated. No-op unless edges carry a
    relation_type, so legacy/untyped graphs are unaffected.
    """
    from domain_schema import FUNCTIONAL_RELATIONS

    news = [
        {
            "head_entity_id": r.head_entity_id,
            "tail_entity_id": r.tail_entity_id,
            "relation_type": r.relation_type,
            "triple_id": r.triple_id,
            "valid_at": r.valid_at or episode.valid_at,
            "expired_at": episode.created_at,
        }
        for r in relations
        if r.relation_type in FUNCTIONAL_RELATIONS
    ]
    if not news:
        return 0

    query = """
    UNWIND $news AS n
    MATCH (h:Entity {entity_id: n.head_entity_id})-[old:RELATES_TO]->(t2:Entity)
    WHERE old.relation_type = n.relation_type
      AND old.triple_id <> n.triple_id
      AND t2.entity_id <> n.tail_entity_id
      AND old.invalid_at IS NULL
      AND old.expired_at IS NULL
    SET old.invalid_at = n.valid_at, old.expired_at = n.expired_at
    RETURN count(old) AS c
    """
    with driver.session() as session:
        return session.run(query, news=news).single()["c"]


def clear_graph(driver):
    """Delete all nodes and relationships. Use for testing / re-runs."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def get_graph_stats(driver) -> dict:
    """Return node and relationship counts."""
    with driver.session() as session:
        nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
        episodes = session.run("MATCH (e:Episodic) RETURN count(e) AS c").single()["c"]
    return {"nodes": nodes, "relationships": rels, "episodes": episodes}


def build_graph(
    triples: list[dict],
    driver=None,
    episode: Episode | None = None,
    incremental_resolve: bool | None = None,
) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """
    Main entry point: convert raw triples to graph objects and ingest into Neo4j.

    If `episode` is provided, relations are stamped with its provenance/temporal
    fields and an :Episodic node is created linking to every entity it mentioned.
    When omitted the behavior is unchanged (no episode node, default timestamps).

    If `incremental_resolve` is True (or env KG_INCREMENTAL_RESOLVE=1), the
    batch's entities are resolved against entities already in the graph before
    upsert, so an ingest attaches to established nodes instead of forking
    near-duplicates (phase 4). Default off to keep ingests deterministic.

    Returns the structured (entities, relations) for downstream use
    (e.g., vectorstore ingestion, visualization).
    """
    entities, relations = build_graph_objects(triples, episode=episode)

    if incremental_resolve is None:
        incremental_resolve = os.getenv("KG_INCREMENTAL_RESOLVE", "0") == "1"

    close_driver = False
    if driver is None:
        driver = get_driver()
        close_driver = True

    try:
        create_constraints(driver)
        if incremental_resolve:
            from canonicalize import resolve_against_graph

            entities, relations = resolve_against_graph(driver, entities, relations)
        upsert_entities(driver, entities)
        upsert_relations(driver, relations)
        if episode is not None:
            upsert_episode(driver, episode, [e.entity_id for e in entities])
            invalidated = invalidate_contradicted_edges(driver, relations, episode)
            if invalidated:
                print(f"Neo4j: invalidated {invalidated} contradicted edge(s)")
        stats = get_graph_stats(driver)
        print(
            f"Neo4j: {stats['nodes']} nodes, {stats['relationships']} relationships, "
            f"{stats['episodes']} episodes"
        )
    finally:
        if close_driver:
            driver.close()

    return entities, relations


if __name__ == "__main__":
    # Load from an output JSON and ingest into Neo4j
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs", "triples.json")
    with open(json_path) as f:
        data = json.load(f)

    print(f"Loading {len(data['triples'])} triples from {json_path}")
    entities, relations = build_graph(data["triples"])
    print(f"Done. {len(entities)} entities, {len(relations)} relations ingested.")
