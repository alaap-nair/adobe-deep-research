"""
build_graph.py -- Neo4j Knowledge Graph Assembly.

Ingests extracted triples into Neo4j as a labeled property graph.
Entities become :Entity nodes, triples become :RELATES_TO edges.
All IDs are deterministic and match the Qdrant vector store.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graph_schema import build_graph_objects, GraphEntity, GraphRelation

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


def upsert_entities(driver, entities: list[GraphEntity]):
    """Batch-upsert entity nodes. Idempotent via MERGE on entity_id."""
    query = """
    UNWIND $entities AS e
    MERGE (n:Entity {entity_id: e.entity_id})
    SET n.name = e.name, n.original_names = e.original_names
    """
    records = [e.model_dump() for e in entities]
    with driver.session() as session:
        session.run(query, entities=records)


def upsert_relations(driver, relations: list[GraphRelation]):
    """Batch-upsert relationship edges. Idempotent via MERGE on triple_id."""
    query = """
    UNWIND $relations AS r
    MATCH (h:Entity {entity_id: r.head_entity_id})
    MATCH (t:Entity {entity_id: r.tail_entity_id})
    MERGE (h)-[rel:RELATES_TO {triple_id: r.triple_id}]->(t)
    SET rel.relation = r.relation, rel.evidence = r.evidence
    """
    records = [r.model_dump() for r in relations]
    with driver.session() as session:
        session.run(query, relations=records)


def clear_graph(driver):
    """Delete all nodes and relationships. Use for testing / re-runs."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def get_graph_stats(driver) -> dict:
    """Return node and relationship counts."""
    with driver.session() as session:
        nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
    return {"nodes": nodes, "relationships": rels}


def build_graph(triples: list[dict], driver=None) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """
    Main entry point: convert raw triples to graph objects and ingest into Neo4j.

    Returns the structured (entities, relations) for downstream use
    (e.g., vectorstore ingestion, visualization).
    """
    entities, relations = build_graph_objects(triples)

    close_driver = False
    if driver is None:
        driver = get_driver()
        close_driver = True

    try:
        create_constraints(driver)
        upsert_entities(driver, entities)
        upsert_relations(driver, relations)
        stats = get_graph_stats(driver)
        print(f"Neo4j: {stats['nodes']} nodes, {stats['relationships']} relationships")
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
