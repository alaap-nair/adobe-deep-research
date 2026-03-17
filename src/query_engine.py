"""
query_engine.py -- Unified query interface across Neo4j and Qdrant.

Supports semantic search (via Qdrant embeddings) and graph traversal
(via Neo4j Cypher), linked by shared entity_id / triple_id identifiers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from neo4j import GraphDatabase

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    ENTITY_COLLECTION, EVIDENCE_COLLECTION,
)
from build_vectorstore import embed_texts, get_embedding_model, get_client


class QueryEngine:
    """Unified query interface across Neo4j (graph) and Qdrant (vectors)."""

    def __init__(self, neo4j_driver=None, qdrant_client=None, model=None):
        self.neo4j = neo4j_driver or GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.qdrant = qdrant_client or get_client()
        self.model = model or get_embedding_model()

    def close(self):
        self.neo4j.close()
        self.qdrant.close()

    # --- Qdrant semantic search ---

    def search_entities(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over entity names. Returns entity_id, name, score."""
        vector = embed_texts([query], self.model)[0]
        results = self.qdrant.query_points(
            collection_name=ENTITY_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "entity_id": r.payload["entity_id"],
                "name": r.payload["name"],
                "score": r.score,
            }
            for r in results
        ]

    def search_evidence(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over evidence text. Returns triple_id, evidence, score."""
        vector = embed_texts([query], self.model)[0]
        results = self.qdrant.query_points(
            collection_name=EVIDENCE_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "triple_id": r.payload["triple_id"],
                "head_entity_id": r.payload["head_entity_id"],
                "tail_entity_id": r.payload["tail_entity_id"],
                "relation": r.payload["relation"],
                "evidence": r.payload["evidence"],
                "score": r.score,
            }
            for r in results
        ]

    # --- Neo4j graph traversal ---

    def get_entity_neighborhood(self, entity_id: str, hops: int = 1) -> dict:
        """
        Fetch all nodes and edges within N hops of the given entity in Neo4j.
        Returns {"nodes": [...], "edges": [...]}.
        """
        query = f"""
        MATCH path = (start:Entity {{entity_id: $entity_id}})-[*1..{hops}]-(connected)
        WITH nodes(path) AS ns, relationships(path) AS rs
        UNWIND ns AS n
        WITH collect(DISTINCT n) AS nodes, rs
        UNWIND rs AS r
        RETURN
            [n IN nodes | {{entity_id: n.entity_id, name: n.name}}] AS nodes,
            collect(DISTINCT {{
                triple_id: r.triple_id,
                relation: r.relation,
                evidence: r.evidence,
                source: startNode(r).entity_id,
                target: endNode(r).entity_id
            }}) AS edges
        """
        with self.neo4j.session() as session:
            result = session.run(query, entity_id=entity_id).single()
            if result is None:
                return {"nodes": [], "edges": []}
            return {"nodes": result["nodes"], "edges": result["edges"]}

    # --- Hybrid query ---

    def hybrid_query(self, query: str, top_k: int = 5, hops: int = 1) -> dict:
        """
        Semantic search + graph expansion.

        1. Search Qdrant for matching entities and evidence.
        2. For each matching entity, fetch its Neo4j neighborhood.
        3. Return combined results with scores and graph context.
        """
        entity_hits = self.search_entities(query, top_k)
        evidence_hits = self.search_evidence(query, top_k)

        # Expand top entity matches in Neo4j
        neighborhoods = {}
        for hit in entity_hits[:3]:  # limit graph expansion to top 3
            eid = hit["entity_id"]
            neighborhoods[eid] = self.get_entity_neighborhood(eid, hops)

        return {
            "entity_matches": entity_hits,
            "evidence_matches": evidence_hits,
            "graph_neighborhoods": neighborhoods,
        }


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What does glycolysis produce?"
    print(f"Query: {query}\n")

    engine = QueryEngine()
    try:
        results = engine.hybrid_query(query)

        print("--- Entity Matches ---")
        for hit in results["entity_matches"]:
            print(f"  {hit['name']} (score: {hit['score']:.4f}, id: {hit['entity_id']})")

        print("\n--- Evidence Matches ---")
        for hit in results["evidence_matches"]:
            print(f"  [{hit['relation']}] {hit['evidence']} (score: {hit['score']:.4f})")

        print("\n--- Graph Neighborhoods ---")
        for eid, neighborhood in results["graph_neighborhoods"].items():
            print(f"\n  {eid}:")
            for edge in neighborhood["edges"]:
                print(f"    {edge['source']} --[{edge['relation']}]--> {edge['target']}")
    finally:
        engine.close()
