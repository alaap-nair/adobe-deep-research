import os
import time

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Load environment from this module's directory so imports work
# regardless of the current working directory.
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
NEO4J_WAIT_SECONDS = int(os.getenv("NEO4J_WAIT_SECONDS", "60"))

driver = None
_did_preconnect_wait = False

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _open_session():
    db = NEO4J_DATABASE.strip() if NEO4J_DATABASE else None
    return _get_driver().session(database=db) if db else _get_driver().session()


def _get_driver():
    global driver, _did_preconnect_wait
    if driver is not None:
        return driver
    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
        raise RuntimeError(
            "Missing Neo4j credentials. Set NEO4J_URI, NEO4J_USER/NEO4J_USERNAME, and NEO4J_PASSWORD."
        )
    # Aura instances can take a short time to wake; wait once before the first connection.
    if not _did_preconnect_wait and NEO4J_WAIT_SECONDS > 0:
        time.sleep(NEO4J_WAIT_SECONDS)
        _did_preconnect_wait = True
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver


def create_vector_index():
    with _open_session() as session:
        session.run(
            """
            CREATE VECTOR INDEX chunkEmbeddings IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """
        )
    print("Vector index ready.")


def ingest_triple(triple: dict, source_text: str):
    embedding = embedder.encode(source_text).tolist()
    with _open_session() as session:
        session.run(
            """
            MERGE (h:Entity {name: $head})
            MERGE (t:Entity {name: $tail})
            MERGE (h)-[r:RELATION {type: $relation}]->(t)
            MERGE (c:Chunk {text: $source_text})
            SET c.embedding = $embedding
            SET c.evidence = $evidence
            // Neo4j does not allow relationships to point to relationships.
            // We attach chunk evidence to a relationship by storing the relationship elementId on the chunk.
            SET c.supports_rel = coalesce(c.supports_rel, []) + elementId(r)
        """,
            head=triple["head"],
            tail=triple["tail"],
            relation=triple["relation"],
            source_text=source_text,
            embedding=embedding,
            evidence=triple.get("evidence", ""),
        )


def hybrid_search(query_text: str, top_k: int = 5):
    query_embedding = embedder.encode(query_text).tolist()
    with _open_session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('chunkEmbeddings', $top_k, $embedding)
            YIELD node AS chunk, score
            UNWIND coalesce(chunk.supports_rel, []) AS rid
            MATCH (h:Entity)-[r:RELATION]->(t:Entity)
            WHERE elementId(r) = rid
            RETURN h.name AS head, r.type AS relation,
                   t.name AS tail, chunk.evidence AS evidence, score
            ORDER BY score DESC
        """,
            top_k=top_k,
            embedding=query_embedding,
        )
        return [dict(row) for row in result]


def vector_retrieve_chunks(query_text: str, top_k: int = 4):
    """
    Vector retrieval over :Chunk nodes.
    Returns the most similar chunk texts (and their evidence text, if any).
    """
    query_embedding = embedder.encode(query_text).tolist()
    with _open_session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('chunkEmbeddings', $top_k, $embedding)
            YIELD node AS chunk, score
            RETURN elementId(chunk) AS chunk_id,
                   chunk.text AS text,
                   chunk.evidence AS evidence,
                   score
            ORDER BY score DESC
            """,
            top_k=top_k,
            embedding=query_embedding,
        )
        return [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "evidence": row["evidence"],
                "score": row["score"],
            }
            for row in result
        ]


def graph_traverse_entities(
    entities: list[str],
    max_hops: int = 2,
    seed_limit: int = 10,
    max_edges: int = 25,
):
    """
    Knowledge graph retrieval over :Entity nodes.
    - Seeds are matched by keyword overlap with Entity.name (case-insensitive).
    - Traversal is 1..max_hops over :RELATION relationships.
    - Returns a kg_trace structure: {nodes: [...], edges: [...]}
      where edges include optional connected chunk evidence (via Chunk.supports_rel containing
      the relationship elementId).
    """
    entities_lower = [e.strip().lower() for e in (entities or []) if e and e.strip()]
    if not entities_lower:
        return {"nodes": [], "edges": []}

    with _open_session() as session:
        seeds_result = session.run(
            """
            MATCH (e:Entity)
            WHERE any(kw IN $keywords WHERE
                toLower(e.name) = kw OR toLower(e.name) CONTAINS kw
            )
            RETURN DISTINCT e.name AS name
            LIMIT $seed_limit
            """,
            keywords=entities_lower,
            seed_limit=seed_limit,
        )
        seed_names = [row["name"] for row in seeds_result]
        if not seed_names:
            return {"nodes": [], "edges": []}

        edges_by_key: dict[tuple[str, str, str], dict] = {}

        def merge_edge(from_name: str, relation: str, to_name: str, evidence_list: list[str]):
            # Key uses direction-preserving (startNode -> endNode).
            key = (from_name, relation, to_name)
            if key not in edges_by_key:
                edges_by_key[key] = {
                    "from": from_name,
                    "to": to_name,
                    "relation": relation,
                    "evidence": set(),
                }
            for ev in (evidence_list or []):
                ev_str = (ev or "").strip()
                if ev_str:
                    edges_by_key[key]["evidence"].add(ev_str)

        # 1-hop edges directly connected to seeds.
        hop1_query = """
            MATCH (seed:Entity)-[r:RELATION]-(n:Entity)
            WHERE seed.name IN $seed_names
            OPTIONAL MATCH (c:Chunk)
            WHERE elementId(r) IN coalesce(c.supports_rel, [])
            RETURN startNode(r).name AS from_name,
                   r.type AS relation,
                   endNode(r).name AS to_name,
                   collect(DISTINCT coalesce(c.evidence, c.text, '')) AS evidence
            LIMIT $max_edges
        """
        hop1_result = session.run(hop1_query, seed_names=seed_names, max_edges=max_edges)
        for row in hop1_result:
            merge_edge(row["from_name"], row["relation"], row["to_name"], row["evidence"])

        if max_hops >= 2:
            # 2-hop edges: only edges on the second hop (r2).
            hop2_query = """
                MATCH (seed:Entity)-[r1:RELATION]-(mid:Entity)-[r2:RELATION]-(n:Entity)
                WHERE seed.name IN $seed_names
                OPTIONAL MATCH (c:Chunk)
                WHERE elementId(r2) IN coalesce(c.supports_rel, [])
                RETURN startNode(r2).name AS from_name,
                       r2.type AS relation,
                       endNode(r2).name AS to_name,
                       collect(DISTINCT coalesce(c.evidence, c.text, '')) AS evidence
                LIMIT $max_edges
            """
            hop2_result = session.run(hop2_query, seed_names=seed_names, max_edges=max_edges)
            for row in hop2_result:
                merge_edge(row["from_name"], row["relation"], row["to_name"], row["evidence"])

    # Build kg_trace with stable, JSON-serializable structures.
    node_names = set(seed_names)
    edges_out = []
    for edge in edges_by_key.values():
        node_names.add(edge["from"])
        node_names.add(edge["to"])

    # Deterministic-ish ordering for easier debugging/logging.
    for key in sorted(edges_by_key.keys()):
        e = edges_by_key[key]
        evidence_list = sorted(e["evidence"])[:5]
        edges_out.append(
            {
                "from": e["from"],
                "to": e["to"],
                "relation": e["relation"],
                "evidence": evidence_list,
            }
        )

    nodes_out = [{"name": name, "label": "Entity"} for name in sorted(node_names)]
    return {"nodes": nodes_out, "edges": edges_out}


def graph_trace_from_chunks(chunk_ids: list[str], max_edges: int = 25):
    """
    KG retrieval derived from top vector chunks:
    - Given chunk elementIds, follow Chunk.supports_rel (relationship elementIds)
      to recover the corresponding (Entity)-[:RELATION]->(Entity) facts.
    This guarantees KG usage is visible even if entity name matching fails.
    """
    chunk_ids = [c for c in (chunk_ids or []) if c]
    if not chunk_ids:
        return {"nodes": [], "edges": []}

    with _open_session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)
            WHERE elementId(c) IN $chunk_ids
            UNWIND coalesce(c.supports_rel, []) AS rid
            MATCH (h:Entity)-[r:RELATION]->(t:Entity)
            WHERE elementId(r) = rid
            RETURN h.name AS from_name,
                   r.type AS relation,
                   t.name AS to_name,
                   collect(DISTINCT coalesce(c.evidence, c.text, '')) AS evidence
            LIMIT $max_edges
            """,
            chunk_ids=chunk_ids,
            max_edges=max_edges,
        )
        edges = []
        node_names = set()
        for row in result:
            node_names.add(row["from_name"])
            node_names.add(row["to_name"])
            edges.append(
                {
                    "from": row["from_name"],
                    "to": row["to_name"],
                    "relation": row["relation"],
                    "evidence": [e for e in (row["evidence"] or []) if (e or "").strip()][:5],
                }
            )

    nodes = [{"name": name, "label": "Entity"} for name in sorted(node_names)]
    return {"nodes": nodes, "edges": edges}
