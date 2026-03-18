import os
from dataclasses import dataclass


def _rel_type(relation: str) -> str:
    # Neo4j relationship types must be uppercase and use underscores.
    return (relation or "").strip().upper()


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str | None = None

    @staticmethod
    def from_env() -> "Neo4jConfig":
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DATABASE") or None
        if not password:
            raise EnvironmentError("Missing NEO4J_PASSWORD in environment/.env")
        return Neo4jConfig(uri=uri, user=user, password=password, database=database)


class Neo4jEngine:
    def __init__(self, cfg: Neo4jConfig):
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError("Install neo4j driver: pip install neo4j") from e

        self._cfg = cfg
        self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self):
        self._driver.close()

    def _session(self):
        if self._cfg.database:
            return self._driver.session(database=self._cfg.database)
        return self._driver.session()

    def setup_schema(self, *, vector_dimensions: int, vector_index_name: str = "chunk_embedding_index"):
        """
        Creates constraints and a vector index for (:Chunk {embedding}).
        Compatible with Neo4j 5.11+ vector indexes.
        """
        cypher = [
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            f"""CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {int(vector_dimensions)}, `vector.similarity_function`: 'cosine'}}}}""",
        ]
        with self._session() as s:
            for q in cypher:
                s.run(q).consume()

    def upsert_chunks(self, chunks: list[dict]):
        """
        chunks: [{id, text, source, chunk_index, embedding}]
        """
        q = """
UNWIND $rows AS row
MERGE (c:Chunk {id: row.id})
SET c.text = row.text,
    c.source = row.source,
    c.chunk_index = row.chunk_index,
    c.embedding = row.embedding
"""
        with self._session() as s:
            s.run(q, rows=chunks).consume()

    def upsert_entities(self, names: list[str]):
        q = """
UNWIND $names AS name
WITH trim(name) AS n WHERE n IS NOT NULL AND n <> ""
MERGE (:Entity {name: n})
"""
        with self._session() as s:
            s.run(q, names=names).consume()

    def upsert_triples(self, triples: list[dict]):
        """
        triples: [{head, relation, tail, evidence}]
        Creates (:Entity)-[:REL {evidence}]->(:Entity)
        """
        q = """
UNWIND $triples AS t
WITH t
WHERE t.head IS NOT NULL AND t.tail IS NOT NULL AND t.relation IS NOT NULL
MERGE (h:Entity {name: trim(t.head)})
MERGE (ta:Entity {name: trim(t.tail)})
WITH h, ta, t, toUpper(trim(t.relation)) AS rel
CALL apoc.merge.relationship(h, rel, {}, {evidence: t.evidence}, ta) YIELD rel AS r
RETURN count(r) AS created
"""
        # APOC is commonly installed; if not available, fall back to static relationship types.
        with self._session() as s:
            try:
                s.run(q, triples=triples).consume()
                return
            except Exception:
                pass

        # Fallback without APOC: store relation as a property on a generic relationship type.
        q2 = """
UNWIND $triples AS t
MERGE (h:Entity {name: trim(t.head)})
MERGE (ta:Entity {name: trim(t.tail)})
MERGE (h)-[r:RELATED_TO {relation: trim(t.relation)}]->(ta)
SET r.evidence = t.evidence
"""
        with self._session() as s:
            s.run(q2, triples=triples).consume()

    def link_chunk_mentions(self, chunk_ids_and_text: list[tuple[str, str]], entity_names: list[str]):
        """
        Simple string-match linker: (:Chunk)-[:MENTIONS]->(:Entity)
        """
        # Build in python to avoid heavy cypher string ops.
        pairs = []
        lowered_entities = [(e, e.lower()) for e in entity_names if e]
        for cid, text in chunk_ids_and_text:
            lt = (text or "").lower()
            for e, el in lowered_entities:
                if el and el in lt:
                    pairs.append({"chunk_id": cid, "entity": e})

        if not pairs:
            return

        q = """
UNWIND $pairs AS p
MATCH (c:Chunk {id: p.chunk_id})
MATCH (e:Entity {name: p.entity})
MERGE (c)-[:MENTIONS]->(e)
"""
        with self._session() as s:
            s.run(q, pairs=pairs).consume()

    def vector_search(self, query_embedding: list[float], *, k: int = 5, index_name: str = "chunk_embedding_index"):
        """
        Returns top-k chunks by vector similarity. Requires Neo4j 5.11+.
        """
        q = f"""
CALL db.index.vector.queryNodes('{index_name}', $k, $emb)
YIELD node, score
RETURN node.id AS id, node.source AS source, node.chunk_index AS chunk_index, score AS score, node.text AS text
ORDER BY score DESC
"""
        with self._session() as s:
            rows = s.run(q, k=int(k), emb=query_embedding).data()
        return rows

