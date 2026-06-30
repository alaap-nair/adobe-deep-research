from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from neo4j import GraphDatabase


@dataclass
class GraphChunkResult:
    source: str
    text: str
    score: float
    concepts: list[str]


class Neo4jBioGraph:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        trust_all_certificates: bool = False,
    ) -> None:
        if trust_all_certificates:
            uri = uri.replace("neo4j+s://", "neo4j+ssc://", 1).replace("bolt+s://", "bolt+ssc://", 1)
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def create_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT bio_source_path IF NOT EXISTS FOR (n:BioSource) REQUIRE n.path IS UNIQUE",
            "CREATE CONSTRAINT bio_chunk_id IF NOT EXISTS FOR (n:BioChunk) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT bio_term_value IF NOT EXISTS FOR (n:BioTerm) REQUIRE n.value IS UNIQUE",
            "CREATE CONSTRAINT bio_concept_value IF NOT EXISTS FOR (n:BioConcept) REQUIRE n.value IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()

    def clear_graph(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (n)
                WHERE n:BioSource OR n:BioChunk OR n:BioTerm OR n:BioConcept
                DETACH DELETE n
                """
            ).consume()

    def _batched(self, rows: Iterable[dict], batch_size: int) -> Iterator[list[dict]]:
        batch: list[dict] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def ingest_source(self, source_path: str, source_name: str, chunks: Iterable[dict], batch_size: int = 64) -> int:
        chunk_count = 0
        previous_chunk_id: str | None = None
        next_pairs_buffer: list[dict[str, str]] = []
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (source:BioSource {path: $source_path})
                SET source.name = $source_name
                WITH source
                OPTIONAL MATCH (source)-[:HAS_CHUNK]->(existing:BioChunk)
                DETACH DELETE existing
                """,
                source_path=source_path,
                source_name=source_name,
            ).consume()

            for batch in self._batched(chunks, batch_size=batch_size):
                chunk_count += len(batch)
                session.run(
                    """
                    MERGE (source:BioSource {path: $source_path})
                    SET source.name = $source_name
                    WITH source
                    UNWIND $rows AS row
                    MERGE (chunk:BioChunk {id: row.chunk_id})
                    SET chunk.text = row.text, chunk.source = $source_name, chunk.ordinal = row.ordinal
                    MERGE (source)-[:HAS_CHUNK]->(chunk)
                    FOREACH (term_row IN row.terms |
                        MERGE (term:BioTerm {value: term_row.value})
                        MERGE (chunk)-[term_rel:HAS_TERM]->(term)
                        SET term_rel.count = term_row.count
                    )
                    FOREACH (concept_value IN row.concepts |
                        MERGE (concept:BioConcept {value: concept_value})
                        MERGE (chunk)-[:MENTIONS]->(concept)
                    )
                    FOREACH (pair IN row.concept_pairs |
                        MERGE (left:BioConcept {value: pair.left})
                        MERGE (right:BioConcept {value: pair.right})
                        MERGE (left)-[rel:RELATED_TO]->(right)
                        SET rel.weight = coalesce(rel.weight, 0) + 1
                        MERGE (right)-[back:RELATED_TO]->(left)
                        SET back.weight = coalesce(back.weight, 0) + 1
                    )
                    """,
                    source_path=source_path,
                    source_name=source_name,
                    rows=batch,
                ).consume()

                batch_ids = [row["chunk_id"] for row in batch]
                if previous_chunk_id:
                    next_pairs_buffer.append({"left_id": previous_chunk_id, "right_id": batch_ids[0]})
                next_pairs_buffer.extend(
                    {"left_id": left_id, "right_id": right_id}
                    for left_id, right_id in zip(batch_ids, batch_ids[1:])
                )
                previous_chunk_id = batch_ids[-1]
                if len(next_pairs_buffer) >= batch_size * 4:
                    session.run(
                        """
                        UNWIND $pairs AS pair
                        MATCH (left:BioChunk {id: pair.left_id})
                        MATCH (right:BioChunk {id: pair.right_id})
                        MERGE (left)-[:NEXT]->(right)
                        """,
                        pairs=next_pairs_buffer,
                    ).consume()
                    next_pairs_buffer = []

            if next_pairs_buffer:
                session.run(
                    """
                    UNWIND $pairs AS pair
                    MATCH (left:BioChunk {id: pair.left_id})
                    MATCH (right:BioChunk {id: pair.right_id})
                    MERGE (left)-[:NEXT]->(right)
                    """,
                    pairs=next_pairs_buffer,
                ).consume()

            session.run(
                """
                MATCH (term:BioTerm)
                WHERE NOT EXISTS { MATCH (:BioChunk)-[:HAS_TERM]->(term) }
                DETACH DELETE term
                """
            ).consume()
            session.run(
                """
                MATCH (concept:BioConcept)
                WHERE NOT EXISTS { MATCH (:BioChunk)-[:MENTIONS]->(concept) }
                DETACH DELETE concept
                """
            ).consume()
        return chunk_count

    def search_chunks(self, question_terms: list[str], question_concepts: list[str], top_k: int = 3) -> list[GraphChunkResult]:
        with self.driver.session(database=self.database) as session:
            records = session.run(
                """
                WITH $question_terms AS question_terms, $question_concepts AS question_concepts
                MATCH (source:BioSource)-[:HAS_CHUNK]->(chunk:BioChunk)
                OPTIONAL MATCH (chunk)-[term_rel:HAS_TERM]->(term:BioTerm)
                WHERE term.value IN question_terms
                WITH source, chunk, question_concepts,
                     count(DISTINCT term) AS unique_overlap,
                     coalesce(sum(term_rel.count), 0) AS total_overlap
                OPTIONAL MATCH (chunk)-[:MENTIONS]->(concept:BioConcept)
                WHERE concept.value IN question_concepts
                WITH source, chunk, unique_overlap, total_overlap,
                     collect(DISTINCT concept.value) AS matched_concepts,
                     question_concepts
                OPTIONAL MATCH (chunk)-[:MENTIONS]->(chunk_concept:BioConcept)-[rel:RELATED_TO]->(neighbor:BioConcept)
                WHERE chunk_concept.value IN matched_concepts OR neighbor.value IN question_concepts
                WITH source, chunk, matched_concepts, unique_overlap, total_overlap,
                     collect(DISTINCT neighbor.value)[0..8] AS expanded_concepts,
                     coalesce(sum(rel.weight), 0) AS concept_weight
                WITH source, chunk,
                     matched_concepts + expanded_concepts AS all_concepts,
                     (unique_overlap * 2.0) + total_overlap + (size(matched_concepts) * 3.0) + (concept_weight * 0.25) AS score
                RETURN source.name AS source,
                       chunk.text AS text,
                       score AS score,
                       [concept IN all_concepts WHERE concept IS NOT NULL][0..8] AS concepts
                ORDER BY score DESC, chunk.ordinal ASC
                LIMIT $top_k
                """,
                question_terms=question_terms,
                question_concepts=question_concepts,
                top_k=top_k,
            )
            return [GraphChunkResult(**record.data()) for record in records]
