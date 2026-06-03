"""
query_engine.py -- Unified query interface across Neo4j and Qdrant.

Supports semantic search (via Qdrant embeddings) and graph traversal
(via Neo4j Cypher), linked by shared entity_id / triple_id identifiers.
"""

import os
import sys
import difflib
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    ENTITY_COLLECTION,
    EVIDENCE_COLLECTION,
    CHUNK_COLLECTION,
    QA_TOP_K_CHUNKS,
    QA_TOP_K_ENTITIES,
    QA_TOP_K_EVIDENCE,
    QA_GRAPH_HOPS,
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
        self._entity_catalog = None
        self._entity_name_lookup = None

    def close(self):
        self.neo4j.close()
        self.qdrant.close()

    # --- Shared helpers ---

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        stopwords = {
            "what",
            "which",
            "who",
            "whom",
            "where",
            "when",
            "why",
            "how",
            "does",
            "do",
            "did",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "in",
            "on",
            "and",
            "or",
            "with",
            "from",
            "by",
            "please",
        }
        tokens = re.findall(r"[a-zA-Z0-9\u00B5\u2070-\u209F]+", (query or "").lower())
        filtered = [token for token in tokens if token not in stopwords]
        candidates: list[str] = []
        for token in filtered:
            candidates.append(token)
        for window in (2, 3):
            for index in range(len(filtered) - window + 1):
                phrase = " ".join(filtered[index : index + window]).strip()
                if phrase and phrase not in candidates:
                    candidates.append(phrase)
        return candidates

    def _scroll_collection(self, collection_name: str, limit: int = 128) -> list[dict]:
        """Fetch payloads from a Qdrant collection without assuming a single page."""
        try:
            offset = None
            records: list[dict] = []
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    records.append(point.payload or {})
                if offset is None:
                    break
            return records
        except Exception:
            return []

    def _load_entity_catalog(self) -> list[dict]:
        """Load entity payloads from Qdrant and cache normalized aliases."""
        if self._entity_catalog is not None and self._entity_name_lookup is not None:
            return self._entity_catalog

        catalog: list[dict] = []
        lookup: dict[str, dict] = {}
        payloads = self._scroll_collection(ENTITY_COLLECTION)
        if not payloads:
            for hit in self.search_entities("", top_k=32):
                payloads.append(
                    {
                        "entity_id": hit.get("entity_id"),
                        "name": hit.get("name"),
                        "original_names": [hit.get("name")] if hit.get("name") else [],
                    }
                )

        for payload in payloads:
            entity_id = payload.get("entity_id")
            name = payload.get("name") or ""
            original_names = payload.get("original_names") or []
            aliases = [name, *original_names]
            aliases = [alias for alias in aliases if alias]
            if not entity_id or not aliases:
                continue
            record = {
                "entity_id": entity_id,
                "name": name or aliases[0],
                "aliases": aliases,
                "original_names": original_names,
                "display_name": original_names[0] if original_names else (name or aliases[0]),
            }
            catalog.append(record)
            lookup[entity_id] = record

        self._entity_catalog = catalog
        self._entity_name_lookup = lookup
        return catalog

    def _entity_name(self, entity_id: str) -> str:
        record = self._entity_name_lookup.get(entity_id) if self._entity_name_lookup else None
        if record:
            return record["name"]
        return entity_id or ""

    def _entity_display_name(self, entity_id: str) -> str:
        record = self._entity_name_lookup.get(entity_id) if self._entity_name_lookup else None
        if record:
            return record.get("display_name") or record["name"]
        return entity_id or ""

    def _alias_matches(self, candidate: str, alias: str) -> float:
        candidate_norm = self._normalize_text(candidate)
        alias_norm = self._normalize_text(alias)
        if not candidate_norm or not alias_norm:
            return 0.0
        if candidate_norm == alias_norm:
            return 1.0
        if candidate_norm in alias_norm or alias_norm in candidate_norm:
            return 0.95
        ratio = difflib.SequenceMatcher(None, candidate_norm, alias_norm).ratio()
        return ratio

    def _best_entity_match(self, candidate: str) -> dict | None:
        """Find the best catalog match for a candidate query phrase."""
        best_record = None
        best_score = 0.0
        best_alias = ""
        for record in self._load_entity_catalog():
            for alias in record["aliases"]:
                score = self._alias_matches(candidate, alias)
                if score > best_score:
                    best_score = score
                    best_record = record
                    best_alias = alias
        if best_record is None or best_score < 0.78:
            return None
        match_type = "keyword" if self._normalize_text(candidate) == self._normalize_text(best_alias) else "fuzzy"
        return {
            "entity_id": best_record["entity_id"],
            "name": best_record["name"],
            "matched_text": candidate,
            "matched_alias": best_alias,
            "score": round(best_score, 4),
            "match_type": match_type,
        }

    def _dedupe_entities(self, entities: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for entity in entities:
            entity_id = entity.get("entity_id")
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            deduped.append(entity)
        return deduped

    def _dedupe_by_key(self, items: list[dict], key: str) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in items:
            value = item.get(key)
            if value is None or value in seen:
                continue
            seen.add(value)
            deduped.append(item)
        return deduped

    def _dedupe_textual_hits(
        self,
        items: list[dict],
        id_key: str,
        text_key: str,
        limit: int,
    ) -> list[dict]:
        """Dedupe retrieval hits by id and normalized text while preserving rank."""
        deduped: list[dict] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        for item in items:
            item_id = item.get(id_key)
            normalized_text = self._normalize_text(item.get(text_key, ""))
            if item_id and item_id in seen_ids:
                continue
            if normalized_text and normalized_text in seen_texts:
                continue
            if item_id:
                seen_ids.add(item_id)
            if normalized_text:
                seen_texts.add(normalized_text)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    # --- Context-precision retrieval modes (A10 Part 2) ---

    def _mmr_select(
        self, query: str, candidates: list[dict], top_k: int, lambda_mult: float = 0.7
    ) -> list[dict]:
        """Maximal Marginal Relevance selection over candidate chunks.

        Balances query relevance against diversity so the cross-chapter chunk a
        multi-hop question needs is not crowded out by a cluster of near-duplicate
        top hits. Embeddings from ``build_vectorstore`` are L2-normalized, so
        cosine similarity is just the dot product.
        """
        if len(candidates) <= top_k:
            return candidates
        texts = [c.get("text") or "" for c in candidates]
        try:
            doc_vecs = embed_texts(texts, self.model)
            query_vec = embed_texts([query], self.model)[0]
        except Exception:
            return candidates[:top_k]

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        relevance = [dot(query_vec, dv) for dv in doc_vecs]
        selected: list[int] = []
        remaining = set(range(len(candidates)))
        while remaining and len(selected) < top_k:
            best_idx, best_score = None, float("-inf")
            for i in remaining:
                diversity = max((dot(doc_vecs[i], doc_vecs[j]) for j in selected), default=0.0)
                score = lambda_mult * relevance[i] - (1 - lambda_mult) * diversity
                if score > best_score:
                    best_score, best_idx = score, i
            selected.append(best_idx)
            remaining.discard(best_idx)
        return [candidates[i] for i in selected]

    def _graph_boost_select(
        self,
        candidates: list[dict],
        top_k: int,
        resolved_entities: list[dict],
        hops: int,
    ) -> list[dict]:
        """Re-rank chunks by graph connectivity to the question's entities.

        Leverages the dual store: gather entity names within ``hops`` of the
        question's resolved entities in Neo4j, then boost chunks whose text
        mentions those graph-connected entities. This targets multi-hop
        questions where the bridging chunk sits in a different chapter than the
        literal query terms.
        """
        if len(candidates) <= top_k:
            return candidates

        neighbor_names: set[str] = set()
        for entity in resolved_entities:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue
            neighborhood = self.get_entity_neighborhood(entity_id, hops=hops)
            for node in neighborhood.get("nodes", []):
                for candidate_name in [node.get("name"), *(node.get("original_names") or [])]:
                    if candidate_name:
                        neighbor_names.add(candidate_name.lower())
        if not neighbor_names:
            return candidates[:top_k]

        boost = 0.05
        scored: list[dict] = []
        for chunk in candidates:
            text = (chunk.get("text") or "").lower()
            hits = sum(1 for name in neighbor_names if name in text)
            enriched = dict(chunk)
            enriched["graph_boost_hits"] = hits
            enriched["boosted_score"] = float(chunk.get("score") or 0.0) + boost * hits
            scored.append(enriched)
        scored.sort(key=lambda c: c["boosted_score"], reverse=True)
        return scored[:top_k]

    # --- Qdrant semantic search ---

    def search_entities(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over entity names. Returns entity_id, name, score."""
        try:
            vector = embed_texts([query], self.model)[0]
            results = self.qdrant.query_points(
                collection_name=ENTITY_COLLECTION,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            return []
        return [
            {
                "entity_id": r.payload["entity_id"],
                "name": r.payload["name"],
                "score": r.score,
            }
            for r in results
        ]

    def search_evidence(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over evidence text. Returns triple metadata and score."""
        self._load_entity_catalog()
        try:
            vector = embed_texts([query], self.model)[0]
            results = self.qdrant.query_points(
                collection_name=EVIDENCE_COLLECTION,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            return []
        return [
            {
                "triple_id": r.payload["triple_id"],
                "head_entity_id": r.payload["head_entity_id"],
                "tail_entity_id": r.payload["tail_entity_id"],
                "relation": r.payload["relation"],
                "evidence": r.payload["evidence"],
                "head_name": self._entity_display_name(r.payload["head_entity_id"]),
                "tail_name": self._entity_display_name(r.payload["tail_entity_id"]),
                "score": r.score,
            }
            for r in results
        ]

    def search_chunks(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over chunk text. Returns chunk metadata and score."""
        try:
            vector = embed_texts([query], self.model)[0]
            results = self.qdrant.query_points(
                collection_name=CHUNK_COLLECTION,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            return []
        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "text": r.payload["text"],
                "source_name": r.payload.get("source_name"),
                "chunk_index": r.payload.get("chunk_index"),
                "snippet": r.payload["text"][:400],
                "score": r.score,
            }
            for r in results
        ]

    def resolve_query_entities(self, query: str, top_k: int = None) -> list[dict]:
        """
        Resolve likely entities from the question using keywords, fuzzy matches,
        and semantic fallback.
        """
        top_k = top_k or QA_TOP_K_ENTITIES
        keywords = self._extract_keywords(query)
        resolved: list[dict] = []

        for candidate in keywords:
            match = self._best_entity_match(candidate)
            if match:
                resolved.append(match)

        resolved = self._dedupe_entities(resolved)
        if len(resolved) < top_k:
            semantic_hits = self.search_entities(query, top_k=max(top_k, 5))
            for hit in semantic_hits:
                resolved.append(
                    {
                        "entity_id": hit["entity_id"],
                        "name": hit["name"],
                        "matched_text": query,
                        "matched_alias": hit["name"],
                        "score": round(hit["score"], 4),
                        "match_type": "semantic",
                    }
                )
            resolved = self._dedupe_entities(resolved)

        resolved = sorted(
            resolved,
            key=lambda item: (
                {"keyword": 0, "fuzzy": 1, "semantic": 2}.get(item.get("match_type"), 3),
                -float(item.get("score", 0.0)),
                item.get("name", ""),
            ),
        )
        return resolved[:top_k]

    # --- Neo4j graph traversal ---

    def get_entity_neighborhood(self, entity_id: str, hops: int = 1) -> dict:
        """
        Fetch all nodes and edges within N hops of the given entity in Neo4j.
        Returns {"nodes": [...], "edges": [...]}.
        """
        query = f"""
        MATCH (start:Entity {{entity_id: $entity_id}})
        OPTIONAL MATCH path = (start)-[*1..{hops}]-(connected:Entity)
        WITH
            collect(DISTINCT start) + collect(DISTINCT connected) AS raw_nodes,
            collect(DISTINCT path) AS paths
        WITH
            [n IN raw_nodes WHERE n IS NOT NULL] AS nodes,
            [p IN paths WHERE p IS NOT NULL] AS paths
        UNWIND paths AS path
        UNWIND relationships(path) AS rel
        MATCH (source:Entity)-[edge:RELATES_TO {{triple_id: rel.triple_id}}]->(target:Entity)
        WITH nodes, collect(DISTINCT edge) AS edges
        RETURN
            [n IN nodes | {{
                entity_id: n.entity_id,
                name: n.name,
                original_names: n.original_names
            }}] AS nodes,
            [e IN edges | {{
                triple_id: e.triple_id,
                relation: e.relation,
                evidence: e.evidence,
                source_entity_id: startNode(e).entity_id,
                source_name: CASE
                    WHEN startNode(e).original_names IS NOT NULL AND size(startNode(e).original_names) > 0
                    THEN startNode(e).original_names[0]
                    ELSE startNode(e).name
                END,
                target_entity_id: endNode(e).entity_id,
                target_name: CASE
                    WHEN endNode(e).original_names IS NOT NULL AND size(endNode(e).original_names) > 0
                    THEN endNode(e).original_names[0]
                    ELSE endNode(e).name
                END
            }}] AS edges
        """
        try:
            with self.neo4j.session() as session:
                result = session.run(query, entity_id=entity_id).single()
                if result is None:
                    return {"nodes": [], "edges": []}
                return {"nodes": result["nodes"], "edges": result["edges"]}
        except Exception:
            return {"nodes": [], "edges": []}

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

    def retrieve_context(
        self,
        query: str,
        top_k_chunks: int = None,
        top_k_entities: int = None,
        top_k_evidence: int = None,
        hops: int = None,
    ) -> dict:
        """
        Retrieve structured context for answer generation.

        Returns query analysis, vector hits, and a deduped graph trace.
        """
        top_k_chunks = top_k_chunks or QA_TOP_K_CHUNKS
        top_k_entities = top_k_entities or QA_TOP_K_ENTITIES
        top_k_evidence = top_k_evidence or QA_TOP_K_EVIDENCE
        hops = hops or QA_GRAPH_HOPS

        try:
            resolved_entities = self.resolve_query_entities(query, top_k=top_k_entities)
        except TypeError:
            resolved_entities = self.resolve_query_entities(query)

        query_analysis = {
            "keywords": self._extract_keywords(query),
            "resolved_entities": resolved_entities,
        }

        # Chunk selection strategy (A10 Part 2 context-precision experiments):
        #   USE_RERANKER=1            -> cross-encoder rerank of a 3x pool (C1).
        #   RETRIEVAL_MODE=mmr        -> MMR diversity over a 6x pool (C2).
        #   RETRIEVAL_MODE=graph_boost-> graph-connectivity rescoring of a 6x pool (C3).
        #   default                   -> 2x candidates, dedupe-truncate (C0 baseline).
        retrieval_mode = os.getenv("RETRIEVAL_MODE", "").strip().lower()
        if os.getenv("USE_RERANKER", "0") == "1":
            from reranker import rerank as _rerank_chunks

            candidate_pool = self._dedupe_textual_hits(
                self.search_chunks(query, top_k=max(top_k_chunks * 3, top_k_chunks)),
                id_key="chunk_id",
                text_key="text",
                limit=top_k_chunks * 3,
            )
            chunk_hits = _rerank_chunks(query, candidate_pool, top_k_chunks)
        elif retrieval_mode in ("mmr", "graph_boost"):
            candidate_pool = self._dedupe_textual_hits(
                self.search_chunks(query, top_k=max(top_k_chunks * 6, top_k_chunks)),
                id_key="chunk_id",
                text_key="text",
                limit=top_k_chunks * 6,
            )
            if retrieval_mode == "mmr":
                chunk_hits = self._mmr_select(query, candidate_pool, top_k_chunks)
            else:
                chunk_hits = self._graph_boost_select(
                    candidate_pool, top_k_chunks, resolved_entities, hops
                )
        else:
            chunk_hits = self._dedupe_textual_hits(
                self.search_chunks(query, top_k=max(top_k_chunks * 2, top_k_chunks)),
                id_key="chunk_id",
                text_key="text",
                limit=top_k_chunks,
            )
        evidence_hits = self._dedupe_textual_hits(
            self.search_evidence(query, top_k=max(top_k_evidence * 2, top_k_evidence)),
            id_key="triple_id",
            text_key="evidence",
            limit=top_k_evidence,
        )

        seed_entity_ids: list[str] = []
        for entity in query_analysis["resolved_entities"]:
            entity_id = entity.get("entity_id")
            if entity_id and entity_id not in seed_entity_ids:
                seed_entity_ids.append(entity_id)
        for hit in evidence_hits:
            for entity_id in [hit.get("head_entity_id"), hit.get("tail_entity_id")]:
                if entity_id and entity_id not in seed_entity_ids:
                    seed_entity_ids.append(entity_id)

        retrieved_nodes: list[dict] = []
        traversed_edges: list[dict] = []
        for entity_id in seed_entity_ids:
            neighborhood = self.get_entity_neighborhood(entity_id, hops=hops)
            retrieved_nodes.extend(neighborhood.get("nodes", []))
            traversed_edges.extend(neighborhood.get("edges", []))

        retrieved_nodes = self._dedupe_by_key(retrieved_nodes, "entity_id")
        traversed_edges = self._dedupe_by_key(traversed_edges, "triple_id")[:8]

        return {
            "query_analysis": query_analysis,
            "vector_hits": {
                "chunks": chunk_hits,
                "evidence": evidence_hits,
            },
            "graph_trace": {
                "seed_entity_ids": seed_entity_ids,
                "retrieved_nodes": retrieved_nodes,
                "traversed_edges": traversed_edges,
            },
        }


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What does glycolysis produce?"
    print(f"Query: {query}\n")

    engine = QueryEngine()
    try:
        results = engine.retrieve_context(query)
        print(results)
    finally:
        engine.close()
