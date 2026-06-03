"""
graph_schema.py -- ID generation and Pydantic models for the knowledge graph.

Provides deterministic IDs that link entities and triples across Neo4j and Qdrant.
The same input always produces the same ID, so both databases stay in sync
without a lookup table.
"""

import hashlib
import re
import uuid
from typing import Optional
from pydantic import BaseModel, Field

# Namespace for deterministic UUID5 generation (used by Qdrant point IDs)
_NAMESPACE = uuid.NAMESPACE_URL


def normalize_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace, replace spaces with underscores."""
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    return name


def entity_id(name: str) -> str:
    """Deterministic entity ID from a name. e.g. 'ATP synthase' -> 'ent:atp_synthase'"""
    return f"ent:{normalize_name(name)}"


def triple_id(head: str, relation: str, tail: str) -> str:
    """Deterministic triple ID from head|relation|tail hash."""
    key = f"{normalize_name(head)}|{normalize_name(relation)}|{normalize_name(tail)}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"triple:{h}"


def to_qdrant_id(string_id: str) -> str:
    """Convert a string ID to a deterministic UUID string for Qdrant point IDs."""
    return str(uuid.uuid5(_NAMESPACE, string_id))


class GraphEntity(BaseModel):
    """A node in the knowledge graph."""
    entity_id: str
    name: str
    original_names: list[str] = Field(default_factory=list)
    node_type: Optional[str] = None  # W3B: one of domain_schema.NODE_TYPES or None


class GraphRelation(BaseModel):
    """An edge in the knowledge graph."""
    triple_id: str
    head_entity_id: str
    tail_entity_id: str
    relation: str
    evidence: str
    relation_type: Optional[str] = None  # W3B: one of domain_schema.RELATION_TYPES or None


def build_graph_objects(
    triples: list[dict],
    canonicalize: bool | None = None,
) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """
    Convert raw extraction triples into structured GraphEntity and GraphRelation objects.
    Handles deduplication of entities and collects all surface forms.

    If `canonicalize` is True (or env var KG_CANONICALIZE=1), runs the W3A
    rule + embedding hybrid dedup pass on the result. Default behavior is
    controlled by KG_CANONICALIZE so tests can opt-out without code changes.
    """
    import os

    entity_map: dict[str, GraphEntity] = {}
    relations: list[GraphRelation] = []

    for t in triples:
        head_name = t["head"].strip()
        tail_name = t["tail"].strip()
        rel = t["relation"].strip()
        evidence = t["evidence"].strip()
        head_type = t.get("head_type")
        tail_type = t.get("tail_type")
        relation_type = t.get("relation_type")

        head_eid = entity_id(head_name)
        tail_eid = entity_id(tail_name)

        # Upsert head entity
        if head_eid not in entity_map:
            entity_map[head_eid] = GraphEntity(
                entity_id=head_eid,
                name=normalize_name(head_name).replace("_", " "),
                original_names=[head_name],
                node_type=head_type,
            )
        else:
            if head_name not in entity_map[head_eid].original_names:
                entity_map[head_eid].original_names.append(head_name)
            if entity_map[head_eid].node_type is None and head_type:
                entity_map[head_eid].node_type = head_type

        # Upsert tail entity
        if tail_eid not in entity_map:
            entity_map[tail_eid] = GraphEntity(
                entity_id=tail_eid,
                name=normalize_name(tail_name).replace("_", " "),
                original_names=[tail_name],
                node_type=tail_type,
            )
        else:
            if tail_name not in entity_map[tail_eid].original_names:
                entity_map[tail_eid].original_names.append(tail_name)
            if entity_map[tail_eid].node_type is None and tail_type:
                entity_map[tail_eid].node_type = tail_type

        # Build relation
        tid = triple_id(head_name, rel, tail_name)
        relations.append(GraphRelation(
            triple_id=tid,
            head_entity_id=head_eid,
            tail_entity_id=tail_eid,
            relation=rel,
            evidence=evidence,
            relation_type=relation_type,
        ))

    entities = sorted(entity_map.values(), key=lambda e: e.name)

    if canonicalize is None:
        canonicalize = os.getenv("KG_CANONICALIZE", "0") == "1"
    if canonicalize and entities:
        from canonicalize import apply_canonicalization

        entities, relations = apply_canonicalization(entities, relations)

    return entities, relations
