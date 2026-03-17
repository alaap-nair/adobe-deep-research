"""
graph_schema.py -- ID generation and Pydantic models for the knowledge graph.

Provides deterministic IDs that link entities and triples across Neo4j and Qdrant.
The same input always produces the same ID, so both databases stay in sync
without a lookup table.
"""

import hashlib
import re
import uuid
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


class GraphRelation(BaseModel):
    """An edge in the knowledge graph."""
    triple_id: str
    head_entity_id: str
    tail_entity_id: str
    relation: str
    evidence: str


def build_graph_objects(triples: list[dict]) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """
    Convert raw extraction triples into structured GraphEntity and GraphRelation objects.
    Handles deduplication of entities and collects all surface forms.
    """
    entity_map: dict[str, GraphEntity] = {}
    relations: list[GraphRelation] = []

    for t in triples:
        head_name = t["head"].strip()
        tail_name = t["tail"].strip()
        rel = t["relation"].strip()
        evidence = t["evidence"].strip()

        head_eid = entity_id(head_name)
        tail_eid = entity_id(tail_name)

        # Upsert head entity
        if head_eid not in entity_map:
            entity_map[head_eid] = GraphEntity(
                entity_id=head_eid,
                name=normalize_name(head_name).replace("_", " "),
                original_names=[head_name],
            )
        elif head_name not in entity_map[head_eid].original_names:
            entity_map[head_eid].original_names.append(head_name)

        # Upsert tail entity
        if tail_eid not in entity_map:
            entity_map[tail_eid] = GraphEntity(
                entity_id=tail_eid,
                name=normalize_name(tail_name).replace("_", " "),
                original_names=[tail_name],
            )
        elif tail_name not in entity_map[tail_eid].original_names:
            entity_map[tail_eid].original_names.append(tail_name)

        # Build relation
        tid = triple_id(head_name, rel, tail_name)
        relations.append(GraphRelation(
            triple_id=tid,
            head_entity_id=head_eid,
            tail_entity_id=tail_eid,
            relation=rel,
            evidence=evidence,
        ))

    entities = sorted(entity_map.values(), key=lambda e: e.name)
    return entities, relations
