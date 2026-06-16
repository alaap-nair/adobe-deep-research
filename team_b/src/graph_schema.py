"""
graph_schema.py -- ID generation and Pydantic models for the knowledge graph.

Provides deterministic IDs that link entities and triples across Neo4j and Qdrant.
The same input always produces the same ID, so both databases stay in sync
without a lookup table.
"""

import hashlib
import re
import uuid
from datetime import datetime, timezone
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


def episode_id(source: str, content: str) -> str:
    """
    Deterministic episode ID from source + content hash.

    An *episode* is one unit of ingestion (here: one document/passage per
    pipeline run). Hashing the content makes re-ingesting the same source
    idempotent -- same bytes always produce the same `ep:` id.
    """
    key = f"{source}|{content}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"ep:{h}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GraphEntity(BaseModel):
    """A node in the knowledge graph."""
    entity_id: str
    name: str
    original_names: list[str] = Field(default_factory=list)
    node_type: Optional[str] = None  # W3B: one of domain_schema.NODE_TYPES or None


class Episode(BaseModel):
    """
    A unit of ingestion (one document/passage per pipeline run).

    Episodes provide provenance: every entity/relation produced by an episode
    links back to it via a :MENTIONS edge in Neo4j. The episode also carries
    the event time (`valid_at`) used to stamp the facts it asserts.
    """
    episode_id: str
    source: str                 # file path / document id this episode came from
    content: str                # the raw text ingested in this episode
    valid_at: datetime          # event time -- when the facts became true in the world
    created_at: datetime        # transaction time -- when we ingested this episode


class GraphRelation(BaseModel):
    """An edge in the knowledge graph."""
    triple_id: str
    head_entity_id: str
    tail_entity_id: str
    relation: str
    evidence: str
    relation_type: Optional[str] = None  # W3B: one of domain_schema.RELATION_TYPES or None

    # Bi-temporal + provenance (Graphiti-style). All optional so pre-existing
    # triple JSONs and tests stay valid.
    episode_ids: list[str] = Field(default_factory=list)  # episodes that asserted this fact
    valid_at: Optional[datetime] = None                   # when the fact became true (event time)
    invalid_at: Optional[datetime] = None                 # when it stopped being true; None = still believed
    created_at: datetime = Field(default_factory=_utcnow)  # when we ingested it (transaction time)
    expired_at: Optional[datetime] = None                 # when we retracted it; None = not retracted


def build_episode(source: str, content: str, valid_at: Optional[datetime] = None) -> Episode:
    """Construct an Episode for a document/passage. `valid_at` defaults to ingest time."""
    now = _utcnow()
    return Episode(
        episode_id=episode_id(source, content),
        source=source,
        content=content,
        valid_at=valid_at or now,
        created_at=now,
    )


def build_graph_objects(
    triples: list[dict],
    canonicalize: bool | None = None,
    episode: Optional[Episode] = None,
) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """
    Convert raw extraction triples into structured GraphEntity and GraphRelation objects.
    Handles deduplication of entities and collects all surface forms.

    If `canonicalize` is True (or env var KG_CANONICALIZE=1), runs the W3A
    rule + embedding hybrid dedup pass on the result. Default behavior is
    controlled by KG_CANONICALIZE so tests can opt-out without code changes.

    If `episode` is provided, each relation is stamped with the episode's id
    (provenance) and `valid_at`/`created_at` timestamps (Graphiti-style
    bi-temporal model). When omitted, relations get no provenance and a
    `created_at` defaulted to now -- keeping older callers unchanged.
    """
    import os

    entity_map: dict[str, GraphEntity] = {}
    relations: list[GraphRelation] = []

    rel_temporal: dict = {}
    if episode is not None:
        rel_temporal = {
            "episode_ids": [episode.episode_id],
            "valid_at": episode.valid_at,
            "created_at": episode.created_at,
        }

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
            **rel_temporal,
        ))

    entities = sorted(entity_map.values(), key=lambda e: e.name)

    if canonicalize is None:
        canonicalize = os.getenv("KG_CANONICALIZE", "0") == "1"
    if canonicalize and entities:
        from canonicalize import apply_canonicalization

        entities, relations = apply_canonicalization(entities, relations)

    return entities, relations
