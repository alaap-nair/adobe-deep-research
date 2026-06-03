"""
schema.py -- Pydantic models for validating extraction output.

Provides Triple and ExtractionResult models to ensure output quality
and catch malformed triples before they enter the knowledge graph.

W3B: head_type, tail_type, relation_type are optional to preserve backwards
compatibility with pre-W3B triple JSONs, but if provided they must belong to
the allowed sets in src/domain_schema.py.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from domain_schema import NODE_TYPES, RELATION_TYPES


class Triple(BaseModel):
    head: str = Field(..., min_length=1)
    head_type: Optional[str] = None
    relation: str = Field(..., min_length=1)
    relation_type: Optional[str] = None
    tail: str = Field(..., min_length=1)
    tail_type: Optional[str] = None
    evidence: str = Field(..., min_length=1)

    @field_validator("relation")
    @classmethod
    def relation_is_concise(cls, v):
        if len(v.strip().split()) > 5:
            raise ValueError(f"Relation too long: {v}")
        return v.strip()

    @field_validator("head_type", "tail_type")
    @classmethod
    def valid_node_type(cls, v):
        if v is None:
            return v
        v = v.strip()
        if v not in NODE_TYPES:
            raise ValueError(
                f"Unknown node type {v!r}. Allowed: {sorted(NODE_TYPES)}"
            )
        return v

    @field_validator("relation_type")
    @classmethod
    def valid_relation_type(cls, v):
        if v is None:
            return v
        v = v.strip()
        if v not in RELATION_TYPES:
            raise ValueError(
                f"Unknown relation type {v!r}. Allowed: {sorted(RELATION_TYPES)}"
            )
        return v


class ExtractionResult(BaseModel):
    model: str
    num_triples: int
    num_entities: int
    entities: list[str]
    triples: list[Triple]

    def validate_counts(self):
        assert self.num_triples == len(self.triples)
        assert self.num_entities == len(self.entities)
