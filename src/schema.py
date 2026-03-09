"""
schema.py -- Pydantic models for validating extraction output.

Provides Triple and ExtractionResult models to ensure output quality
and catch malformed triples before they enter the knowledge graph.
"""

from pydantic import BaseModel, Field, field_validator


class Triple(BaseModel):
    head: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    tail: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=1)

    @field_validator("relation")
    @classmethod
    def relation_is_concise(cls, v):
        if len(v.strip().split()) > 5:
            raise ValueError(f"Relation too long: {v}")
        return v.strip()


class ExtractionResult(BaseModel):
    model: str
    num_triples: int
    num_entities: int
    entities: list[str]
    triples: list[Triple]

    def validate_counts(self):
        assert self.num_triples == len(self.triples)
        assert self.num_entities == len(self.entities)
