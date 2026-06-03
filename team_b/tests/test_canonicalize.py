"""Tests for the W3A canonicalization pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from canonicalize import (
    apply_canonicalization,
    cluster_entities,
    rule_normalize,
)
from graph_schema import GraphEntity, GraphRelation, build_graph_objects


class TestRuleNormalize:
    def test_lowercase(self):
        assert rule_normalize("Lichen") == "lichen"

    def test_strip_punctuation_keeps_hyphen(self):
        assert rule_normalize("Light-dependent reactions.") == "light-dependent reaction"

    def test_collapses_whitespace(self):
        assert rule_normalize("  Calvin   cycle  ") == "calvin cycle"

    def test_unicode_fold(self):
        # NFKC fold turns superscript-2 into "2"
        assert rule_normalize("CO²") == "co2"

    def test_plural_singular(self):
        assert rule_normalize("Lichens") == rule_normalize("Lichen")
        assert rule_normalize("Photosystems") == rule_normalize("Photosystem")

    def test_empty(self):
        assert rule_normalize("") == ""
        assert rule_normalize("...") == ""


class TestClusterEntities:
    def test_identical_strings_cluster(self):
        # Predictable embedder: identical names share clusters
        def embed(names):
            mapping = {n: [1.0 if c == n else 0.0 for c in names] for n in names}
            return [mapping[n] for n in names]

        ids = cluster_entities(["a", "b", "a"], threshold=0.99, embedder=embed)
        assert ids[0] == ids[2]
        assert ids[1] != ids[0]

    def test_threshold_separates(self):
        # Two orthogonal vectors should not cluster
        def embed(names):
            return [[1.0, 0.0], [0.0, 1.0]]

        ids = cluster_entities(["x", "y"], threshold=0.5, embedder=embed)
        assert ids[0] != ids[1]

    def test_high_similarity_clusters(self):
        def embed(names):
            return [[1.0, 0.0], [0.99, 0.14]]  # cos ~= 0.99

        ids = cluster_entities(["x", "y"], threshold=0.95, embedder=embed)
        assert ids[0] == ids[1]


class TestApplyCanonicalization:
    def _entities_relations(self, triples):
        return build_graph_objects(triples, canonicalize=False)

    def test_collapses_singular_plural(self):
        triples = [
            {"head": "Lichen", "relation": "consists of", "tail": "fungus", "evidence": "..."},
            {"head": "Lichens", "relation": "live in", "tail": "soil", "evidence": "..."},
        ]
        ents, rels = self._entities_relations(triples)
        assert len({e.entity_id for e in ents}) == 4  # lichen, lichens, fungus, soil

        # Mock embedder so embedding stage is a no-op (every name unique cluster)
        def embed(names):
            # orthogonal one-hot vectors so no two names cluster
            n = len(names)
            return [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]

        ents2, rels2 = apply_canonicalization(ents, rels, threshold=0.99, embedder=embed)
        names = {e.name for e in ents2}
        # "lichen" / "lichens" should collapse via rule_normalize
        assert sum(1 for n in names if "lichen" in n) == 1
        # Both edges from lichen should survive
        assert len(rels2) == 2

    def test_drops_self_loops(self):
        triples = [
            {"head": "Lichens", "relation": "are", "tail": "Lichen", "evidence": "..."},
            {"head": "fungus", "relation": "decomposes", "tail": "wood", "evidence": "..."},
        ]
        ents, rels = self._entities_relations(triples)

        def embed(names):
            # orthogonal one-hot vectors so no two names cluster
            n = len(names)
            return [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]

        ents2, rels2 = apply_canonicalization(ents, rels, threshold=0.99, embedder=embed)
        # First triple should drop (head==tail after merge)
        assert len(rels2) == 1
        assert rels2[0].relation == "decomposes"

    def test_preserves_node_type(self):
        triples = [
            {
                "head": "Lichen",
                "head_type": "Organism",
                "relation": "consists of",
                "relation_type": "CONSISTS_OF",
                "tail": "fungus",
                "tail_type": "Organism",
                "evidence": "...",
            }
        ]
        ents, rels = self._entities_relations(triples)
        assert all(e.node_type == "Organism" for e in ents)
        assert rels[0].relation_type == "CONSISTS_OF"

        def embed(names):
            # orthogonal one-hot vectors so no two names cluster
            n = len(names)
            return [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]

        ents2, rels2 = apply_canonicalization(ents, rels, threshold=0.99, embedder=embed)
        assert rels2[0].relation_type == "CONSISTS_OF"
