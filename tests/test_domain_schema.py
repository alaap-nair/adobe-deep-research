"""Tests for the W3B Triple type validators and domain schema."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from domain_schema import NODE_TYPES, RELATION_TYPES
from schema import Triple


class TestTripleTypes:
    def test_accepts_valid_types(self):
        t = Triple(
            head="RuBisCO",
            head_type="Enzyme",
            relation="catalyzes",
            relation_type="CATALYZES",
            tail="carbon fixation",
            tail_type="Process",
            evidence="RuBisCO catalyzes carbon fixation.",
        )
        assert t.head_type == "Enzyme"
        assert t.relation_type == "CATALYZES"

    def test_rejects_invalid_node_type(self):
        with pytest.raises(ValueError, match="Unknown node type"):
            Triple(
                head="X",
                head_type="GobbledygookType",
                relation="r",
                tail="Y",
                tail_type="Molecule",
                evidence="...",
            )

    def test_rejects_invalid_relation_type(self):
        with pytest.raises(ValueError, match="Unknown relation type"):
            Triple(
                head="X",
                head_type="Molecule",
                relation="r",
                relation_type="ZAPS",
                tail="Y",
                tail_type="Molecule",
                evidence="...",
            )

    def test_backwards_compat_accepts_no_types(self):
        # Pre-W3B triples without type fields still validate
        t = Triple(head="A", relation="r", tail="B", evidence="x")
        assert t.head_type is None
        assert t.tail_type is None
        assert t.relation_type is None


class TestSchemaContent:
    def test_node_types_nonempty(self):
        assert "Molecule" in NODE_TYPES
        assert "Enzyme" in NODE_TYPES
        assert "Process" in NODE_TYPES

    def test_relation_types_nonempty(self):
        assert "CATALYZES" in RELATION_TYPES
        assert "PRODUCES" in RELATION_TYPES
        assert "MUTUALISTIC_WITH" in RELATION_TYPES
