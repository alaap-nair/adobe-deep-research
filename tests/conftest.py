"""
conftest.py -- Shared test fixtures for the knowledge graph engine tests.
"""

import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SAMPLE_TRIPLES = [
    {
        "head": "glycolysis",
        "relation": "occur in",
        "tail": "cytosol",
        "evidence": "It occurs in the cytosol and does not require oxygen.",
    },
    {
        "head": "glycolysis",
        "relation": "produce",
        "tail": "ATP",
        "evidence": "Glycolysis produces a net gain of ATP.",
    },
    {
        "head": "glycolysis",
        "relation": "break down",
        "tail": "glucose",
        "evidence": "one molecule of glucose is broken down into two molecules of pyruvate",
    },
    {
        "head": "pyruvate",
        "relation": "transport into",
        "tail": "mitochondrion",
        "evidence": "pyruvate is transported into the mitochondrion",
    },
    {
        "head": "electron transport chain",
        "relation": "located in",
        "tail": "inner mitochondrial membrane",
        "evidence": "The electron transport chain (ETC) is located in the inner mitochondrial membrane",
    },
]


@pytest.fixture
def sample_triples():
    return SAMPLE_TRIPLES


@pytest.fixture
def sample_graph_objects():
    from graph_schema import build_graph_objects
    return build_graph_objects(SAMPLE_TRIPLES)


@pytest.fixture
def qdrant_memory_client():
    from qdrant_client import QdrantClient
    client = QdrantClient(location=":memory:")
    yield client
    client.close()
