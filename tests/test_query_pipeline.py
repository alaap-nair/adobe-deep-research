"""
test_query_pipeline.py -- QA-focused tests for the hybrid question-answer flow.

These tests are written against the planned public interfaces so they can
exercise retrieval, context assembly, and CLI behavior without depending on
live Neo4j/Qdrant/OpenRouter services.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_schema import build_graph_objects


class FakeQdrant:
    def __init__(self, entity_hits=None, evidence_hits=None, chunk_hits=None):
        self.entity_hits = entity_hits or []
        self.evidence_hits = evidence_hits or []
        self.chunk_hits = chunk_hits or []

    def query_points(self, collection_name, query, limit, with_payload=True):
        if collection_name == "entities":
            points = self.entity_hits[:limit]
        elif collection_name == "evidence":
            points = self.evidence_hits[:limit]
        else:
            points = self.chunk_hits[:limit]
        return SimpleNamespace(points=points)

    def close(self):
        return None


class FakeNeo4jSession:
    def __init__(self, neighborhood):
        self.neighborhood = neighborhood

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, entity_id=None):
        node_rows = []
        edge_rows = []
        for node in self.neighborhood.get(entity_id, {}).get("nodes", []):
            node_rows.append(node)
        for edge in self.neighborhood.get(entity_id, {}).get("edges", []):
            edge_rows.append(edge)
        return SimpleNamespace(single=lambda: {"nodes": node_rows, "edges": edge_rows})


class FakeNeo4jDriver:
    def __init__(self, neighborhood):
        self.neighborhood = neighborhood

    def session(self):
        return FakeNeo4jSession(self.neighborhood)

    def close(self):
        return None


def _make_point(payload, score=0.99):
    return SimpleNamespace(payload=payload, score=score)


def test_build_graph_objects_handles_glycolisis_fixture():
    triples = [
        {
            "head": "glycolysis",
            "relation": "produce",
            "tail": "ATP",
            "evidence": "Glycolysis produces a net gain of ATP.",
        }
    ]
    entities, relations = build_graph_objects(triples)
    assert any(entity.entity_id == "ent:glycolysis" for entity in entities)
    assert relations[0].triple_id.startswith("triple:")


def test_chunking_builds_stable_records():
    chunking = pytest.importorskip("chunking")

    text = (
        "Glycolysis occurs in the cytosol. It produces ATP and pyruvate.\n\n"
        "A second paragraph keeps the split paragraph-first."
    )
    chunks = chunking.build_chunk_records(text, source_name="biology_7_2.txt", max_chars=80, overlap=10)

    assert len(chunks) >= 2
    assert chunks[0]["chunk_id"].startswith("chunk:biology_7_2:")
    assert chunks[0]["source_name"] == "biology_7_2.txt"
    assert chunks[0]["chunk_index"] == 0
    assert "ATP" in chunks[0]["text"] or "ATP" in chunks[1]["text"]


def test_resolve_query_entities_recovers_misspelling(monkeypatch):
    query_engine = pytest.importorskip("query_engine")

    entity_payloads = [
        _make_point({"entity_id": "ent:glycolysis", "name": "glycolysis"}, 0.95),
        _make_point({"entity_id": "ent:atp", "name": "atp"}, 0.10),
    ]
    qdrant = FakeQdrant(entity_hits=entity_payloads)
    driver = FakeNeo4jDriver({})

    engine = query_engine.QueryEngine(neo4j_driver=driver, qdrant_client=qdrant, model=SimpleNamespace())
    monkeypatch.setattr(query_engine, "embed_texts", lambda texts, model=None: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(engine, "search_entities", lambda question, top_k=5: [
        {"entity_id": "ent:glycolysis", "name": "glycolysis", "score": 0.95}
    ])

    resolved = engine.resolve_query_entities("What does glycolisis produce")

    assert any(item["entity_id"] == "ent:glycolysis" for item in resolved)
    assert any(item["match_type"] == "fuzzy" for item in resolved)
    engine.close()


def test_retrieve_context_dedupes_and_caps(monkeypatch):
    query_engine = pytest.importorskip("query_engine")

    entity_hits = [
        {"entity_id": "ent:glycolysis", "name": "glycolysis", "score": 0.99},
        {"entity_id": "ent:atp", "name": "atp", "score": 0.88},
        {"entity_id": "ent:pyruvate", "name": "pyruvate", "score": 0.77},
        {"entity_id": "ent:oxygen", "name": "oxygen", "score": 0.66},
    ]
    chunk_hits = [
        {"chunk_id": "chunk-1", "text": "Glycolysis produces ATP.", "source_name": "passage", "chunk_index": 0, "score": 0.9},
        {"chunk_id": "chunk-1", "text": "Glycolysis produces ATP.", "source_name": "passage", "chunk_index": 0, "score": 0.8},
        {"chunk_id": "chunk-2", "text": "Glycolysis breaks down glucose.", "source_name": "passage", "chunk_index": 1, "score": 0.7},
        {"chunk_id": "chunk-3", "text": "ATP synthase uses the proton gradient.", "source_name": "passage", "chunk_index": 2, "score": 0.6},
        {"chunk_id": "chunk-4", "text": "Pyruvate enters the mitochondrion.", "source_name": "passage", "chunk_index": 3, "score": 0.5},
        {"chunk_id": "chunk-5", "text": "Additional supporting text.", "source_name": "passage", "chunk_index": 4, "score": 0.4},
        {"chunk_id": "chunk-6", "text": "Extra text beyond the cap.", "source_name": "passage", "chunk_index": 5, "score": 0.3},
    ]
    evidence_hits = [
        {"triple_id": "triple-1", "head_entity_id": "ent:glycolysis", "tail_entity_id": "ent:atp", "relation": "produce", "evidence": "Glycolysis produces a net gain of ATP.", "score": 0.91},
        {"triple_id": "triple-1", "head_entity_id": "ent:glycolysis", "tail_entity_id": "ent:atp", "relation": "produce", "evidence": "Glycolysis produces a net gain of ATP.", "score": 0.90},
    ]
    neighborhood = {
        "ent:glycolysis": {
            "nodes": [
                {"entity_id": "ent:glycolysis", "name": "glycolysis"},
                {"entity_id": "ent:atp", "name": "atp"},
            ],
            "edges": [
                {
                    "triple_id": "triple-1",
                    "source_entity_id": "ent:glycolysis",
                    "source_name": "glycolysis",
                    "relation": "produce",
                    "target_entity_id": "ent:atp",
                    "target_name": "atp",
                    "evidence": "Glycolysis produces a net gain of ATP.",
                }
            ],
        }
    }

    qdrant = FakeQdrant(entity_hits=entity_hits, evidence_hits=evidence_hits, chunk_hits=chunk_hits)
    driver = FakeNeo4jDriver(neighborhood)
    engine = query_engine.QueryEngine(neo4j_driver=driver, qdrant_client=qdrant, model=SimpleNamespace())

    monkeypatch.setattr(engine, "resolve_query_entities", lambda question: [
        {"entity_id": "ent:glycolysis", "name": "glycolysis", "match_type": "fuzzy", "score": 0.99},
        {"entity_id": "ent:atp", "name": "atp", "match_type": "semantic", "score": 0.88},
    ])
    monkeypatch.setattr(engine, "search_chunks", lambda question, top_k=5: chunk_hits)
    monkeypatch.setattr(engine, "search_evidence", lambda question, top_k=5: evidence_hits)

    context = engine.retrieve_context("What does glycolisis produce")

    assert set(context.keys()) >= {"query_analysis", "vector_hits", "graph_trace"}
    assert context["query_analysis"]["resolved_entities"][0]["match_type"] == "fuzzy"
    assert any(node["entity_id"] == "ent:glycolysis" for node in context["graph_trace"]["retrieved_nodes"])
    assert len(context["vector_hits"]["chunks"]) <= 5
    assert len(context["graph_trace"]["traversed_edges"]) <= 8
    assert len({chunk["chunk_id"] for chunk in context["vector_hits"]["chunks"]}) == len(context["vector_hits"]["chunks"])
    assert len({edge["triple_id"] for edge in context["graph_trace"]["traversed_edges"]}) == len(context["graph_trace"]["traversed_edges"])
    assert context["graph_trace"]["traversed_edges"][0]["source_name"] == "glycolysis"
    assert "ATP" in context["graph_trace"]["traversed_edges"][0]["evidence"]
    engine.close()


def test_ask_cli_writes_json_and_trace(monkeypatch, tmp_path, capsys):
    ask = pytest.importorskip("ask")

    output_dir = tmp_path / "outputs" / "answers"
    output_dir.mkdir(parents=True)

    qa_json = {
        "question": "What does glycolisis produce",
        "answer": "ATP",
        "citations": [
            "chunk:chunk-1 | passage | Glycolysis produces a net gain of ATP.",
            "triple:triple-1 | passage --[produce]--> ATP",
        ],
        "reasoning": "The context shows glycolysis produces ATP and the graph trace confirms the edge.",
    }
    trace_json = {"graph_trace": {"traversed_edges": [{"relation": "produce"}]}}

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def retrieve_context(self, question, **kwargs):
            return {
                "question": question,
                "query_analysis": {"keywords": ["glycolysis"], "resolved_entities": []},
                "vector_hits": {"chunks": [], "evidence": []},
                "graph_trace": trace_json["graph_trace"],
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr(ask, "QueryEngine", DummyEngine)
    monkeypatch.setattr(ask, "generate_answer", lambda question, context: qa_json)
    monkeypatch.setattr(ask, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(ask, "slugify_question", lambda question: "what-does-glycolisis-produce")
    monkeypatch.setattr(sys, "argv", ["ask.py", "What does glycolisis produce"])

    result = ask.main()
    stdout = capsys.readouterr().out.strip()
    assert result in (None, 0, qa_json)
    assert json.loads(stdout) == qa_json

    answer_path = output_dir / "what-does-glycolisis-produce.json"
    trace_path = output_dir / "what-does-glycolisis-produce_trace.json"
    assert answer_path.exists()
    assert trace_path.exists()

    with answer_path.open() as f:
        saved = json.load(f)
    assert saved == qa_json
