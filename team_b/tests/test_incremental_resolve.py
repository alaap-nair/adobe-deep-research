"""
test_incremental_resolve.py -- Phase-4 incremental entity resolution.

Unit tests for resolve_entities_against (pure, no DB) covering rule-key matches,
embedding matches with a stub embedder, surface-form unions, and relation
rewriting. Plus a live Neo4j test that a second ingest attaches to the existing
node rather than forking a near-duplicate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_schema import build_graph_objects, build_episode, GraphEntity, GraphRelation
from canonicalize import resolve_entities_against


def _existing(entity_id, name, original_names=None):
    return {"entity_id": entity_id, "name": name, "original_names": original_names or [name]}


class TestResolveEntitiesAgainst:
    def test_rule_key_match_adopts_existing_id(self):
        existing = [_existing("ent:glycolysis", "glycolysis", ["Glycolysis"])]
        new_ents = [
            GraphEntity(entity_id="ent:glycolysis", name="glycolysis", original_names=["GLYCOLYSIS"]),
        ]
        resolved, _ = resolve_entities_against(new_ents, [], existing)
        assert [e.entity_id for e in resolved] == ["ent:glycolysis"]
        # Surface forms from both the prior node and the new batch are unioned
        assert "Glycolysis" in resolved[0].original_names
        assert "GLYCOLYSIS" in resolved[0].original_names

    def test_no_match_keeps_new_entity(self):
        existing = [_existing("ent:glycolysis", "glycolysis")]
        new_ents = [GraphEntity(entity_id="ent:telomere", name="telomere", original_names=["telomere"])]
        resolved, _ = resolve_entities_against(new_ents, [], existing)
        assert [e.entity_id for e in resolved] == ["ent:telomere"]

    def test_embedding_match_merges_near_duplicate(self):
        existing = [_existing("ent:photosystem_ii", "photosystem ii")]
        new_ents = [GraphEntity(entity_id="ent:psii", name="psii", original_names=["PSII"])]

        # Stub embedder: identical vector for both names -> cosine 1.0 -> merge.
        def stub(names):
            return [[1.0, 0.0] for _ in names]

        resolved, _ = resolve_entities_against(new_ents, [], existing, embedder=stub)
        assert [e.entity_id for e in resolved] == ["ent:photosystem_ii"]
        assert "PSII" in resolved[0].original_names

    def test_embedding_below_threshold_does_not_merge(self):
        existing = [_existing("ent:photosystem_ii", "photosystem ii")]
        new_ents = [GraphEntity(entity_id="ent:psii", name="psii", original_names=["PSII"])]

        def stub(names):
            # Orthogonal vectors -> cosine 0 -> no merge.
            return [[1.0, 0.0] if n == "photosystem ii" else [0.0, 1.0] for n in names]

        resolved, _ = resolve_entities_against(new_ents, [], existing, embedder=stub)
        assert [e.entity_id for e in resolved] == ["ent:psii"]

    def test_relations_rewritten_and_temporal_preserved(self):
        existing = [_existing("ent:glycolysis", "glycolysis")]
        ep = build_episode("doc.txt", "glycolysis produces atp")
        entities, relations = build_graph_objects(
            [{"head": "Glycolysis", "relation": "produce", "tail": "ATP", "evidence": "..."}],
            episode=ep,
        )
        resolved_e, resolved_r = resolve_entities_against(entities, relations, existing)
        ids = {e.entity_id for e in resolved_e}
        assert "ent:glycolysis" in ids
        assert len(resolved_r) == 1
        rel = resolved_r[0]
        assert rel.head_entity_id == "ent:glycolysis"
        assert rel.episode_ids == [ep.episode_id]      # provenance survives
        assert rel.created_at == ep.created_at


# --- Live Neo4j test ---

try:
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from build_graph import get_driver, create_constraints, clear_graph, build_graph, get_graph_stats

    _d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _d.verify_connectivity()
    _d.close()
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False


@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
def test_incremental_resolve_attaches_to_existing_node():
    driver = get_driver()
    clear_graph(driver)
    create_constraints(driver)
    try:
        ep1 = build_episode("v1.txt", "glycolysis text")
        build_graph(
            [{"head": "Glycolysis", "relation": "produce", "tail": "ATP", "evidence": "..."}],
            driver, episode=ep1,
        )
        before = get_graph_stats(driver)["nodes"]

        # Second ingest references the same concept with a different surface
        # form; rule normalization should fold it onto the existing node.
        ep2 = build_episode("v2.txt", "glycolysis text 2")
        build_graph(
            [{"head": "GLYCOLYSIS", "relation": "break down", "tail": "glucose", "evidence": "..."}],
            driver, episode=ep2, incremental_resolve=True,
        )
        after = get_graph_stats(driver)["nodes"]

        # glucose is genuinely new (+1); glycolysis must NOT have forked.
        assert after == before + 1
        with driver.session() as s:
            cnt = s.run(
                "MATCH (n:Entity {entity_id: 'ent:glycolysis'}) RETURN count(n) AS c"
            ).single()["c"]
        assert cnt == 1
    finally:
        clear_graph(driver)
        driver.close()
