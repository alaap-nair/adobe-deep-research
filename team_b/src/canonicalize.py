"""
canonicalize.py -- Entity deduplication and canonicalization (W3A).

Two-stage hybrid:
  1. rule_normalize()    -- unicode fold, strip punctuation, lowercase, lemmatize.
                            Collapses surface noise like "Photosystems" / "photosystem"
                            and "Lichen,"/"Lichen" into the same key.
  2. cluster_entities()  -- BGE-large embeddings + greedy single-link clustering
                            at cosine >= threshold. Catches near-dupes that survive
                            rule normalization (e.g. "Photosystem II" vs "PSII").

Both stages are exposed individually for testing; apply_canonicalization()
wires them together and rewrites a (entities, relations) tuple in place.
"""

from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from collections import defaultdict
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_schema import GraphEntity, GraphRelation, entity_id, normalize_name, triple_id

# Plural-singular fallback: words we should NOT strip a trailing 's' from.
_PLURAL_KEEP = {
    "is", "as", "us", "gas", "mass", "class", "process", "synthesis",
    "photosynthesis", "lens", "series", "species", "analysis", "axis",
}

DEFAULT_CLUSTER_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# Rule-based normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s\-]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_lemmatizer = None


def _get_lemmatizer():
    """Lazy WordNet lemmatizer. Downloads corpus on first run if missing."""
    global _lemmatizer
    if _lemmatizer is not None:
        return _lemmatizer
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        return None
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


_GREEK_NOUN_SUFFIXES = ("sis", "us", "is", "ies")  # glycolysis, mycelius, axis, etc.


def _fallback_singular(token: str) -> str:
    """Cheap plural -> singular when WordNet is unavailable or unsure."""
    if len(token) <= 3 or token in _PLURAL_KEEP:
        return token
    # Preserve Greek/Latin noun endings that *look* like plurals
    if any(token.endswith(suf) for suf in _GREEK_NOUN_SUFFIXES):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def rule_normalize(name: str) -> str:
    """
    Produce a normalized clustering key from an entity name.

    Steps: unicode NFKC fold -> strip surrounding quotes/brackets ->
    drop punctuation (keep hyphens) -> lowercase -> collapse whitespace ->
    lemmatize each token (or fall back to plural-stripping).

    The output is intended for *grouping*, not display. The original surface
    form is preserved separately on GraphEntity.original_names.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name)
    s = s.strip().strip("\"'`()[]{}<>")
    s = _PUNCT_RE.sub(" ", s)
    s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s).strip()
    if not s:
        return ""
    tokens = s.split(" ")
    lemmatizer = _get_lemmatizer()
    out: list[str] = []
    for tok in tokens:
        if lemmatizer is not None:
            lemma = lemmatizer.lemmatize(tok, pos="n")
            # WordNet sometimes leaves common biology plurals alone; fall through.
            if lemma == tok:
                lemma = _fallback_singular(tok)
            out.append(lemma)
        else:
            out.append(_fallback_singular(tok))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Embedding-based clustering
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def cluster_entities(
    names: list[str],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    embedder=None,
) -> list[int]:
    """
    Greedy single-link clustering on BGE embeddings of `names`.

    Returns a list of cluster ids (one per input name); names that share a
    cluster id should be merged. Cluster id 0 corresponds to the first input,
    cluster id k is assigned to the (k+1)-th cluster discovered.

    Walks names in input order. For each name: if its embedding has cosine
    >= threshold to *any* prior centroid, join that cluster (first match
    wins); otherwise, open a new cluster and seed its centroid.
    """
    if not names:
        return []
    if embedder is None:
        from build_vectorstore import embed_texts

        vectors = embed_texts(names)
    else:
        vectors = embedder(names)

    cluster_ids: list[int] = []
    centroids: list[list[float]] = []
    for vec in vectors:
        joined = -1
        for idx, centroid in enumerate(centroids):
            if _cosine(vec, centroid) >= threshold:
                joined = idx
                break
        if joined == -1:
            cluster_ids.append(len(centroids))
            centroids.append(list(vec))
        else:
            cluster_ids.append(joined)
    return cluster_ids


# ---------------------------------------------------------------------------
# Top-level: rewrite entities + relations to canonical form
# ---------------------------------------------------------------------------


def _pick_canonical_name(entities: list[GraphEntity]) -> str:
    """Pick the canonical display name for a merged group: most surface forms wins."""
    return max(
        entities,
        key=lambda e: (len(e.original_names), -len(e.name), e.name),
    ).name


def _merge_entities(group: list[GraphEntity], canonical_name: str) -> GraphEntity:
    canonical_id = entity_id(canonical_name)
    seen: list[str] = []
    for ent in group:
        for surface in [ent.name, *ent.original_names]:
            if surface and surface not in seen:
                seen.append(surface)
    return GraphEntity(
        entity_id=canonical_id,
        name=normalize_name(canonical_name).replace("_", " "),
        original_names=seen,
    )


def apply_canonicalization(
    entities: list[GraphEntity],
    relations: list[GraphRelation],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    embedder=None,
) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """
    Collapse near-duplicate entities and rewrite relations to canonical IDs.

    Stage 1 -- group by rule_normalize() key (deterministic, no model needed).
    Stage 2 -- on the per-group canonical names, run greedy embedding clustering;
               clusters at cosine >= `threshold` are merged.

    Self-loop relations (head==tail after merge) are dropped. Triple IDs are
    recomputed from canonical names so duplicate edges collapse via dedup.
    """
    if not entities:
        return entities, relations

    # --- Stage 1: rule-based grouping ---
    rule_groups: dict[str, list[GraphEntity]] = defaultdict(list)
    for ent in entities:
        key = rule_normalize(ent.name) or ent.entity_id
        rule_groups[key].append(ent)

    rule_canonicals: list[GraphEntity] = []
    for key in sorted(rule_groups.keys()):
        group = rule_groups[key]
        canon_name = _pick_canonical_name(group)
        rule_canonicals.append(_merge_entities(group, canon_name))

    # --- Stage 2: embedding clustering across rule canonicals ---
    if len(rule_canonicals) > 1:
        cluster_ids = cluster_entities(
            [e.name for e in rule_canonicals],
            threshold=threshold,
            embedder=embedder,
        )
    else:
        cluster_ids = [0]

    cluster_groups: dict[int, list[GraphEntity]] = defaultdict(list)
    for cid, ent in zip(cluster_ids, rule_canonicals):
        cluster_groups[cid].append(ent)

    final_entities: list[GraphEntity] = []
    final_id_for_old: dict[str, str] = {}

    for cid in sorted(cluster_groups.keys()):
        group = cluster_groups[cid]
        canon_name = _pick_canonical_name(group)
        merged = _merge_entities(group, canon_name)
        final_entities.append(merged)
        for ent in group:
            final_id_for_old[ent.entity_id] = merged.entity_id

    # Walk back to original entities to fully populate id_remap
    id_remap: dict[str, str] = {}
    for original_ent in entities:
        rule_key = rule_normalize(original_ent.name) or original_ent.entity_id
        # find which rule_canonical this original landed in -> then which final cluster
        rule_canon = next(
            r
            for r in rule_canonicals
            if r.entity_id == entity_id(_pick_canonical_name(rule_groups[rule_key]))
        )
        id_remap[original_ent.entity_id] = final_id_for_old[rule_canon.entity_id]

    # --- Rewrite relations ---
    final_name_for_id = {e.entity_id: e.name for e in final_entities}
    seen_triple_ids: set[str] = set()
    final_relations: list[GraphRelation] = []
    for rel in relations:
        new_head = id_remap.get(rel.head_entity_id, rel.head_entity_id)
        new_tail = id_remap.get(rel.tail_entity_id, rel.tail_entity_id)
        if new_head == new_tail:
            continue  # drop self-loops introduced by merging
        head_name = final_name_for_id.get(new_head, new_head)
        tail_name = final_name_for_id.get(new_tail, new_tail)
        new_tid = triple_id(head_name, rel.relation, tail_name)
        if new_tid in seen_triple_ids:
            continue
        seen_triple_ids.add(new_tid)
        rebuilt = GraphRelation(
            triple_id=new_tid,
            head_entity_id=new_head,
            tail_entity_id=new_tail,
            relation=rel.relation,
            evidence=rel.evidence,
        )
        # Preserve W3B type fields and bi-temporal / provenance metadata so a
        # canonicalization pass never silently drops an edge's history.
        for attr in (
            "relation_type",
            "episode_ids",
            "valid_at",
            "invalid_at",
            "created_at",
            "expired_at",
        ):
            if hasattr(rel, attr):
                setattr(rebuilt, attr, getattr(rel, attr))
        final_relations.append(rebuilt)

    final_entities.sort(key=lambda e: e.name)
    return final_entities, final_relations


# ---------------------------------------------------------------------------
# Incremental resolution: new batch vs entities already in the graph (phase 4)
# ---------------------------------------------------------------------------

_TEMPORAL_ATTRS = (
    "relation_type",
    "episode_ids",
    "valid_at",
    "invalid_at",
    "created_at",
    "expired_at",
)


def resolve_entities_against(
    new_entities: list[GraphEntity],
    new_relations: list[GraphRelation],
    existing: list[dict],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    embedder=None,
) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """
    Resolve a freshly-extracted batch against entities already in the graph.

    `existing` is a list of {entity_id, name, original_names} dicts (e.g. loaded
    from Neo4j). A new entity is matched to an existing one when their
    rule_normalize() keys agree, or -- if an `embedder` is supplied -- when their
    names embed at cosine >= `threshold`. Matched entities adopt the *existing*
    canonical id and union their surface forms, so incremental ingests attach to
    established nodes instead of forking near-duplicates (e.g. "PSII" landing on
    a prior "photosystem II"). Relations are rewritten to the resolved ids with
    triple_ids recomputed; temporal/provenance metadata is preserved.

    Pure function (no DB access) so it can be unit-tested offline.
    """
    if not new_entities:
        return new_entities, new_relations

    existing_by_id = {ex.get("entity_id"): ex for ex in existing}
    existing_by_key: dict[str, dict] = {}
    for ex in existing:
        key = rule_normalize(ex.get("name", "")) or ex.get("entity_id")
        existing_by_key.setdefault(key, ex)
    existing_names = [ex.get("name", "") for ex in existing]
    existing_vecs = None  # embedded lazily on first miss

    id_remap: dict[str, str] = {}
    name_for_id: dict[str, str] = {}
    surfaces_for_id: dict[str, list[str]] = defaultdict(list)
    type_for_id: dict[str, str | None] = {}

    def _register(rid: str, name: str, surfaces: list[str], node_type):
        name_for_id.setdefault(rid, name)
        bucket = surfaces_for_id[rid]
        for s in surfaces:
            if s and s not in bucket:
                bucket.append(s)
        if type_for_id.get(rid) is None and node_type:
            type_for_id[rid] = node_type

    for ent in new_entities:
        key = rule_normalize(ent.name) or ent.entity_id
        match = existing_by_key.get(key)
        if match is None and embedder is not None and existing_names:
            if existing_vecs is None:
                existing_vecs = embedder(existing_names)
            new_vec = embedder([ent.name])[0]
            best_i, best_score = -1, 0.0
            for i, ev in enumerate(existing_vecs):
                score = _cosine(new_vec, ev)
                if score > best_score:
                    best_score, best_i = score, i
            if best_i >= 0 and best_score >= threshold:
                match = existing[best_i]

        if match is not None:
            rid = match["entity_id"]
            id_remap[ent.entity_id] = rid
            prior = existing_by_id.get(rid, {}).get("original_names") or []
            _register(
                rid,
                match.get("name") or ent.name,
                [*prior, match.get("name"), *ent.original_names, ent.name],
                ent.node_type,
            )
        else:
            id_remap[ent.entity_id] = ent.entity_id
            _register(ent.entity_id, ent.name, [*ent.original_names, ent.name], ent.node_type)

    resolved_entities = [
        GraphEntity(
            entity_id=rid,
            name=name_for_id[rid],
            original_names=[s for s in surfaces_for_id[rid] if s],
            node_type=type_for_id.get(rid),
        )
        for rid in dict.fromkeys(id_remap.values())
    ]

    final_name_for_id = {e.entity_id: e.name for e in resolved_entities}
    seen_triple_ids: set[str] = set()
    resolved_relations: list[GraphRelation] = []
    for rel in new_relations:
        new_head = id_remap.get(rel.head_entity_id, rel.head_entity_id)
        new_tail = id_remap.get(rel.tail_entity_id, rel.tail_entity_id)
        if new_head == new_tail:
            continue
        head_name = final_name_for_id.get(new_head, new_head)
        tail_name = final_name_for_id.get(new_tail, new_tail)
        new_tid = triple_id(head_name, rel.relation, tail_name)
        if new_tid in seen_triple_ids:
            continue
        seen_triple_ids.add(new_tid)
        rebuilt = GraphRelation(
            triple_id=new_tid,
            head_entity_id=new_head,
            tail_entity_id=new_tail,
            relation=rel.relation,
            evidence=rel.evidence,
        )
        for attr in _TEMPORAL_ATTRS:
            if hasattr(rel, attr):
                setattr(rebuilt, attr, getattr(rel, attr))
        resolved_relations.append(rebuilt)

    return resolved_entities, resolved_relations


def _load_existing_entities(driver) -> list[dict]:
    """Load {entity_id, name, original_names} for every :Entity already stored."""
    with driver.session() as session:
        result = session.run(
            "MATCH (n:Entity) RETURN n.entity_id AS entity_id, n.name AS name, "
            "n.original_names AS original_names"
        )
        return [
            {
                "entity_id": r["entity_id"],
                "name": r["name"] or "",
                "original_names": r["original_names"] or [],
            }
            for r in result
        ]


def resolve_against_graph(
    driver,
    entities: list[GraphEntity],
    relations: list[GraphRelation],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    embedder=None,
) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """Load existing entities from Neo4j and resolve `entities`/`relations`
    against them. Defaults to the BGE embedder used elsewhere; falls back to
    rule-only matching if embeddings are unavailable."""
    existing = _load_existing_entities(driver)
    if not existing:
        return entities, relations
    if embedder is None:
        try:
            from build_vectorstore import embed_texts

            embedder = lambda names: embed_texts(names)  # noqa: E731
        except Exception:
            embedder = None
    return resolve_entities_against(entities, relations, existing, threshold, embedder)


__all__ = [
    "apply_canonicalization",
    "cluster_entities",
    "rule_normalize",
    "resolve_entities_against",
    "resolve_against_graph",
    "DEFAULT_CLUSTER_THRESHOLD",
]
