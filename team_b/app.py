"""
app.py -- Simple chat UI for the dual-store KAG system (Team B).

Run:
    streamlit run app.py

Serves the Assignment-10 demo format over the benchmark-winning retrieval
config (A10 Part 2: bge-reranker-v2-m3). For each question it shows the final
answer, citations/evidence, the retrieved chunks, and the graph nodes/edges
that grounded the answer. A Generalized/Personalized toggle routes retrieval
through the Graphiti personalization seam (currently a labeled stub).
"""

import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st

from ui_backend import ask, build_engine, check_resources

try:
    from streamlit_agraph import agraph, Node, Edge, Config

    HAS_AGRAPH = True
except Exception:  # pragma: no cover - optional dep
    HAS_AGRAPH = False


EXAMPLE_QUESTIONS = [
    "What does glycolysis produce?",
    "Where does glycolysis occur?",
    "What is the role of ATP synthase?",
    "What does the electron transport chain create?",
    "What happens to pyruvate after glycolysis?",
]


# --- cached resources ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding + reranker models…")
def get_engine():
    """One QueryEngine for the whole session (loads the BGE embedder once)."""
    return build_engine()


@st.cache_resource(show_spinner=False)
def get_health(_engine):
    return check_resources(_engine)


# --- rendering helpers --------------------------------------------------------

def render_graph(nodes, edges):
    """Draw the grounding subgraph, with a text fallback if agraph is absent."""
    if not nodes and not edges:
        st.info("No graph nodes were traversed for this question "
                "(Neo4j may be empty or unreachable).")
        return

    if HAS_AGRAPH and nodes:
        agraph_nodes = [
            Node(id=n["entity_id"], label=n.get("name", n["entity_id"]), size=18)
            for n in nodes
        ]
        known = {n["entity_id"] for n in nodes}
        agraph_edges = []
        for e in edges:
            s, t = e.get("source_entity_id"), e.get("target_entity_id")
            if s in known and t in known:
                agraph_edges.append(Edge(source=s, target=t, label=e.get("relation", "")))
        cfg = Config(width="100%", height=420, directed=True,
                     physics=True, hierarchical=False,
                     nodeHighlightBehavior=True, collapsible=False)
        agraph(nodes=agraph_nodes, edges=agraph_edges, config=cfg)

    # Always show the edge list too — readable and demo-proof.
    with st.expander(f"Graph edges ({len(edges)})", expanded=not HAS_AGRAPH):
        for e in edges:
            st.markdown(
                f"- **{e.get('source_name', e.get('source_entity_id'))}** "
                f"—[{e.get('relation','')}]→ "
                f"**{e.get('target_name', e.get('target_entity_id'))}**"
            )


def render_result(res: dict):
    if res.get("personalization_note"):
        st.warning(res["personalization_note"], icon="👤")

    st.markdown("#### Answer")
    st.markdown(f"### {res['answer']}")
    tv = res.get("temporal_view", {})
    if tv.get("include_invalid"):
        temporal_str = "including invalidated facts"
    elif tv.get("as_of"):
        temporal_str = f"as of {tv['as_of']}"
    else:
        temporal_str = "current facts"
    st.caption(
        f"mode: **{res['mode']}**  ·  retrieval: {res['retrieval_config']}  "
        f"·  view: {temporal_str}"
    )

    # Citations
    st.markdown("#### Citations / Evidence")
    if res["citations"]:
        for i, c in enumerate(res["citations"], 1):
            st.markdown(f"`[{i}]` {c}")
    else:
        st.caption("No citations returned for this answer.")

    col_graph, col_chunks = st.columns([1, 1])

    with col_graph:
        st.markdown("#### Graph (nodes / edges)")
        render_graph(res["nodes"], res["edges"])

    with col_chunks:
        st.markdown(f"#### Retrieved chunks ({len(res['chunks'])})")
        if not res["chunks"]:
            st.caption("No chunks retrieved.")
        for ch in res["chunks"]:
            score = ch.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            with st.container(border=True):
                st.caption(
                    f"`{ch.get('chunk_id','?')}`  ·  {ch.get('source_name','?')}  "
                    f"·  score {score_str}"
                )
                st.write(ch.get("snippet") or ch.get("text", ""))

    with st.expander(f"Evidence triples ({len(res['evidence'])})"):
        for ev in res["evidence"]:
            st.markdown(
                f"- `{ev.get('triple_id','?')}` — {ev.get('evidence','')}"
            )

    with st.expander("Reasoning trace"):
        st.write(res["reasoning"])


# --- app ----------------------------------------------------------------------

def main():
    st.set_page_config(page_title="KAG Chat — Dual-Store", page_icon="🧬", layout="wide")
    st.title("🧬 KAG Chat")
    st.caption(
        "Dual-store knowledge-augmented generation · Neo4j graph + Qdrant vectors · "
        "bge-reranker-v2-m3 (Assignment 10 Part 2 winner)"
    )

    engine = get_engine()
    health = get_health(engine)

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Answer mode",
            ["Generalized", "Personalized"],
            help="Generalized = standard KAG pipeline. "
                 "Personalized = same retrieval routed through the Graphiti "
                 "per-user graph seam (preview/stub).",
        )
        user_id = None
        if mode == "Personalized":
            user_id = st.text_input("User ID", value="demo-user")
        use_reranker = st.toggle(
            "Use bge-reranker (winning config)", value=True,
            help="The A10 Part-2 winner. Turn off to compare against the "
                 "no-rerank baseline.",
        )

        st.divider()
        st.subheader("Temporal view")
        st.caption(
            "The graph trace defaults to facts currently believed true. "
            "These options query the bi-temporal graph differently."
        )
        include_invalid = st.toggle(
            "Include invalidated facts", value=False,
            help="Include edges that were later contradicted/retracted "
                 "(invalid_at set). Off = current truth only.",
        )
        as_of = None
        if st.checkbox("Query as of a past date", value=False,
                       disabled=include_invalid,
                       help="Point-in-time view: show facts that were valid on "
                            "the chosen date. Ignored when 'Include invalidated' is on."):
            as_of_date = st.date_input("As of date")
            as_of = datetime(as_of_date.year, as_of_date.month, as_of_date.day,
                             tzinfo=timezone.utc)

        st.divider()
        st.caption("Service status")
        st.write(f"Qdrant: {'🟢' if health['qdrant'] else '🔴'}  ·  "
                 f"Neo4j: {'🟢' if health['neo4j'] else '🔴'}")
        for w in health["warnings"]:
            st.caption(f"⚠️ {w}")

    # Question input
    example = st.selectbox(
        "Example questions", ["—"] + EXAMPLE_QUESTIONS, index=0,
    )
    default_q = "" if example == "—" else example
    question = st.text_input(
        "Ask a question", value=default_q,
        placeholder="e.g. What does glycolysis produce?",
    )
    submit = st.button("Ask", type="primary")

    if submit:
        if not question.strip():
            st.error("Please enter a question.")
            return
        try:
            with st.spinner("Retrieving + synthesizing…"):
                res = ask(
                    question,
                    engine,
                    personalized=(mode == "Personalized"),
                    user_id=user_id,
                    use_reranker=use_reranker,
                    as_of=as_of,
                    include_invalid=include_invalid,
                )
            render_result(res)
        except Exception as exc:
            st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
