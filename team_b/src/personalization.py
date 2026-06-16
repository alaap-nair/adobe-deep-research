"""
personalization.py -- Per-user graph context injection seam (Graphiti).

This is the single integration point for the optional "personalized answer"
mode in the chat UI. The *generalized* path runs the standard dual-store KAG
pipeline untouched; the *personalized* path runs the same real retrieval and
then passes the retrieved context through `inject_user_context()` before
synthesis, so user-specific facts can be merged into the graph trace.

STATUS: stub. The real implementation (A10 R&D) stands up a Graphiti per-user
episodic graph, queries it for facts relevant to the question, and merges those
facts into `context["graph_trace"]` (and optionally a `user_facts` block the
prompt builder can surface). Until that lands, this returns the context
unchanged plus a clearly-labeled note the UI renders as a banner -- so the
toggle is demonstrably live and the answer path is never faked.
"""

from typing import Any

STUB_NOTE = (
    "Personalized mode (preview): retrieval is real, but the Graphiti per-user "
    "graph is not yet wired in. Answer below is the standard pipeline output. "
    "(A10 R&D — see personalization.py)"
)


def inject_user_context(
    context: dict[str, Any], user_id: str | None = None
) -> tuple[dict[str, Any], str | None]:
    """Merge per-user graph context into the retrieval context.

    Args:
        context: the dict returned by ``QueryEngine.retrieve_context()``.
        user_id: identifier for the user whose graph to query (unused in stub).

    Returns:
        (context, note). ``note`` is a human-readable string shown in the UI
        when personalization is active, or ``None`` when no note applies.

    TODO(A10): replace the stub body with:
        facts = graphiti.search(user_id, query=context["query_analysis"]...)
        context["graph_trace"]["retrieved_nodes"] += facts.nodes
        context["graph_trace"]["traversed_edges"] += facts.edges
        context["user_facts"] = facts.summaries
    """
    context.setdefault("_personalization", {})
    context["_personalization"] = {
        "status": "stub",
        "user_id": user_id,
        "note": STUB_NOTE,
    }
    return context, STUB_NOTE
