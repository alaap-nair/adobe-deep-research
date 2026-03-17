"""
visualize_graph.py -- Interactive knowledge graph visualization.

Generates a self-contained HTML file using pyvis that can be opened
in any browser. Nodes are colored by degree centrality, edges are
labeled with the relation, and hovering shows the evidence text.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from pyvis.network import Network
from graph_schema import build_graph_objects, GraphEntity, GraphRelation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color palette: low degree -> cool, high degree -> warm
_COLORS = ["#4FC3F7", "#4DD0E1", "#4DB6AC", "#81C784", "#AED581",
            "#DCE775", "#FFD54F", "#FFB74D", "#FF8A65", "#E57373"]


def build_networkx_graph(
    entities: list[GraphEntity],
    relations: list[GraphRelation],
) -> nx.DiGraph:
    """Create a NetworkX directed graph from structured graph objects."""
    G = nx.DiGraph()
    for e in entities:
        G.add_node(e.entity_id, label=e.name, original_names=e.original_names)
    for r in relations:
        G.add_edge(
            r.head_entity_id,
            r.tail_entity_id,
            triple_id=r.triple_id,
            relation=r.relation,
            evidence=r.evidence,
        )
    return G


def visualize_pyvis(
    G: nx.DiGraph,
    output_path: str,
    title: str = "Knowledge Graph",
):
    """Convert a NetworkX graph into an interactive pyvis HTML visualization."""
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
        heading=title,
    )

    # Compute degree centrality for node coloring
    centrality = nx.degree_centrality(G)
    max_cent = max(centrality.values()) if centrality else 1

    for node_id, data in G.nodes(data=True):
        cent = centrality.get(node_id, 0)
        color_idx = min(int((cent / max_cent) * (len(_COLORS) - 1)), len(_COLORS) - 1)
        degree = G.degree(node_id)
        label = data.get("label", node_id)
        hover = f"<b>{label}</b><br>ID: {node_id}<br>Connections: {degree}"
        net.add_node(
            node_id,
            label=label,
            title=hover,
            color=_COLORS[color_idx],
            size=15 + degree * 3,
        )

    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "")
        evidence = data.get("evidence", "")
        hover = f"<b>{relation}</b><br><i>{evidence}</i>"
        net.add_edge(u, v, title=hover, label=relation, color="#888888")

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        },
        "stabilization": {"iterations": 150}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "font": {"size": 10, "color": "#aaaaaa"},
        "smooth": {"type": "continuous"}
      },
      "nodes": {
        "font": {"size": 14},
        "borderWidth": 2,
        "borderWidthSelected": 4
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.write_html(output_path)
    return output_path


def visualize_from_triples(
    entities: list[GraphEntity],
    relations: list[GraphRelation],
    output_path: str = None,
) -> str:
    """Build a visualization from structured graph objects."""
    if output_path is None:
        output_path = os.path.join(ROOT, "outputs", "graph_visualization.html")
    G = build_networkx_graph(entities, relations)
    return visualize_pyvis(G, output_path)


def visualize_from_json(json_path: str, output_path: str = None) -> str:
    """Load a triples JSON file and generate the visualization."""
    with open(json_path) as f:
        data = json.load(f)
    entities, relations = build_graph_objects(data["triples"])
    if output_path is None:
        base = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(ROOT, "outputs", f"{base}_graph.html")
    return visualize_from_triples(entities, relations, output_path)


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs", "triples.json")
    out = visualize_from_json(json_path)
    print(f"Visualization saved to {out}")
    print("Open in your browser to view the interactive graph.")
