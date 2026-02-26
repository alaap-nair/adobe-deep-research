import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

TRIPLES = Path("outputs/triples.json")
OUT_IMG = Path("outputs/graph.png")
OUT_GEXF = Path("outputs/graph.gexf")

def main():
    data = json.loads(TRIPLES.read_text(encoding="utf-8"))
    triples = data["triples"]

    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, relation=r)

    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)

    # Simple viz
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1200, font_size=8, arrows=True)
    edge_labels = {(u, v): G.edges[u, v]["relation"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=200)
    plt.close()

    nx.write_gexf(G, OUT_GEXF)
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"Wrote graph image -> {OUT_IMG}")
    print(f"Wrote graph file  -> {OUT_GEXF} (open in Gephi if you want)")

if __name__ == "__main__":
    main()