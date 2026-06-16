// Deterministic graph layout via dagre. Stable node IDs + deterministic layout
// mean that when the subgraph grows between episodes, existing nodes keep their
// positions and only the new ones animate in (no full reshuffle).

import dagre from "dagre";
import type { GraphEdge, GraphNode } from "./types";

export interface Positioned {
  id: string;
  x: number;
  y: number;
}

const NODE_W = 168;
const NODE_H = 52;

export function layoutGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
): Record<string, { x: number; y: number }> {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: "LR",
    nodesep: 26,
    ranksep: 66,
    marginx: 20,
    marginy: 20,
  });
  g.setDefaultEdgeLabel(() => ({}));

  const ids = new Set(nodes.map((n) => n.id));
  const profile = nodes.find((n) => n.type === "profile");

  // Lay out the entity clusters using ONLY domain edges. The profile and bridge
  // edges point "backward" into the clusters (e.g. ATP → Calvin cycle), which
  // would inflate dagre's rank assignment and sprawl the graph; we draw them as
  // connectors instead and anchor the profile node by hand (below).
  for (const n of nodes) {
    if (n.type === "profile") continue; // positioned manually as the left anchor
    g.setNode(n.id, { width: NODE_W, height: NODE_H });
  }
  for (const e of edges) {
    if (e.kind === "domain" && ids.has(e.source) && ids.has(e.target)) {
      g.setEdge(e.source, e.target);
    }
  }

  dagre.layout(g);

  const out: Record<string, { x: number; y: number }> = {};
  for (const n of nodes) {
    if (n.type === "profile") continue;
    const pos = g.node(n.id);
    if (pos) out[n.id] = { x: pos.x - pos.width / 2, y: pos.y - pos.height / 2 };
  }

  // Anchor the profile node to the left, vertically centered on the cluster.
  if (profile) {
    const xs = Object.values(out).map((p) => p.x);
    const ys = Object.values(out).map((p) => p.y);
    if (xs.length) {
      const minX = Math.min(...xs);
      const midY = (Math.min(...ys) + Math.max(...ys)) / 2;
      out[profile.id] = { x: minX - 230, y: midY };
    } else {
      out[profile.id] = { x: 0, y: 0 };
    }
  }
  return out;
}

export const NODE_TYPE_COLOR: Record<string, string> = {
  profile: "#EB1000",
  Pathway: "#2680EB",
  Enzyme: "#9256D9",
  Molecule: "#2D9D78",
  Process: "#E68619",
  CellularStructure: "#6E6E6E",
};

export const NODE_TYPE_LABEL: Record<string, string> = {
  profile: "User profile",
  Pathway: "Pathway",
  Enzyme: "Enzyme",
  Molecule: "Molecule",
  Process: "Process",
  CellularStructure: "Cellular structure",
};
