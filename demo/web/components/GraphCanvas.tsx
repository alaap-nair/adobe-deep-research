"use client";

import { useEffect, useMemo, useRef } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import { EntityNode, type EntityNodeData } from "./EntityNode";
import { layoutGraph, NODE_TYPE_COLOR } from "@/lib/layout";
import type { Subgraph } from "@/lib/types";

const nodeTypes = { entity: EntityNode };

const EDGE_COLOR: Record<string, string> = {
  profile: "#EB1000",
  domain: "#B3B3B3",
  bridge: "#E68619",
};

function Canvas({ subgraph, prevNodeIds }: { subgraph: Subgraph; prevNodeIds: Set<string> }) {
  const { fitView } = useReactFlow();
  const positions = useMemo(
    () => layoutGraph(subgraph.nodes, subgraph.edges),
    [subgraph],
  );

  const nodes: Node<EntityNodeData>[] = useMemo(
    () =>
      subgraph.nodes.map((n) => ({
        id: n.id,
        type: "entity",
        position: positions[n.id] ?? { x: 0, y: 0 },
        data: { label: n.label, type: n.type, isNew: !prevNodeIds.has(n.id) },
        draggable: true,
      })),
    [subgraph, positions, prevNodeIds],
  );

  const edges: Edge[] = useMemo(
    () =>
      subgraph.edges.map((e) => {
        const isBridge = e.kind === "bridge";
        const isProfile = e.kind === "profile";
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          label: e.label,
          type: "default",
          animated: isBridge,
          style: {
            stroke: EDGE_COLOR[e.kind] ?? "#B3B3B3",
            strokeWidth: isBridge ? 2.5 : isProfile ? 2 : 1.5,
            strokeDasharray: isProfile ? "5 4" : undefined,
          },
          labelStyle: { fill: "#7A7A7A", fontWeight: 600, fontSize: 9 },
          labelBgStyle: { fill: "#ffffff", fillOpacity: 0.9 },
        };
      }),
    [subgraph],
  );

  // Re-fit whenever the graph changes so growth stays in frame.
  const count = subgraph.nodes.length;
  useEffect(() => {
    const t = setTimeout(() => fitView({ padding: 0.18, duration: 600 }), 80);
    return () => clearTimeout(t);
  }, [count, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      minZoom={0.3}
      maxZoom={1.6}
      proOptions={{ hideAttribution: true }}
      nodesConnectable={false}
      edgesFocusable={false}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#E3E3E6" />
      <Controls showInteractive={false} className="!shadow-card !border-spectrum-border" />
    </ReactFlow>
  );
}

export function GraphCanvas({ subgraph }: { subgraph: Subgraph }) {
  // Track which node IDs were present last render so new ones animate in.
  const prevRef = useRef<Set<string>>(new Set());
  const prevNodeIds = prevRef.current;
  useEffect(() => {
    prevRef.current = new Set(subgraph.nodes.map((n) => n.id));
  }, [subgraph]);

  return (
    <div className="h-full w-full overflow-hidden rounded-xl2 border border-spectrum-border bg-[#fbfbfc]">
      <ReactFlowProvider>
        <Canvas subgraph={subgraph} prevNodeIds={prevNodeIds} />
      </ReactFlowProvider>
    </div>
  );
}

export { NODE_TYPE_COLOR };
