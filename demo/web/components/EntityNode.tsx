"use client";

import { Handle, Position } from "@xyflow/react";
import { motion } from "framer-motion";
import { NODE_TYPE_COLOR } from "@/lib/layout";
import type { NodeType } from "@/lib/types";

export interface EntityNodeData {
  label: string;
  type: NodeType;
  isNew: boolean;
  [key: string]: unknown;
}

// A single graph node. The profile node (Maya) is the red anchor; entity nodes
// are quiet white chips with a type-colored left bar. New nodes spring in.
export function EntityNode({ data }: { data: EntityNodeData }) {
  const color = NODE_TYPE_COLOR[data.type] ?? "#6E6E6E";
  const isProfile = data.type === "profile";

  return (
    <motion.div
      initial={data.isNew ? { scale: 0.4, opacity: 0 } : false}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 22 }}
    >
      <Handle type="target" position={Position.Left} className="!opacity-0" />
      {isProfile ? (
        <div className="flex flex-col items-center justify-center rounded-full bg-adobe-red px-5 py-3 text-white shadow-cardHover ring-4 ring-adobe-redTint">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-white/80">
            User
          </span>
          <span className="text-base font-bold leading-tight">{data.label}</span>
        </div>
      ) : (
        <div
          className="flex items-center gap-2 rounded-xl border border-spectrum-border bg-white py-2 pl-2 pr-3 shadow-card"
          style={{ borderLeft: `4px solid ${color}` }}
        >
          <span className="text-[13px] font-semibold text-spectrum-ink">{data.label}</span>
        </div>
      )}
      <Handle type="source" position={Position.Right} className="!opacity-0" />
    </motion.div>
  );
}
