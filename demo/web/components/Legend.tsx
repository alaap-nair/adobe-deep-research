"use client";

import { NODE_TYPE_COLOR, NODE_TYPE_LABEL } from "@/lib/layout";
import type { DataMode } from "@/lib/types";

const ORDER = ["profile", "Pathway", "Enzyme", "Molecule", "Process", "CellularStructure"];

export function ModeBadge({ mode }: { mode: DataMode }) {
  const live = mode === "live";
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ring-1",
        live
          ? "bg-node-molecule/10 text-node-molecule ring-node-molecule/30"
          : "bg-spectrum-surface text-spectrum-ink2 ring-spectrum-border",
      ].join(" ")}
      title={
        live
          ? "Served live by the Graphiti backend"
          : "Replaying the frozen snapshot of the real run (no backend connected)"
      }
    >
      <span
        className={[
          "h-1.5 w-1.5 rounded-full",
          live ? "animate-pulse bg-node-molecule" : "bg-spectrum-ink3",
        ].join(" ")}
      />
      {live ? "Live · Graphiti" : "Snapshot"}
    </span>
  );
}

export function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      {ORDER.map((t) => (
        <div key={t} className="flex items-center gap-1.5">
          <span
            className="h-2.5 w-2.5 rounded-sm"
            style={{ background: NODE_TYPE_COLOR[t] }}
          />
          <span className="text-[10px] font-medium text-spectrum-ink3">
            {NODE_TYPE_LABEL[t]}
          </span>
        </div>
      ))}
    </div>
  );
}
