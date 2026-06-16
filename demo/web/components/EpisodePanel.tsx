"use client";

import { AnimatePresence, motion } from "framer-motion";
import type { Episode } from "@/lib/types";

function fmtTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function EpisodePanel({ episode }: { episode: Episode | null }) {
  return (
    <AnimatePresence mode="wait">
      {episode && (
        <motion.div
          key={episode.id}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
          className="rounded-xl2 border border-spectrum-border bg-white p-4 shadow-card"
        >
          <div className="mb-2 flex items-center justify-between">
            <span
              className={[
                "rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                episode.type === "upload"
                  ? "bg-node-pathway/10 text-node-pathway"
                  : "bg-adobe-redTint text-adobe-redDark",
              ].join(" ")}
            >
              {episode.type === "upload" ? "Episode · Upload" : "Episode · Query"}
            </span>
            <span className="text-[11px] text-spectrum-ink3">{fmtTime(episode.timestamp)}</span>
          </div>

          <h3 className="text-[13px] font-bold text-spectrum-ink">{episode.title}</h3>

          <div className="mt-2 rounded-lg bg-spectrum-surface2 p-2.5 text-[12px] leading-snug text-spectrum-ink2 ring-1 ring-spectrum-border">
            {episode.text}
          </div>

          <div className="mt-3">
            <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-spectrum-ink3">
              Entities extracted → linked to Maya
            </div>
            <div className="flex flex-wrap gap-1.5">
              {episode.extracted_entities.map((e, i) => (
                <motion.span
                  key={e}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.15 + i * 0.04 }}
                  className="rounded-md border border-spectrum-border bg-white px-2 py-0.5 text-[11px] font-medium text-spectrum-ink2"
                >
                  {e}
                </motion.span>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
