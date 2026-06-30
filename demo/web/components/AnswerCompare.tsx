"use client";

import { motion } from "framer-motion";
import type { AnswerSide, AskResult } from "@/lib/types";

function Column({ side, accent, delay }: { side: AnswerSide; accent: boolean; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className={[
        "flex h-full flex-col rounded-xl2 border bg-white p-5 shadow-card",
        accent ? "border-adobe-red ring-1 ring-adobe-red/30" : "border-spectrum-border",
      ].join(" ")}
    >
      <div className="mb-1 flex items-center gap-2">
        <span
          className={[
            "rounded-md px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider",
            accent ? "bg-adobe-red text-white" : "bg-spectrum-surface text-spectrum-ink2",
          ].join(" ")}
        >
          {side.mode}
        </span>
        {accent && (
          <span className="text-[11px] font-semibold text-adobe-redDark">
            user subgraph injected
          </span>
        )}
      </div>
      <p className="mb-3 text-[11px] text-spectrum-ink3">{side.subtitle}</p>

      <p className="text-[13.5px] leading-relaxed text-spectrum-ink">{side.answer}</p>

      {side.used_context.length > 0 && (
        <div className="mt-4">
          <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-adobe-redDark">
            Personal context retrieved
          </div>
          <ul className="space-y-1">
            {side.used_context.map((c) => (
              <li
                key={c}
                className="rounded-md bg-adobe-redTint px-2 py-1 text-[11px] font-medium text-adobe-redDark"
              >
                {c}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-auto pt-4">
        <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-spectrum-ink3">
          Citations
        </div>
        <ul className="space-y-1">
          {side.citations.map((c) => (
            <li
              key={c}
              className="truncate rounded-md bg-spectrum-surface2 px-2 py-1 text-[11px] text-spectrum-ink2 ring-1 ring-spectrum-border"
              title={c}
            >
              {c}
            </li>
          ))}
        </ul>
      </div>
    </motion.div>
  );
}

export function AnswerCompare({ result }: { result: AskResult }) {
  return (
    <div className="flex h-full flex-col">
      <div className="mb-3">
        <div className="text-[10px] font-bold uppercase tracking-wider text-spectrum-ink3">
          Final question
        </div>
        <div className="text-[15px] font-semibold text-spectrum-ink">{result.question}</div>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-2 gap-4">
        <Column side={result.generalized} accent={false} delay={0.05} />
        <Column side={result.personalized} accent delay={0.18} />
      </div>
    </div>
  );
}
