"use client";

import { motion } from "framer-motion";

export interface Stage {
  key: string;
  label: string;
  caption: string;
}

export function Stepper({
  stages,
  current,
}: {
  stages: Stage[];
  current: number;
}) {
  return (
    <ol className="flex flex-col gap-1">
      {stages.map((s, i) => {
        const active = i === current;
        const done = i < current;
        return (
          <li key={s.key} className="relative flex items-start gap-3 py-1.5">
            <div className="flex flex-col items-center">
              <div
                className={[
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold transition-colors",
                  active
                    ? "bg-adobe-red text-white"
                    : done
                      ? "bg-spectrum-ink text-white"
                      : "bg-spectrum-surface text-spectrum-ink3 ring-1 ring-spectrum-border",
                ].join(" ")}
              >
                {done ? "✓" : i + 1}
              </div>
              {i < stages.length - 1 && (
                <div className={done ? "mt-1 h-5 w-px bg-spectrum-ink/30" : "mt-1 h-5 w-px bg-spectrum-border"} />
              )}
            </div>
            <div className="-mt-0.5">
              <div
                className={[
                  "text-[13px] font-semibold leading-tight",
                  active ? "text-spectrum-ink" : done ? "text-spectrum-ink2" : "text-spectrum-ink3",
                ].join(" ")}
              >
                {s.label}
              </div>
              {active && (
                <motion.div
                  initial={{ opacity: 0, y: -2 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-[11px] leading-snug text-spectrum-ink3"
                >
                  {s.caption}
                </motion.div>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
