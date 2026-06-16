"use client";

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { GraphCanvas } from "@/components/GraphCanvas";
import { Stepper, type Stage } from "@/components/Stepper";
import { EpisodePanel } from "@/components/EpisodePanel";
import { AnswerCompare } from "@/components/AnswerCompare";
import { Legend, ModeBadge } from "@/components/Legend";
import { ask, getPersonaAndEpisodes, getSubgraph } from "@/lib/api";
import type { AskResult, DataMode, Episode, Persona, Subgraph } from "@/lib/types";

const STAGES: Stage[] = [
  { key: "ep1", label: "Episode 1 — Query", caption: "Maya asks about the Calvin cycle. Entities are extracted and linked to her profile." },
  { key: "ep2", label: "Episode 2 — Upload", caption: "Maya uploads her light-reactions notes. A second cluster forms in her subgraph." },
  { key: "ep3", label: "Episode 3 — Query", caption: "Maya connects the two halves. Bridge edges link the clusters." },
  { key: "ask", label: "Ask — live", caption: "A new question runs through the pipeline, generating an answer live." },
  { key: "compare", label: "Compare", caption: "Generalized vs Personalized — the user subgraph changes the answer." },
];

const UPTO = [1, 2, 3, 3, 3];

export default function Page() {
  const [stage, setStage] = useState(0);
  const [persona, setPersona] = useState<Persona | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [finalQuestion, setFinalQuestion] = useState("");
  const [subgraph, setSubgraph] = useState<Subgraph>({ nodes: [], edges: [] });
  const [askResult, setAskResult] = useState<AskResult | null>(null);
  const [asking, setAsking] = useState(false);
  const [mode, setMode] = useState<DataMode>("snapshot");
  const [ready, setReady] = useState(false);

  // Initial load: persona + episodes + first subgraph.
  useEffect(() => {
    (async () => {
      const [meta, sg] = await Promise.all([getPersonaAndEpisodes(), getSubgraph(1)]);
      setPersona(meta.data.persona);
      setEpisodes(meta.data.episodes);
      setFinalQuestion(meta.data.finalQuestion);
      setSubgraph(sg.data);
      setMode(meta.mode === "live" && sg.mode === "live" ? "live" : "snapshot");
      setReady(true);
    })();
  }, []);

  const loadSubgraph = useCallback(async (upto: number) => {
    const sg = await getSubgraph(upto);
    setSubgraph(sg.data);
    setMode((m) => (sg.mode === "live" ? m : "snapshot"));
  }, []);

  // The live "money shot": fire /ask when entering the Ask stage.
  const runAsk = useCallback(async () => {
    if (askResult) return;
    setAsking(true);
    const res = await ask();
    setAskResult(res.data);
    setMode((m) => (res.mode === "live" ? m : "snapshot"));
    setAsking(false);
  }, [askResult]);

  const goTo = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(STAGES.length - 1, next));
      setStage(clamped);
      void loadSubgraph(UPTO[clamped]);
      if (clamped === 3) void runAsk();
    },
    [loadSubgraph, runAsk],
  );

  const next = () => goTo(stage + 1);
  const back = () => goTo(stage - 1);
  const restart = () => {
    setAskResult(null);
    goTo(0);
  };

  const currentEpisode = stage <= 2 ? episodes[stage] ?? null : null;
  const isCompare = stage === 4;
  const atEnd = stage === STAGES.length - 1;

  return (
    <main className="flex h-screen flex-col bg-spectrum-surface2">
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between border-b border-spectrum-border bg-white px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-7 items-center rounded-md bg-adobe-red px-2 text-[13px] font-extrabold tracking-tight text-white">
            Adobe
          </div>
          <div className="leading-tight">
            <div className="text-[14px] font-bold text-spectrum-ink">
              Personalized Knowledge Graph
            </div>
            <div className="text-[11px] text-spectrum-ink3">
              Graphiti-powered per-user context · Adobe × UpSync
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Legend />
          <ModeBadge mode={mode} />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Left rail */}
        <aside className="flex w-[360px] shrink-0 flex-col gap-4 overflow-y-auto border-r border-spectrum-border bg-white p-5">
          {persona && (
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-spectrum-ink3">
                Persona
              </div>
              <div className="mt-1 text-[14px] font-bold text-spectrum-ink">
                {persona.name} <span className="font-normal text-spectrum-ink3">· {persona.role}</span>
              </div>
              <p className="mt-1 text-[12px] leading-snug text-spectrum-ink2">{persona.blurb}</p>
            </div>
          )}

          <div className="border-t border-spectrum-border pt-4">
            <Stepper stages={STAGES} current={stage} />
          </div>

          <div className="border-t border-spectrum-border pt-4">
            {stage <= 2 ? (
              <EpisodePanel episode={currentEpisode} />
            ) : (
              <div className="rounded-xl2 border border-spectrum-border bg-spectrum-surface2 p-4">
                <div className="text-[10px] font-bold uppercase tracking-wider text-spectrum-ink3">
                  What just happened
                </div>
                <p className="mt-1.5 text-[12px] leading-snug text-spectrum-ink2">
                  Three episodes built Maya&apos;s personal subgraph. Now a new question runs
                  through the pipeline <strong>twice</strong> — once with no personal context, and
                  once with her subgraph injected.
                </p>
              </div>
            )}
          </div>
        </aside>

        {/* Main stage */}
        <section className="relative flex min-h-0 flex-1 flex-col p-5">
          <AnimatePresence mode="wait">
            {isCompare && askResult ? (
              <motion.div
                key="compare"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="min-h-0 flex-1"
              >
                <AnswerCompare result={askResult} />
              </motion.div>
            ) : (
              <motion.div
                key="graph"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="relative min-h-0 flex-1"
              >
                <GraphCanvas subgraph={subgraph} />

                {/* Live "Ask" overlay */}
                <AnimatePresence>
                  {stage === 3 && (
                    <motion.div
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 12 }}
                      className="pointer-events-none absolute inset-x-0 bottom-5 mx-auto w-[min(680px,92%)] rounded-xl2 border border-spectrum-border bg-white/95 p-4 shadow-cardHover backdrop-blur"
                    >
                      <div className="text-[10px] font-bold uppercase tracking-wider text-spectrum-ink3">
                        New question
                      </div>
                      <div className="text-[14px] font-semibold text-spectrum-ink">
                        {finalQuestion}
                      </div>
                      <div className="mt-2 flex items-center gap-2 text-[12px] font-medium text-adobe-redDark">
                        {asking ? (
                          <>
                            <span className="h-3 w-3 animate-spin rounded-full border-2 border-adobe-red border-t-transparent" />
                            Generating answers live…
                          </>
                        ) : (
                          <>Ready — continue to compare ↦</>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Controls */}
          <div className="mt-4 flex shrink-0 items-center justify-between">
            <button
              onClick={back}
              disabled={stage === 0}
              className="rounded-lg border border-spectrum-border bg-white px-4 py-2 text-[13px] font-semibold text-spectrum-ink2 shadow-card transition enabled:hover:bg-spectrum-surface disabled:opacity-40"
            >
              ← Back
            </button>

            <div className="text-[12px] text-spectrum-ink3">
              Step {stage + 1} of {STAGES.length}
            </div>

            {atEnd ? (
              <button
                onClick={restart}
                className="rounded-lg border border-spectrum-border bg-white px-5 py-2 text-[13px] font-semibold text-spectrum-ink2 shadow-card transition hover:bg-spectrum-surface"
              >
                ↺ Restart
              </button>
            ) : (
              <button
                onClick={next}
                disabled={!ready || (stage === 3 && asking)}
                className="rounded-lg bg-adobe-red px-5 py-2 text-[13px] font-bold text-white shadow-card transition enabled:hover:bg-adobe-redDark disabled:opacity-50"
              >
                {stage === 2 ? "Ask a new question →" : stage === 3 ? "Compare answers →" : "Next →"}
              </button>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
