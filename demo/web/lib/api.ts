// Data layer. Tries the live Graphiti FastAPI backend first; on any failure
// (backend down, e.g. the deployed Vercel link), transparently falls back to
// the bundled snapshot.json — the frozen replay of the same real run.
//
// Every fetch reports which source served it so the UI can show a Live|Snapshot
// badge (honest live-vs-mocked disclosure, per the assignment).

import type { AskResult, DataMode, Episode, Snapshot, Subgraph } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

// Short timeout so a missing backend falls back to the snapshot fast.
const LIVE_TIMEOUT_MS = 2500;

let snapshotCache: Snapshot | null = null;

async function loadSnapshot(): Promise<Snapshot> {
  if (snapshotCache) return snapshotCache;
  const res = await fetch("/snapshot.json", { cache: "force-cache" });
  if (!res.ok) throw new Error("snapshot.json missing");
  snapshotCache = (await res.json()) as Snapshot;
  return snapshotCache;
}

async function liveFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), LIVE_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}${path}`, { ...init, signal: ctrl.signal });
    if (!res.ok) throw new Error(`live ${path} -> ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export interface Tagged<T> {
  data: T;
  mode: DataMode;
}

export async function getPersonaAndEpisodes(): Promise<
  Tagged<{ persona: Snapshot["persona"]; episodes: Episode[]; finalQuestion: string }>
> {
  try {
    const data = await liveFetch<{
      persona: Snapshot["persona"];
      episodes: Episode[];
      final_question: string;
    }>("/episodes");
    return {
      data: { persona: data.persona, episodes: data.episodes, finalQuestion: data.final_question },
      mode: "live",
    };
  } catch {
    const snap = await loadSnapshot();
    return {
      data: { persona: snap.persona, episodes: snap.episodes, finalQuestion: snap.meta.final_question },
      mode: "snapshot",
    };
  }
}

export async function getSubgraph(upto: number): Promise<Tagged<Subgraph>> {
  try {
    const data = await liveFetch<Subgraph>(`/subgraph?upto=${upto}`);
    return { data, mode: "live" };
  } catch {
    const snap = await loadSnapshot();
    const sg = snap.subgraphs[String(upto)] ?? snap.subgraphs["0"];
    return { data: sg, mode: "snapshot" };
  }
}

export async function ask(): Promise<Tagged<AskResult>> {
  try {
    const data = await liveFetch<AskResult>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    return { data, mode: "live" };
  } catch {
    const snap = await loadSnapshot();
    return { data: snap.ask, mode: "snapshot" };
  }
}
