// Shared data contract between the FastAPI backend and the web app.
// The backend's /episodes, /subgraph and /ask responses match these shapes,
// and public/snapshot.json bundles a frozen copy of all three.

export type NodeType =
  | "profile"
  | "Pathway"
  | "Enzyme"
  | "Molecule"
  | "Process"
  | "CellularStructure";

export type EdgeKind = "profile" | "domain" | "bridge";

export interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  episode: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  episode: number;
  kind: EdgeKind;
  evidence: string;
}

export interface Subgraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Episode {
  id: string;
  index: number;
  type: "query" | "upload";
  title: string;
  timestamp: string;
  text: string;
  source_description: string;
  extracted_entities: string[];
}

export interface AnswerSide {
  mode: string;
  subtitle: string;
  answer: string;
  citations: string[];
  used_context: string[];
}

export interface AskResult {
  question: string;
  generalized: AnswerSide;
  personalized: AnswerSide;
}

export interface Persona {
  name: string;
  role: string;
  blurb: string;
}

export interface Snapshot {
  meta: { user: string; group_id: string; note: string; final_question: string };
  persona: Persona;
  episodes: Episode[];
  subgraphs: Record<string, Subgraph>;
  ask: AskResult;
}

export type DataMode = "live" | "snapshot";
