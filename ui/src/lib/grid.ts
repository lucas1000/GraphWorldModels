/** Room names mirroring world.py ROOM_NAMES. */
export const GRID_SIZE = 5;

export const ROOM_NAMES: Record<string, string> = {
  "0,0": "Kitchen",       "1,0": "Pantry",        "2,0": "Dining Room",  "3,0": "Lounge",       "4,0": "Balcony",
  "0,1": "Hallway",       "1,1": "Living Room",   "2,1": "Study",        "3,1": "Music Room",   "4,1": "Sunroom",
  "0,2": "Basement",      "1,2": "Storage",       "2,2": "Workshop",     "3,2": "Gallery",      "4,2": "Conservatory",
  "0,3": "Cellar",        "1,3": "Wine Room",     "2,3": "Armory",       "3,3": "Trophy Room",  "4,3": "Chapel",
  "0,4": "Dungeon",       "1,4": "Tunnel",        "2,4": "Crypt",        "3,4": "Vault",        "4,4": "Garden",
};

export function roomName(x: number, y: number): string {
  return ROOM_NAMES[`${x},${y}`] ?? `Room(${x},${y})`;
}

/** Shared types matching the server WebSocket protocol. */
export interface EntityState {
  id: string;
  name: string;
  type: string;
  x: number;
  y: number;
  pickable: boolean;
  locked: boolean;
  held: boolean;
}

export interface ActionState {
  type: string;
  direction: string | null;
  target_id: string | null;
}

export interface ModelStats {
  edge_count: number;
  total_visits: number;
  avg_visits: number;
  max_visits: number;
  min_visits: number;
}

export interface ActionEvaluation {
  action: string;
  q_value: number | null;
  immediate_next: string | null;
  immediate_reward: number;
  n_rollouts: number;
  chosen: boolean;
}

export interface RolloutTrace {
  path: string[];
  expected_return: number | null;
  steps: number;
  /** True when the rollout saw a goal-sized reward (reached escape). */
  success?: boolean;
}

export interface RolloutEvaluation {
  action: string;
  q_value: number | null;
  immediate_next: string | null;
  immediate_reward: number;
  n_rollouts: number;
  chosen: boolean;
  chosen_by_model: boolean;
  rollouts: RolloutTrace[];
}

export interface RolloutData {
  current_room: string;
  evaluations: RolloutEvaluation[];
  /** Generalised demo: false at infra/tool nodes (no choice being made). */
  decision_point?: boolean;
}

export type EnvironmentView = "spatial" | "scene" | "generalised";

export interface StateField {
  key: string;
  label: string;
  kind: "text" | "number" | "list" | "reward";
}

export interface PresetQuery {
  label: string;
  query: string;
}

export interface LegendEntry {
  label: string;
  color: string;
  shape: "dot" | "line" | "dash";
}

export interface ContextMeta {
  key: EnvironmentView;
  label: string;
  central_view: "grid" | "scene" | "generalised";
  state_fields: StateField[];
  presets: PresetQuery[];
  legend: LegendEntry[];
}

export interface GraphDelta {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface LastCall {
  /** Actor id the hop originated from. */
  from: string;
  /** Actor id the hop arrived at. */
  to: string;
  /** "delegate" | "invoke_llm" | "invoke_tool" | "return" */
  action: string;
  /** Payload text associated with this hop. */
  text: string;
}

export interface WorkingStateSnapshot {
  task?: string;
  primary_endpoint?: string | null;
  effect_size?: number | null;
  sample_size?: number | null;
  status?: string;
}

export type FragmentStatus = "grounded" | "hallucinated" | "missing";

export interface ArtefactFragmentView {
  status: FragmentStatus;
  text: string;
  source?: string | null;
}

export interface FinalArtefactView {
  sections: {
    indication:  ArtefactFragmentView;
    endpoint:    ArtefactFragmentView;
    effect_size: ArtefactFragmentView;
    sample_size: ArtefactFragmentView;
    confidence:  ArtefactFragmentView;
  };
  grounded: number;
  hallucinated: number;
  total: number;
}

export interface TickMessage {
  type: "tick" | "init";
  /** Demo context this tick/init belongs to. Frontend drops stale msgs. */
  context?: EnvironmentView;
  tick: number;
  agent: { x: number; y: number; room: string };
  entities: EntityState[];
  inventory: string[];
  action: ActionState | null;
  reward: number;
  total_reward: number;
  done: boolean;
  last_cypher: string;
  model_active: boolean;
  model_threshold: number;
  planned_path: string[];
  using_model: boolean;
  action_evaluations: ActionEvaluation[];
  model_stats: ModelStats | null;
  rollout_data: RolloutData | null;
  /** Values keyed by StateField.key for the active context. */
  state_values?: Record<string, unknown>;
  /** Incremental graph nodes/edges produced by the latest step. */
  graph_delta?: GraphDelta;
  /** Full initial graph snapshot sent on reset / connect. */
  init_graph?: GraphDataMessage;
  /** Sent on init — defines STATE fields, Cypher presets, and legend. */
  context_meta?: ContextMeta;
  /** Generalised-context only: the most recent call hop. */
  last_call?: LastCall | null;
  /** Generalised-context only: current WorkingState document. */
  working_state?: WorkingStateSnapshot | null;
  /** Generalised-context only: the artefact being assembled for the human. */
  artefact?: FinalArtefactView | null;
  /** Generalised-context only: human's current state signature. */
  human_state?: string | null;
}

export interface CypherResult {
  type: "cypher_result";
  columns: string[];
  rows: any[][];
  error: string | null;
}

export interface StatusMessage {
  type: "status";
  paused: boolean;
}

export interface SpeedMessage {
  type: "speed";
  tps: number;
}

export type GraphNodeLabel =
  | "State" | "Entity" | "Action" | "Observation" | "Room" | "Waypoint"
  | "Object" | "Part" | "Attribute"
  | "AbstractState" | "AbstractAction"
  | "Actor" | "WorkingState" | "AgentState";

export interface GraphNode {
  id: string;
  label: GraphNodeLabel;
  // State props
  tick?: number;
  room_id?: string;
  done?: boolean;
  stype?: string;
  // Entity / Object / AbstractState / AbstractAction / Actor props
  name?: string;
  entity_type?: string;
  locked?: boolean;
  value?: string;
  // Action props
  atype?: string;
  direction?: string | null;
  target_id?: string | null;
  call_type?: string;
  text?: string;
  // Observation props
  sensor?: string;
  goal_achieved?: boolean;
  // Room / Waypoint / Actor props
  room_x?: number;
  room_y?: number;
  /** Waypoint role: entry | interior | dead_end | goal | box | door; Actor role: Human|Client|Server|Agent|LLM|Tool. */
  role?: string;
  // WorkingState props
  task?: string;
  primary_endpoint?: string | null;
  effect_size?: number | null;
  sample_size?: number | null;
  status?: string;
  // AgentState props
  agent?: string;
  signature?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  rel_type: string;
  reward?: number;
  visit_count?: number;
  r_mean?: number;
  action?: string;
  weight?: number;
}

export interface GraphDataMessage {
  type: "graph_data";
  nodes: GraphNode[];
  edges: GraphEdge[];
  context?: EnvironmentView;
  error: string | null;
}

export interface ModelStatusMessage {
  type: "model_status";
  model_active: boolean;
  model_threshold: number;
}

export type ServerMessage = TickMessage | CypherResult | StatusMessage | SpeedMessage | GraphDataMessage | ModelStatusMessage;
