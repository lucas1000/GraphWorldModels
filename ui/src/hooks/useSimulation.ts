import { useEffect, useReducer, useRef, useCallback } from "react";
import type {
  TickMessage,
  CypherResult,
  GraphDataMessage,
  GraphDelta,
  EntityState,
  ActionState,
  ActionEvaluation,
  RolloutData,
  ServerMessage,
  ModelStats,
  EnvironmentView,
  ContextMeta,
  LastCall,
  WorkingStateSnapshot,
  FinalArtefactView,
} from "../lib/grid";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export interface SimState {
  connected: boolean;
  environmentView: EnvironmentView;
  contextMeta: ContextMeta | null;
  stateValues: Record<string, unknown>;
  tick: number;
  agent: { x: number; y: number; room: string };
  prevAgent: { x: number; y: number } | null;
  entities: EntityState[];
  inventory: string[];
  action: ActionState | null;
  reward: number;
  totalReward: number;
  done: boolean;
  paused: boolean;
  tps: number;
  lastCypher: string;
  cypherResult: CypherResult | null;
  graphData: GraphDataMessage | null;
  visitedCells: Set<string>;
  moveTimestamp: number; // for animation timing
  // World model
  modelActive: boolean;
  modelThreshold: number;
  plannedPath: string[];
  usingModel: boolean;
  actionEvaluations: ActionEvaluation[];
  modelStats: ModelStats | null;
  rolloutData: RolloutData | null;
  // Generalised-context only
  lastCall: LastCall | null;
  workingState: WorkingStateSnapshot | null;
  artefact: FinalArtefactView | null;
  humanState: string | null;
}

const initialState: SimState = {
  connected: false,
  environmentView: "spatial",
  contextMeta: null,
  stateValues: {},
  tick: 0,
  agent: { x: 0, y: 0, room: "Kitchen" },
  prevAgent: null,
  entities: [],
  inventory: [],
  action: null,
  reward: 0,
  totalReward: 0,
  done: false,
  paused: true,
  tps: 4,
  lastCypher: "",
  cypherResult: null,
  graphData: null,
  visitedCells: new Set(["0,0"]),
  moveTimestamp: 0,
  modelActive: false,
  modelThreshold: 2,
  plannedPath: [],
  usingModel: false,
  actionEvaluations: [],
  modelStats: null,
  rolloutData: null,
  lastCall: null,
  workingState: null,
  artefact: null,
  humanState: null,
};

// ---------------------------------------------------------------------------
// Graph-delta merge helper
// ---------------------------------------------------------------------------

function mergeGraphDelta(
  existing: GraphDataMessage | null,
  delta: GraphDelta,
): GraphDataMessage {
  const existingNodes = existing?.nodes ?? [];
  const existingEdges = existing?.edges ?? [];

  // Add nodes that don't already exist (by id)
  const nodeIds = new Set(existingNodes.map((n) => n.id));
  const newNodes = delta.nodes.filter((n) => !nodeIds.has(n.id));

  // For TRANSITION edges replace existing ones (same source|target|action)
  const transitionKeys = new Set(
    delta.edges
      .filter((e) => e.rel_type === "TRANSITION")
      .map((e) => `${e.source}|${e.target}|${e.action}`),
  );
  const keptEdges = existingEdges.filter((e) => {
    if (e.rel_type === "TRANSITION") {
      return !transitionKeys.has(`${e.source}|${e.target}|${e.action}`);
    }
    return true;
  });

  return {
    type: "graph_data",
    nodes: [...existingNodes, ...newNodes],
    edges: [...keptEdges, ...delta.edges],
    error: null,
  };
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

type SimAction =
  | { type: "connected" }
  | { type: "disconnected" }
  | { type: "tick"; payload: TickMessage }
  | { type: "init"; payload: TickMessage }
  | { type: "status"; payload: { paused: boolean } }
  | { type: "speed"; payload: { tps: number } }
  | { type: "cypher_result"; payload: CypherResult }
  | { type: "graph_data"; payload: GraphDataMessage }
  | { type: "model_status"; payload: { model_active: boolean; model_threshold: number } }
  | { type: "request_context_switch"; payload: EnvironmentView };

function simReducer(state: SimState, action: SimAction): SimState {
  switch (action.type) {
    case "connected":
      return { ...state, connected: true };

    case "disconnected":
      return { ...state, connected: false };

    case "init": {
      const p = action.payload;
      const view = (p.context ?? state.environmentView) as EnvironmentView;
      const visited = new Set<string>();
      visited.add(`${p.agent.x},${p.agent.y}`);
      return {
        ...state,
        environmentView: view,
        contextMeta: p.context_meta ?? state.contextMeta,
        stateValues: p.state_values ?? {},
        tick: p.tick,
        agent: p.agent,
        prevAgent: null,
        entities: p.entities,
        inventory: p.inventory,
        action: p.action,
        reward: p.reward,
        totalReward: p.total_reward,
        done: p.done,
        lastCypher: p.last_cypher,
        visitedCells: visited,
        paused: true,
        moveTimestamp: performance.now(),
        modelActive: p.model_active ?? false,
        modelThreshold: p.model_threshold ?? 2,
        plannedPath: p.planned_path ?? [],
        usingModel: p.using_model ?? false,
        actionEvaluations: p.action_evaluations ?? [],
        modelStats: p.model_stats ?? null,
        rolloutData: p.rollout_data ?? null,
        graphData: p.init_graph ?? null,
        lastCall: p.last_call ?? null,
        workingState: p.working_state ?? null,
        artefact: p.artefact ?? null,
        humanState: p.human_state ?? null,
      };
    }

    case "tick": {
      const p = action.payload;
      // Drop ticks that don't belong to the active context (arrives during switch)
      if (p.context && p.context !== state.environmentView) {
        return state;
      }
      const moved =
        p.agent.x !== state.agent.x || p.agent.y !== state.agent.y;
      const visited = new Set(state.visitedCells);
      visited.add(`${p.agent.x},${p.agent.y}`);
      const graphData = p.graph_delta
        ? mergeGraphDelta(state.graphData, p.graph_delta)
        : state.graphData;
      return {
        ...state,
        stateValues: p.state_values ?? state.stateValues,
        tick: p.tick,
        prevAgent: moved ? { x: state.agent.x, y: state.agent.y } : state.prevAgent,
        agent: p.agent,
        entities: p.entities,
        inventory: p.inventory,
        action: p.action,
        reward: p.reward,
        totalReward: p.total_reward,
        done: p.done,
        lastCypher: p.last_cypher,
        visitedCells: visited,
        moveTimestamp: moved ? performance.now() : state.moveTimestamp,
        modelActive: p.model_active ?? state.modelActive,
        modelThreshold: p.model_threshold ?? state.modelThreshold,
        plannedPath: p.planned_path ?? [],
        usingModel: p.using_model ?? false,
        actionEvaluations: p.action_evaluations ?? [],
        modelStats: p.model_stats ?? state.modelStats,
        rolloutData: p.rollout_data ?? null,
        graphData,
        lastCall: p.last_call ?? state.lastCall,
        workingState: p.working_state ?? state.workingState,
        artefact: p.artefact ?? state.artefact,
        humanState: p.human_state ?? state.humanState,
      };
    }

    case "status":
      return { ...state, paused: action.payload.paused };

    case "speed":
      return { ...state, tps: action.payload.tps };

    case "cypher_result":
      return { ...state, cypherResult: action.payload };

    case "graph_data": {
      const p = action.payload;
      if (p.context && p.context !== state.environmentView) return state;
      return { ...state, graphData: p };
    }

    case "model_status":
      return {
        ...state,
        modelActive: action.payload.model_active,
        modelThreshold: action.payload.model_threshold,
      };

    case "request_context_switch":
      // Local echo — backend will send a fresh `init` that authoritatively
      // updates metadata and graph data. We update the view eagerly so the
      // header toggle feels responsive.
      return { ...state, environmentView: action.payload };

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSimulation(wsUrl: string) {
  const [state, dispatch] = useReducer(simReducer, initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const closedIntentionally = useRef(false);

  const connect = useCallback(() => {
    closedIntentionally.current = false;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => dispatch({ type: "connected" });

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data) as ServerMessage;
      if (msg.type === "tick") {
        dispatch({ type: "tick", payload: msg as TickMessage });
      } else if (msg.type === "init") {
        dispatch({ type: "init", payload: msg as TickMessage });
      } else if (msg.type === "status") {
        dispatch({ type: "status", payload: msg as { paused: boolean; type: "status" } });
      } else if (msg.type === "speed") {
        dispatch({ type: "speed", payload: msg as { tps: number; type: "speed" } });
      } else if (msg.type === "cypher_result") {
        dispatch({ type: "cypher_result", payload: msg as CypherResult });
      } else if (msg.type === "graph_data") {
        dispatch({ type: "graph_data", payload: msg as GraphDataMessage });
      } else if (msg.type === "model_status") {
        dispatch({ type: "model_status", payload: msg as { model_active: boolean; model_threshold: number; type: "model_status" } });
      }
    };

    ws.onclose = () => {
      dispatch({ type: "disconnected" });
      // Only auto-reconnect if the close was unexpected
      if (!closedIntentionally.current) {
        reconnectTimer.current = setTimeout(connect, 2000);
      }
    };

    ws.onerror = () => ws.close();
  }, [wsUrl]);

  useEffect(() => {
    connect();
    return () => {
      closedIntentionally.current = true;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return {
    ...state,
    play: () => send({ type: "play" }),
    pause: () => send({ type: "pause" }),
    step: () => send({ type: "step" }),
    reset: (seed?: number, policy?: string) =>
      send({ type: "reset", seed, policy }),
    setSpeed: (tps: number) => send({ type: "speed", tps }),
    runCypher: (query: string) => send({ type: "cypher", query }),
    fetchGraph: () => send({ type: "graph" }),
    toggleModel: (active: boolean, threshold?: number) =>
      send({ type: "model_toggle", active, threshold }),
    switchEnvironment: (view: EnvironmentView) => {
      dispatch({ type: "request_context_switch", payload: view });
      send({ type: "environment_switch", view });
    },
  };
}
