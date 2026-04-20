import { useState } from "react";
import type { ModelStats, EnvironmentView } from "../lib/grid";

interface Props {
  paused: boolean;
  done: boolean;
  tps: number;
  modelActive: boolean;
  modelThreshold: number;
  usingModel: boolean;
  modelStats: ModelStats | null;
  environmentView: EnvironmentView;
  onEnvironmentChange: (view: EnvironmentView) => void;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: (seed?: number, policy?: string) => void;
  onSpeedChange: (tps: number) => void;
  onToggleModel: (active: boolean, threshold?: number) => void;
  showRouteGraph: boolean;
  onToggleRouteGraph: () => void;
  /** Generalised context only: current human-agent state signature,
   *  used to distinguish the "accepted" vs "rejected" terminal badge. */
  humanState?: string | null;
}

export function ControlBar({
  paused,
  done,
  tps,
  modelActive,
  modelThreshold,
  usingModel,
  modelStats,
  environmentView,
  onEnvironmentChange,
  onPlay,
  onPause,
  onStep,
  onReset,
  onSpeedChange,
  onToggleModel,
  showRouteGraph,
  onToggleRouteGraph,
  humanState,
}: Props) {
  const [policy, setPolicy] = useState("random");
  const [seed, setSeed] = useState(42);
  const [threshold, setThreshold] = useState(modelThreshold);

  return (
    <div className="control-bar">
      <div className="control-bar-left">
        <button
          type="button"
          className={
            environmentView === "spatial" ? "control-bar-nav active" : "control-bar-nav"
          }
          onClick={() => onEnvironmentChange("spatial")}
        >
          SPATIAL
        </button>
        <button
          type="button"
          className={
            environmentView === "scene" ? "control-bar-nav active" : "control-bar-nav"
          }
          onClick={() => onEnvironmentChange("scene")}
        >
          SCENE
        </button>
        <button
          type="button"
          className={
            environmentView === "generalised"
              ? "control-bar-nav active"
              : "control-bar-nav"
          }
          onClick={() => onEnvironmentChange("generalised")}
        >
          GENERALISED
        </button>
      </div>

      <div className="separator" />

      {paused ? (
        <button onClick={onPlay} disabled={done}>
          &#9654; Play
        </button>
      ) : (
        <button onClick={onPause} className="active">
          &#9208; Pause
        </button>
      )}

      <button onClick={onStep} disabled={!paused || done}>
        &#9197; Step
      </button>

      <button
        onClick={() => onReset(seed, policy)}
      >
        &#8634; Reset
      </button>

      <div className="separator" />

      <label>
        Speed: {tps.toFixed(1)} tps
        <input
          type="range"
          min="0.5"
          max="30"
          step="0.5"
          value={tps}
          onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
        />
      </label>

      <div className="separator" />

      <label>
        Policy:
        <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
          <option value="random">Random</option>
          <option value="bfs">BFS</option>
        </select>
      </label>

      <label>
        Seed:
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
          style={{
            width: 60,
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
            padding: "2px 6px",
            borderRadius: 4,
            fontFamily: "inherit",
            fontWeight: 300,
            fontSize: 12,
          }}
        />
      </label>

      <div className="separator" />

      <label className={modelActive ? "model-toggle active" : "model-toggle"}>
        Model:
        <button
          className={modelActive ? "toggle-btn on" : "toggle-btn off"}
          onClick={() => onToggleModel(!modelActive, threshold)}
        >
          {modelActive ? "ON" : "OFF"}
        </button>
      </label>

      <label>
        Threshold: {threshold}
        <input
          type="range"
          min="1"
          max="20"
          step="1"
          value={threshold}
          onChange={(e) => {
            const v = parseInt(e.target.value);
            setThreshold(v);
            if (modelActive) onToggleModel(true, v);
          }}
        />
      </label>

      {modelActive && (
        <StatusBadge
          done={done}
          modelStats={modelStats}
          humanState={humanState}
        />
      )}

      <div style={{ flex: 1 }} />

      <label className="model-toggle">
        Route Graph:
        <button
          className={showRouteGraph ? "toggle-btn on" : "toggle-btn off"}
          onClick={onToggleRouteGraph}
        >
          {showRouteGraph ? "ON" : "OFF"}
        </button>
      </label>
    </div>
  );
}

// ---------------------------------------------------------------------------
// StatusBadge — the "PLANNING | 47 edges" indicator shown in the header
// whenever the world model is active. For the generalised demo the
// terminal outcome can be either accepted or rejected by the human, so
// the badge colours and labels branch on ``humanState``.
// ---------------------------------------------------------------------------

function StatusBadge({
  done,
  modelStats,
  humanState,
}: {
  done: boolean;
  modelStats: ModelStats | null;
  humanState?: string | null;
}) {
  // Unified across all three demos: white outlined "RUNNING" while the
  // episode is live, green outlined "GOAL" on success, red outlined
  // "REJECTED" on a rejected generalised artefact.
  const rejected = humanState === "human:rejected";
  const accepted = humanState === "human:accepted" || (done && !rejected);

  let cls = "model-stats running";
  let label = "RUNNING";
  if (rejected) {
    cls = "model-stats rejected";
    label = "REJECTED";
  } else if (accepted) {
    cls = "model-stats goal";
    label = "GOAL";
  }

  return (
    <span className={cls}>
      {label}
      {modelStats ? ` | ${modelStats.edge_count} edges` : ""}
    </span>
  );
}
