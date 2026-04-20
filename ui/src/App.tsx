import { useEffect, useRef, useState, useCallback } from "react";
import { useSimulation } from "./hooks/useSimulation";
import { GridCanvas } from "./components/GridCanvas";
import { ControlBar } from "./components/ControlBar";
import { StatusPanel } from "./components/StatusPanel";
import { CypherConsole } from "./components/CypherConsole";
import { GraphView } from "./components/GraphView";
import { RolloutPanel } from "./components/RolloutPanel";
import { VaultCanvas } from "./components/VaultCanvas";
import { GeneralisedCanvas } from "./components/GeneralisedCanvas";

const WS_URL =
  (window.location.protocol === "https:" ? "wss://" : "ws://") +
  window.location.host +
  "/ws";

/**
 * Generic drag hook — returns a mousedown handler that tracks horizontal
 * or vertical movement and calls `onDrag` with the delta.
 */
function useDrag(
  axis: "x" | "y",
  onDrag: (delta: number) => void,
) {
  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const start = axis === "x" ? e.clientX : e.clientY;
      const onMove = (ev: MouseEvent) => {
        const current = axis === "x" ? ev.clientX : ev.clientY;
        onDrag(current - start);
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };
      document.body.style.cursor = axis === "x" ? "col-resize" : "row-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [axis, onDrag],
  );
  return onMouseDown;
}

export default function App() {
  const sim = useSimulation(WS_URL);
  const lastGraphFetch = useRef(0);

  const [showRouteGraph, setShowRouteGraph] = useState(false);

  // Column widths (px)
  const [leftWidth, setLeftWidth] = useState(300);
  const [rightWidth, setRightWidth] = useState(500);
  // Panel heights inside left column (px)
  const [statusHeight, setStatusHeight] = useState(280);
  const [cypherHeight, setCypherHeight] = useState(240);

  // Refs to capture sizes at drag start
  const leftWidthRef = useRef(leftWidth);
  const rightWidthRef = useRef(rightWidth);
  const statusHeightRef = useRef(statusHeight);
  const cypherHeightRef = useRef(cypherHeight);

  // Left divider drag
  const onLeftDivider = useDrag(
    "x",
    useCallback((delta: number) => {
      setLeftWidth(Math.max(200, Math.min(600, leftWidthRef.current + delta)));
    }, []),
  );
  const onLeftDown = useCallback(
    (e: React.MouseEvent) => {
      leftWidthRef.current = leftWidth;
      onLeftDivider(e);
    },
    [leftWidth, onLeftDivider],
  );

  // Right divider drag
  const onRightDivider = useDrag(
    "x",
    useCallback((delta: number) => {
      setRightWidth(Math.max(200, Math.min(600, rightWidthRef.current - delta)));
    }, []),
  );
  const onRightDown = useCallback(
    (e: React.MouseEvent) => {
      rightWidthRef.current = rightWidth;
      onRightDivider(e);
    },
    [rightWidth, onRightDivider],
  );

  // Status/Cypher horizontal divider drag
  const onStatusDivider = useDrag(
    "y",
    useCallback((delta: number) => {
      setStatusHeight(Math.max(100, Math.min(600, statusHeightRef.current + delta)));
    }, []),
  );
  const onStatusDown = useCallback(
    (e: React.MouseEvent) => {
      statusHeightRef.current = statusHeight;
      onStatusDivider(e);
    },
    [statusHeight, onStatusDivider],
  );

  // Cypher/Rollout horizontal divider drag
  const onCypherDivider = useDrag(
    "y",
    useCallback((delta: number) => {
      setCypherHeight(Math.max(80, Math.min(600, cypherHeightRef.current + delta)));
    }, []),
  );
  const onCypherDown = useCallback(
    (e: React.MouseEvent) => {
      cypherHeightRef.current = cypherHeight;
      onCypherDivider(e);
    },
    [cypherHeight, onCypherDivider],
  );

  // Reset the auto-refresh counter when the episode resets, and
  // periodically request a full graph fetch as a consistency backstop.
  useEffect(() => {
    if (sim.tick === 0) {
      lastGraphFetch.current = 0;
    } else if (sim.tick > 0 && sim.tick - lastGraphFetch.current >= 5) {
      lastGraphFetch.current = sim.tick;
      sim.fetchGraph();
    }
  }, [sim.tick]);

  if (!sim.connected) {
    return (
      <div className="connecting">
        <div>
          Connecting to server<span className="dot">...</span>
        </div>
        <div style={{ fontSize: 12 }}>
          Make sure the Python backend is running on port 8765
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <ControlBar
        paused={sim.paused}
        done={sim.done}
        tps={sim.tps}
        modelActive={sim.modelActive}
        modelThreshold={sim.modelThreshold}
        usingModel={sim.usingModel}
        modelStats={sim.modelStats}
        environmentView={sim.environmentView}
        onEnvironmentChange={sim.switchEnvironment}
        onPlay={sim.play}
        onPause={sim.pause}
        onStep={sim.step}
        onReset={sim.reset}
        onSpeedChange={sim.setSpeed}
        onToggleModel={sim.toggleModel}
        showRouteGraph={showRouteGraph}
        onToggleRouteGraph={() => setShowRouteGraph((v) => !v)}
        humanState={sim.humanState}
      />
      <div className="main-area">
        {/* Left column: Status + Rollout */}
        <div className="left-panel" style={{ width: leftWidth, minWidth: leftWidth }}>
          <div className="left-section" style={{ height: statusHeight, minHeight: statusHeight }}>
            <StatusPanel
              fields={sim.contextMeta?.state_fields ?? []}
              values={sim.stateValues}
            />
          </div>
          <div className="divider-h" onMouseDown={onStatusDown} />
          <div className="left-section-fill">
            <RolloutPanel rolloutData={sim.rolloutData} />
          </div>
        </div>

        <div className="divider-v" onMouseDown={onLeftDown} />

        {/* Center: context-specific main view */}
        <div className="grid-panel">
          <div className="simulation-pane">
            <h3 className="canvas-pane-title">Simulated Environment</h3>
            <div className="simulation-pane-body">
              {sim.environmentView === "spatial" ? (
                <GridCanvas
                  agent={sim.agent}
                  prevAgent={sim.prevAgent}
                  entities={sim.entities}
                  inventory={sim.inventory}
                  visitedCells={sim.visitedCells}
                  moveTimestamp={sim.moveTimestamp}
                  done={sim.done}
                  plannedPath={sim.plannedPath}
                  modelActive={sim.modelActive}
                  showRouteGraph={showRouteGraph}
                  graphNodes={sim.graphData?.nodes ?? []}
                  graphEdges={sim.graphData?.edges ?? []}
                />
              ) : sim.environmentView === "scene" ? (
                <VaultCanvas
                  agent={sim.agent}
                  entities={sim.entities}
                  inventory={sim.inventory}
                  visitedCells={sim.visitedCells}
                  done={sim.done}
                />
              ) : (
                <GeneralisedCanvas
                  currentNode={sim.agent.room || "human"}
                  lastCall={sim.lastCall}
                  workingState={sim.workingState}
                  tick={sim.tick}
                  done={sim.done}
                  orchState={(sim.stateValues?.["orch_state"] as string) ?? "orch:-----"}
                  searchState={(sim.stateValues?.["search_state"] as string) ?? "search:idle"}
                  memoryState={(sim.stateValues?.["memory_state"] as string) ?? "memory:idle"}
                  humanState={sim.humanState ?? "human:drafting"}
                  decisionPoint={sim.rolloutData?.decision_point ?? false}
                  artefact={sim.artefact}
                />
              )}
            </div>
          </div>
        </div>

        <div className="divider-v" onMouseDown={onRightDown} />

        {/* Right column: Cypher + Graph View */}
        <div className="right-panel" style={{ width: rightWidth, minWidth: rightWidth }}>
          <div className="right-section" style={{ height: cypherHeight, minHeight: cypherHeight }}>
            <CypherConsole
              onRunQuery={sim.runCypher}
              result={sim.cypherResult}
              presets={sim.contextMeta?.presets ?? []}
            />
          </div>
          <div className="divider-h" onMouseDown={onCypherDown} />
          <div className="right-section-fill">
            <GraphView
              nodes={sim.graphData?.nodes ?? []}
              edges={sim.graphData?.edges ?? []}
              legend={sim.contextMeta?.legend ?? []}
              onRefresh={sim.fetchGraph}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
