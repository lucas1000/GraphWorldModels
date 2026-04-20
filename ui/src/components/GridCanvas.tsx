import { useRef, useEffect, useCallback } from "react";
import { GRID_SIZE, ROOM_NAMES, roomName, type EntityState, type GraphNode, type GraphEdge } from "../lib/grid";

const CELL = 110;
const PAD = 20;
/** Radius for item / door / goal markers (matches prior item circles) */
const NODE_R = 10;
const SIZE = GRID_SIZE * CELL + PAD * 2;
const MOVE_DURATION = 180; // ms for lerp animation

function easeOutQuad(t: number) {
  return t * (2 - t);
}

function drawGridNode(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  letter: string,
) {
  ctx.beginPath();
  ctx.arc(x, y, NODE_R, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.fillStyle = "#101010";
  ctx.font = "500 10px 'Reddit Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(letter, x, y);
}

/** Resolve room name -> (x, y) grid coords */
function roomCoords(name: string): [number, number] | null {
  for (const [key, rn] of Object.entries(ROOM_NAMES)) {
    if (rn === name) {
      const [x, y] = key.split(",").map(Number);
      return [x, y];
    }
  }
  return null;
}

interface Props {
  agent: { x: number; y: number; room: string };
  prevAgent: { x: number; y: number } | null;
  entities: EntityState[];
  inventory: string[];
  visitedCells: Set<string>;
  moveTimestamp: number;
  done: boolean;
  plannedPath: string[];
  modelActive: boolean;
  showRouteGraph: boolean;
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
}

export function GridCanvas({
  agent,
  prevAgent,
  entities,
  inventory,
  visitedCells,
  moveTimestamp,
  done,
  plannedPath,
  modelActive,
  showRouteGraph,
  graphNodes,
  graphEdges,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  const draw = useCallback(
    (now: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;
      const dpr = window.devicePixelRatio || 1;

      // Handle high-DPI
      if (canvas.width !== SIZE * dpr || canvas.height !== SIZE * dpr) {
        canvas.width = SIZE * dpr;
        canvas.height = SIZE * dpr;
        canvas.style.width = `${SIZE}px`;
        canvas.style.height = `${SIZE}px`;
        ctx.scale(dpr, dpr);
      }

      ctx.clearRect(0, 0, SIZE, SIZE);

      // -- Draw cell backgrounds and borders --
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          const cx = PAD + x * CELL;
          const cy = PAD + y * CELL;
          const key = `${x},${y}`;
          const isVisited = visitedCells.has(key);
          const isCurrent = x === agent.x && y === agent.y;
          const isGoal = x === 4 && y === 4;

          // Cell background
          if (isCurrent) {
            ctx.fillStyle = "rgba(108, 140, 255, 0.15)";
          } else if (isGoal) {
            ctx.fillStyle = "rgba(74, 222, 128, 0.08)";
          } else if (isVisited) {
            ctx.fillStyle = "rgba(108, 140, 255, 0.05)";
          } else {
            ctx.fillStyle = "#181818";
          }
          ctx.fillRect(cx, cy, CELL, CELL);

          // Cell border
          ctx.strokeStyle = isCurrent ? "#6c8cff" : "#2e2e2e";
          ctx.lineWidth = isCurrent ? 2 : 1;
          ctx.strokeRect(cx + 0.5, cy + 0.5, CELL - 1, CELL - 1);
        }
      }

      // -- Draw route graph overlay (above grid, below labels) --
      if (showRouteGraph) {
        // 1. Draw ALL traversable edges (static grid adjacency)
        for (let y = 0; y < GRID_SIZE; y++) {
          for (let x = 0; x < GRID_SIZE; x++) {
            const x1 = PAD + x * CELL + CELL / 2;
            const y1 = PAD + y * CELL + CELL / 2;
            // Right neighbour
            if (x < GRID_SIZE - 1) {
              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x1 + CELL, y1);
              ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
              ctx.lineWidth = 1;
              ctx.stroke();
            }
            // Bottom neighbour
            if (y < GRID_SIZE - 1) {
              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x1, y1 + CELL);
              ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
              ctx.lineWidth = 1;
              ctx.stroke();
            }
          }
        }

        // 2. Draw traversed TRANSITION edges brighter on top
        const transitions = graphEdges.filter(
          (e) => e.rel_type === "TRANSITION" && e.visit_count && e.visit_count > 0,
        );
        if (transitions.length > 0) {
          // Build room coord lookup
          const roomPositions = new Map<string, [number, number]>();
          for (const n of graphNodes) {
            if (n.label === "Room" && n.room_x != null && n.room_y != null) {
              roomPositions.set(n.id, [n.room_x, n.room_y]);
            }
          }
          const maxVisits = Math.max(1, ...transitions.map((e) => e.visit_count ?? 1));

          for (const edge of transitions) {
            const from = roomPositions.get(edge.source);
            const to = roomPositions.get(edge.target);
            if (!from || !to) continue;
            if (from[0] === to[0] && from[1] === to[1]) continue;

            const x1 = PAD + from[0] * CELL + CELL / 2;
            const y1 = PAD + from[1] * CELL + CELL / 2;
            const x2 = PAD + to[0] * CELL + CELL / 2;
            const y2 = PAD + to[1] * CELL + CELL / 2;

            const visits = edge.visit_count ?? 1;
            const alpha = 0.15 + 0.35 * (visits / maxVisits);

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }

        // 3. Draw room nodes at cell centres
        for (let y = 0; y < GRID_SIZE; y++) {
          for (let x = 0; x < GRID_SIZE; x++) {
            const nx = PAD + x * CELL + CELL / 2;
            const ny = PAD + y * CELL + CELL / 2;
            ctx.beginPath();
            ctx.arc(nx, ny, 3, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
            ctx.fill();
          }
        }
      }

      // -- Planned path (fills, line, step badges — all before room names) --
      if (modelActive && plannedPath.length > 0) {
        const plannedPathCoords = plannedPath
          .map(roomCoords)
          .filter((c): c is [number, number] => c !== null);

        if (plannedPathCoords.length > 1) {
          for (const [px, py] of plannedPathCoords) {
            const cx = PAD + px * CELL;
            const cy = PAD + py * CELL;
            ctx.fillStyle = "rgba(251, 191, 36, 0.12)";
            ctx.fillRect(cx, cy, CELL, CELL);
            ctx.strokeStyle = "rgba(251, 191, 36, 0.5)";
            ctx.lineWidth = 2;
            ctx.strokeRect(cx + 1, cy + 1, CELL - 2, CELL - 2);
          }

          ctx.beginPath();
          ctx.strokeStyle = "rgba(251, 191, 36, 0.6)";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([6, 4]);
          const [fx, fy] = plannedPathCoords[0];
          ctx.moveTo(PAD + fx * CELL + CELL / 2, PAD + fy * CELL + CELL / 2);
          for (let i = 1; i < plannedPathCoords.length; i++) {
            const [px, py] = plannedPathCoords[i];
            ctx.lineTo(PAD + px * CELL + CELL / 2, PAD + py * CELL + CELL / 2);
          }
          ctx.stroke();
          ctx.setLineDash([]);

          for (let i = 0; i < plannedPathCoords.length; i++) {
            const [px, py] = plannedPathCoords[i];
            const mx = PAD + px * CELL + CELL / 2 + 20;
            const my = PAD + py * CELL + CELL / 2;
            ctx.beginPath();
            ctx.arc(mx, my, 8, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(251, 191, 36, 0.8)";
            ctx.fill();
            ctx.fillStyle = "#101010";
            ctx.font = "500 8px 'Reddit Mono', monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(`${i + 1}`, mx, my);
          }
        }
      }

      // -- Draw room name labels (above route graph + full planned-path overlay) --
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          const cx = PAD + x * CELL;
          const cy = PAD + y * CELL;
          const isCurrent = x === agent.x && y === agent.y;
          const isGoal = x === 4 && y === 4;

          ctx.fillStyle = isCurrent ? "#e2e4ed" : "#5a5e72";
          ctx.font = "300 11px 'Reddit Mono', monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          const name = roomName(x, y);
          ctx.fillText(name, cx + CELL / 2, cy + 18);

          if (isGoal) {
            drawGridNode(ctx, cx + CELL / 2, cy + CELL / 2, "#4ade80", "G");
          }
        }
      }

      // -- Draw entities --
      for (const e of entities) {
        if (e.held) continue; // held items drawn in inventory, not on grid
        const ex = PAD + e.x * CELL + CELL / 2;
        const ey = PAD + e.y * CELL + CELL / 2;

        if (e.type === "door") {
          drawGridNode(ctx, ex, ey, e.locked ? "#f87171" : "#4ade80", e.locked ? "L" : "O");
        } else {
          let color = "#fbbf24";
          if (e.name === "key") color = "#fbbf24";
          else if (e.name === "lamp") color = "#fb923c";
          drawGridNode(ctx, ex, ey, color, e.name[0].toUpperCase());
        }
      }

      // -- Draw agent with lerp animation --
      let agentDrawX = agent.x;
      let agentDrawY = agent.y;

      if (prevAgent) {
        const elapsed = now - moveTimestamp;
        const t = Math.min(elapsed / MOVE_DURATION, 1.0);
        const et = easeOutQuad(t);
        agentDrawX = prevAgent.x + (agent.x - prevAgent.x) * et;
        agentDrawY = prevAgent.y + (agent.y - prevAgent.y) * et;
      }

      const ax = PAD + agentDrawX * CELL + CELL / 2;
      const ay = PAD + agentDrawY * CELL + CELL / 2;

      // Glow
      const gradient = ctx.createRadialGradient(ax, ay, 0, ax, ay, 22);
      gradient.addColorStop(0, "rgba(108, 140, 255, 0.4)");
      gradient.addColorStop(1, "rgba(108, 140, 255, 0)");
      ctx.beginPath();
      ctx.arc(ax, ay, 22, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Agent circle
      ctx.beginPath();
      ctx.arc(ax, ay, 14, 0, Math.PI * 2);
      ctx.fillStyle = done ? "#4ade80" : "#6c8cff";
      ctx.fill();

      // Agent label
      ctx.fillStyle = "#fff";
      ctx.font = "500 12px 'Reddit Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("A", ax, ay);

      // -- Continue animation if lerping --
      if (prevAgent) {
        const elapsed = now - moveTimestamp;
        if (elapsed < MOVE_DURATION) {
          rafRef.current = requestAnimationFrame(draw);
          return;
        }
      }
    },
    [agent, prevAgent, entities, inventory, visitedCells, moveTimestamp, done, plannedPath, modelActive, showRouteGraph, graphNodes, graphEdges],
  );

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // Re-trigger animation loop when prevAgent changes (new move)
  useEffect(() => {
    if (prevAgent) {
      rafRef.current = requestAnimationFrame(draw);
    }
  }, [prevAgent, moveTimestamp, draw]);

  return (
    <div className="grid-canvas-body">
      <canvas
        ref={canvasRef}
        style={{ width: SIZE, height: SIZE, borderRadius: 8 }}
      />
    </div>
  );
}
