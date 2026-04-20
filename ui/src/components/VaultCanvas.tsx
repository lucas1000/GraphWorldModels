import { useEffect, useRef } from "react";
import type { EntityState } from "../lib/grid";

/**
 * VaultCanvas — higher-detail view of the spatial grid's southeast corner.
 *
 * Four rooms in a T-shape: Crypt, Vault, Garden in a row; Trophy Room
 * sits above Vault. Each room has N/S/E/W entry/exit waypoints around a
 * Center. Corners are walls — only cardinal moves are possible.
 *
 * The key lives at Trophy.Center (off the direct east-to-Garden path).
 * The agent must detour north from Vault to fetch it before doubling
 * back and crossing the locked door guarding Vault.E ↔ Garden.W.
 */

type Role =
  | "start"
  | "key"
  | "goal"
  | "hub"
  | "passage"
  | "dead_end";

type CellId = "crypt" | "vault" | "garden" | "trophy";

interface Waypoint {
  x: number;
  y: number;
  id: string;
  short: string;
  role: Role;
  cell: CellId;
  cx: number;
  cy: number;
}

interface Cell {
  id: CellId;
  label: string;
  left: number;
  top: number;
  width: number;
  height: number;
}

// Layout constants (SVG pixel coords)
const CELL_W = 170;
const CELL_H = 170;
const CELL_INSET = 30;

// Middle column and mid row
const MID_X = 280;   // left edge of Vault/Trophy
const LEFT_X = 40;   // left edge of Crypt
const RIGHT_X = 520; // left edge of Garden
const TROPHY_Y = 30; // top of Trophy
const ROW_Y = 240;   // top of Crypt/Vault/Garden row

const CELLS: Cell[] = [
  { id: "trophy", label: "Trophy Room", left: MID_X,   top: TROPHY_Y, width: CELL_W, height: CELL_H },
  { id: "crypt",  label: "Crypt",       left: LEFT_X,  top: ROW_Y,    width: CELL_W, height: CELL_H },
  { id: "vault",  label: "Vault",       left: MID_X,   top: ROW_Y,    width: CELL_W, height: CELL_H },
  { id: "garden", label: "Garden",      left: RIGHT_X, top: ROW_Y,    width: CELL_W, height: CELL_H },
];

// Map waypoint grid (x,y) coords from vault_world.py to SVG positions.
// Crypt at grid cols 0-2, row y=0..2. Vault at cols 3-5, y=0..2. Garden
// cols 6-8, y=0..2. Trophy cols 3-5, y=-3..-1.
function cellFor(id: CellId): Cell {
  return CELLS.find((c) => c.id === id)!;
}

function makeWp(id: string, short: string, role: Role, cell: CellId, x: number, y: number, offset: { dx: number; dy: number }): Waypoint {
  const c = cellFor(cell);
  const cx = c.left + c.width / 2 + offset.dx;
  const cy = c.top + c.height / 2 + offset.dy;
  return { x, y, id, short, role, cell, cx, cy };
}

const WAYPOINTS: Waypoint[] = [
  // Crypt (grid cols 0-2, y=0..2)
  makeWp("vp_crypt_n", "N", "dead_end", "crypt", 1, 0, { dx: 0, dy: -CELL_H / 2 + CELL_INSET }),
  makeWp("vp_crypt_w", "W", "dead_end", "crypt", 0, 1, { dx: -CELL_W / 2 + CELL_INSET, dy: 0 }),
  makeWp("vp_crypt_c", "C", "start",    "crypt", 1, 1, { dx: 0, dy: 0 }),
  makeWp("vp_crypt_e", "E", "passage",  "crypt", 2, 1, { dx: CELL_W / 2 - CELL_INSET, dy: 0 }),
  makeWp("vp_crypt_s", "S", "dead_end", "crypt", 1, 2, { dx: 0, dy: CELL_H / 2 - CELL_INSET }),

  // Vault (grid cols 3-5, y=0..2)
  makeWp("vp_vault_n", "N", "passage",  "vault", 4, 0, { dx: 0, dy: -CELL_H / 2 + CELL_INSET }),
  makeWp("vp_vault_w", "W", "passage",  "vault", 3, 1, { dx: -CELL_W / 2 + CELL_INSET, dy: 0 }),
  makeWp("vp_vault_c", "C", "hub",      "vault", 4, 1, { dx: 0, dy: 0 }),
  makeWp("vp_vault_e", "E", "passage",  "vault", 5, 1, { dx: CELL_W / 2 - CELL_INSET, dy: 0 }),
  makeWp("vp_vault_s", "S", "dead_end", "vault", 4, 2, { dx: 0, dy: CELL_H / 2 - CELL_INSET }),

  // Garden (grid cols 6-8, y=0..2)
  makeWp("vp_garden_n", "N", "dead_end", "garden", 7, 0, { dx: 0, dy: -CELL_H / 2 + CELL_INSET }),
  makeWp("vp_garden_w", "W", "passage",  "garden", 6, 1, { dx: -CELL_W / 2 + CELL_INSET, dy: 0 }),
  makeWp("vp_garden_c", "C", "goal",     "garden", 7, 1, { dx: 0, dy: 0 }),
  makeWp("vp_garden_e", "E", "dead_end", "garden", 8, 1, { dx: CELL_W / 2 - CELL_INSET, dy: 0 }),
  makeWp("vp_garden_s", "S", "dead_end", "garden", 7, 2, { dx: 0, dy: CELL_H / 2 - CELL_INSET }),

  // Trophy (grid cols 3-5, y=-3..-1) — above Vault
  makeWp("vp_trophy_n", "N", "dead_end", "trophy", 4, -3, { dx: 0, dy: -CELL_H / 2 + CELL_INSET }),
  makeWp("vp_trophy_w", "W", "dead_end", "trophy", 3, -2, { dx: -CELL_W / 2 + CELL_INSET, dy: 0 }),
  makeWp("vp_trophy_c", "C", "key",      "trophy", 4, -2, { dx: 0, dy: 0 }),
  makeWp("vp_trophy_e", "E", "dead_end", "trophy", 5, -2, { dx: CELL_W / 2 - CELL_INSET, dy: 0 }),
  makeWp("vp_trophy_s", "S", "passage",  "trophy", 4, -1, { dx: 0, dy: CELL_H / 2 - CELL_INSET }),
];

const BY_COORD = new Map(WAYPOINTS.map((w) => [`${w.x},${w.y}`, w]));
const BY_ID = new Map(WAYPOINTS.map((w) => [w.id, w]));

// PATH edges: cardinal neighbours in the logical (x,y) grid.
const PATH_EDGES: { a: Waypoint; b: Waypoint }[] = [];
{
  const seen = new Set<string>();
  for (const w of WAYPOINTS) {
    for (const [dx, dy] of [
      [1, 0], [-1, 0], [0, 1], [0, -1],
    ] as const) {
      const n = BY_COORD.get(`${w.x + dx},${w.y + dy}`);
      if (!n) continue;
      const k = [w.id, n.id].sort().join("|");
      if (seen.has(k)) continue;
      seen.add(k);
      PATH_EDGES.push({ a: w, b: n });
    }
  }
}

function roleFill(role: Role, visited: boolean): string {
  switch (role) {
    case "goal":     return "#4ade80";
    case "start":    return "#6c8cff";
    case "key":      return "#fbbf24";
    case "hub":      return "#60a5fa";
    case "passage":  return "#a78bfa";
    case "dead_end":
    default:         return visited ? "#4b5164" : "#30344a";
  }
}

function isDoorEdge(a: Waypoint, b: Waypoint): boolean {
  const ids = [a.id, b.id].sort().join("|");
  return ids === "vp_garden_w|vp_vault_e";
}

interface Props {
  agent: { x: number; y: number; room: string };
  entities: EntityState[];
  inventory: string[];
  visitedCells: Set<string>;
  done: boolean;
}

export function VaultCanvas({ agent, entities, inventory, visitedCells, done }: Props) {
  const agentWp = BY_COORD.get(`${agent.x},${agent.y}`) ?? BY_ID.get("vp_crypt_c")!;

  // JS-driven agent animation. Each tick, we interpolate explicitly from
  // the *previous* waypoint to the *current* waypoint. This avoids two
  // bugs that CSS transitions exhibit when ticks arrive faster than the
  // transition completes:
  //   1. Veering off path — browser continues interpolation from the
  //      interpolated midpoint, drawing diagonals across cells.
  //   2. Doubling back — new transition reverses direction mid-sweep.
  const agentGroupRef = useRef<SVGGElement>(null);
  const lastAgentRef = useRef<{ x: number; y: number } | null>(null);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const el = agentGroupRef.current;
    if (!el) return;

    // Cancel any in-flight animation from a previous tick.
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    const setTransform = (x: number, y: number) => {
      el.setAttribute("transform", `translate(${x}, ${y})`);
    };

    const prev = lastAgentRef.current;
    const manhattan = prev
      ? Math.abs(prev.x - agent.x) + Math.abs(prev.y - agent.y)
      : 0;
    const teleport = prev === null || manhattan !== 1;
    const prevWp = prev ? BY_COORD.get(`${prev.x},${prev.y}`) : undefined;

    // Teleport (initial mount, reset, context switch, non-cardinal jump):
    // snap instantly; no animation.
    if (teleport || !prevWp) {
      setTransform(agentWp.cx, agentWp.cy);
      lastAgentRef.current = { x: agent.x, y: agent.y };
      return;
    }

    // Cardinal step: interpolate linearly from prev waypoint to current.
    const fromX = prevWp.cx;
    const fromY = prevWp.cy;
    const toX = agentWp.cx;
    const toY = agentWp.cy;
    const duration = 220; // ms — short enough to complete before next tick
    const t0 = performance.now();

    // Snap to the start position first so an interrupted animation cannot
    // leave the agent stranded between waypoints.
    setTransform(fromX, fromY);

    const step = (now: number) => {
      const p = Math.min((now - t0) / duration, 1);
      setTransform(
        fromX + (toX - fromX) * p,
        fromY + (toY - fromY) * p,
      );
      if (p < 1) {
        animFrameRef.current = requestAnimationFrame(step);
      } else {
        animFrameRef.current = null;
      }
    };
    animFrameRef.current = requestAnimationFrame(step);

    lastAgentRef.current = { x: agent.x, y: agent.y };

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [agent.x, agent.y, agentWp.cx, agentWp.cy]);

  const hasKey = inventory.includes("key_v");
  const door = entities.find((e) => e.id === "door_v");
  const doorLocked = door?.locked ?? true;

  const phaseLabel = done
    ? "Escaped"
    : !hasKey
      ? "Seeking key (detour north)"
      : doorLocked
        ? "Returning to door"
        : "Exiting east";

  const vaultE = BY_ID.get("vp_vault_e")!;
  const gardenW = BY_ID.get("vp_garden_w")!;
  const doorX = (vaultE.cx + gardenW.cx) / 2;
  const doorY = vaultE.cy;

  const canvasW = RIGHT_X + CELL_W + 40;
  const canvasH = ROW_Y + CELL_H + 40;

  return (
    <div className="vault-canvas">
      <div className="vault-canvas-main">
      <svg viewBox={`0 0 ${canvasW} ${canvasH}`} className="vault-svg">
        <defs>
          <filter id="vault-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Room rectangles */}
        {CELLS.map((cell) => (
          <g key={cell.id}>
            <rect
              x={cell.left}
              y={cell.top}
              width={cell.width}
              height={cell.height}
              rx="14"
              ry="14"
              fill="#141821"
              stroke={cell.id === "garden" && done ? "#4ade80" : "#2e3446"}
              strokeWidth={cell.id === "garden" && done ? 2 : 1}
            />
            <text
              x={cell.left + 12}
              y={cell.top + 18}
              fontSize="10"
              fill="#6b7280"
              letterSpacing="0.1em"
            >
              {cell.label.toUpperCase()}
            </text>
          </g>
        ))}

        {/* PATH edges */}
        {PATH_EDGES.map((e, i) => {
          const sameCell = e.a.cell === e.b.cell;
          const doorEdge = isDoorEdge(e.a, e.b);
          return (
            <line
              key={`edge-${i}`}
              x1={e.a.cx}
              y1={e.a.cy}
              x2={e.b.cx}
              y2={e.b.cy}
              stroke={doorEdge ? (doorLocked ? "#7f1d1d" : "#14532d") : "#3a3f52"}
              strokeWidth={sameCell ? 1.1 : 1.6}
              strokeOpacity={doorEdge ? 0.7 : sameCell ? 0.6 : 0.85}
              strokeDasharray={sameCell ? undefined : "6,4"}
            />
          );
        })}

        {/* Waypoints */}
        {WAYPOINTS.map((wp) => {
          const visited = visitedCells.has(`${wp.x},${wp.y}`);
          const isCurrent = agentWp.id === wp.id;
          const isSpecial = wp.role === "goal" || wp.role === "start" || wp.role === "key";
          const r = isSpecial ? 11 : 8;
          return (
            <g key={wp.id}>
              <circle
                cx={wp.cx}
                cy={wp.cy}
                r={r}
                fill={roleFill(wp.role, visited)}
                fillOpacity={visited || wp.role !== "dead_end" ? 0.9 : 0.4}
                stroke={isCurrent ? "#ffffff" : "#101010"}
                strokeWidth={isCurrent ? 2 : 0.8}
              />
              <text
                x={wp.cx}
                y={wp.cy + r + 11}
                textAnchor="middle"
                fontSize="9"
                fill="#8b8fa3"
                fontWeight={300}
              >
                {wp.short}
              </text>
            </g>
          );
        })}

        {/* Key — drawn at Trophy.Center until picked up */}
        {!hasKey
          ? (() => {
              const tc = BY_ID.get("vp_trophy_c")!;
              return (
                <g>
                  <circle
                    cx={tc.cx + 16}
                    cy={tc.cy - 16}
                    r="5.5"
                    fill="#fbbf24"
                    stroke="#78350f"
                    strokeWidth="0.8"
                  />
                  <text
                    x={tc.cx + 16}
                    y={tc.cy - 28}
                    textAnchor="middle"
                    fontSize="9"
                    fill="#fbbf24"
                    fontWeight={500}
                  >
                    key
                  </text>
                </g>
              );
            })()
          : null}

        {/* Door — between Vault.E and Garden.W */}
        <g>
          <rect
            x={doorX - 11}
            y={doorY - 22}
            width="22"
            height="44"
            rx="3"
            ry="3"
            fill={doorLocked ? "#991b1b" : "#065f46"}
            stroke={doorLocked ? "#f87171" : "#4ade80"}
            strokeWidth="1.5"
          />
          <text
            x={doorX}
            y={doorY + 38}
            textAnchor="middle"
            fontSize="9"
            fill={doorLocked ? "#fca5a5" : "#86efac"}
            fontWeight={500}
            letterSpacing="0.08em"
          >
            {doorLocked ? "LOCKED" : "UNLOCKED"}
          </text>
        </g>

        {/* Agent — transform controlled by the JS animation effect above */}
        <g
          ref={agentGroupRef}
          className="vault-agent"
          transform={`translate(${agentWp.cx}, ${agentWp.cy})`}
        >
          <circle
            r="10"
            fill="#e11d48"
            stroke="#fecdd3"
            strokeWidth="1.5"
            filter={done ? "url(#vault-glow)" : undefined}
          />
          {hasKey ? (
            <circle r="3.5" cx="7" cy="-7" fill="#fbbf24" stroke="#78350f" strokeWidth="0.7" />
          ) : null}
        </g>
      </svg>

      <div className="vault-legend">
        <div className="vault-phase">
          <span className="vault-phase-label">Phase</span>
          <span className={`vault-phase-value vault-phase-${done ? "done" : "active"}`}>
            {phaseLabel}
          </span>
        </div>
        <div className="vault-badges">
          <span className={`vault-badge ${hasKey ? "vault-badge-on" : "vault-badge-off"}`}>
            Key: {hasKey ? "✓" : "–"}
          </span>
          <span className={`vault-badge ${doorLocked ? "vault-badge-off" : "vault-badge-on"}`}>
            Door: {doorLocked ? "locked" : "unlocked"}
          </span>
          <span className="vault-badge vault-badge-off">
            @ {agentWp.cell}·{agentWp.short}
          </span>
        </div>
      </div>
      </div>
    </div>
  );
}
