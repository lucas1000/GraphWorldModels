import { useEffect, useMemo, useRef, useState } from "react";
import type { FinalArtefactView } from "../lib/grid";

/**
 * GeneralisedCanvas — non-spatial demo view.
 *
 * Renders a fixed 9-actor call graph (human/client/server/orchestrator/LLM
 * and their tools). The current actor is highlighted; the edge traversed
 * by the most recent hop animates a travelling "packet"; each actor carries
 * a text bubble that types the hop payload character-by-character. The
 * WorkingState document is shown as an inline card.
 *
 * The topology is static (mirrored from graph_world_model/generalised_world.py)
 * so the canvas doesn't rely on graph_data arriving — it can render from a
 * cold start the moment the context switches in.
 */

// ---------------------------------------------------------------------------
// Topology — MUST stay in sync with graph_world_model/generalised_world.py
// ---------------------------------------------------------------------------

type Role =
  | "Agent (Human)"
  | "Agent (LLM)"
  | "Client"
  | "Server"
  | "LLM"
  | "Tool";

interface ActorNode {
  id: string;
  role: Role;
  label: string;
  kind: "infra" | "agent" | "tool";
  /** Normalised 0..1 canvas position (x, y). */
  pos: [number, number];
}

// Mirror of graph_world_model/generalised_world.py::ACTORS.
const ACTORS: ActorNode[] = [
  { id: "human",         role: "Agent (Human)", kind: "infra", label: "Researcher",       pos: [0.05, 0.48] },
  { id: "client",        role: "Client",        kind: "infra", label: "Browser/UI",       pos: [0.16, 0.48] },
  { id: "server",        role: "Server",        kind: "infra", label: "API server",       pos: [0.27, 0.48] },
  { id: "orchestrator",  role: "Agent (LLM)",   kind: "agent", label: "Orchestrator",     pos: [0.42, 0.48] },
  { id: "search_agent",  role: "Agent (LLM)",   kind: "agent", label: "Search Agent",     pos: [0.60, 0.24] },
  { id: "memory_agent",  role: "Agent (LLM)",   kind: "agent", label: "Memory Agent",     pos: [0.60, 0.72] },
  { id: "llm",           role: "LLM",           kind: "tool",  label: "Reasoning LLM",    pos: [0.45, 0.08] },
  { id: "regulatory_db", role: "Tool",   kind: "tool",  label: "Regulatory DB",    pos: [0.80, 0.08] },
  { id: "pubmed",        role: "Tool",   kind: "tool",  label: "PubMed",           pos: [0.86, 0.30] },
  { id: "db",            role: "Tool",   kind: "tool",  label: "WorkingState DB",  pos: [0.86, 0.60] },
];

type CallAction = "request" | "forward" | "dispatch" | "respond"
                | "delegate" | "invoke_llm" | "invoke_db" | "invoke_tool" | "return";

interface TopologyEdge {
  src: string;
  dst: string;
  action: CallAction;
}

const TOPOLOGY: TopologyEdge[] = [
  { src: "human",         dst: "client",        action: "request" },
  { src: "client",        dst: "human",         action: "respond" },
  { src: "client",        dst: "server",        action: "forward" },
  { src: "server",        dst: "client",        action: "respond" },
  { src: "server",        dst: "orchestrator",  action: "dispatch" },
  { src: "orchestrator",  dst: "server",        action: "respond" },
  { src: "orchestrator",  dst: "llm",           action: "invoke_llm" },
  { src: "llm",           dst: "orchestrator",  action: "return" },
  { src: "orchestrator",  dst: "db",            action: "invoke_db" },
  { src: "db",            dst: "orchestrator",  action: "return" },
  { src: "orchestrator",  dst: "search_agent",  action: "delegate" },
  { src: "search_agent",  dst: "orchestrator",  action: "return" },
  { src: "orchestrator",  dst: "memory_agent", action: "delegate" },
  { src: "memory_agent", dst: "orchestrator",  action: "return" },
  { src: "search_agent",  dst: "llm",           action: "invoke_llm" },
  { src: "llm",           dst: "search_agent",  action: "return" },
  { src: "search_agent",  dst: "regulatory_db", action: "invoke_tool" },
  { src: "regulatory_db", dst: "search_agent",  action: "return" },
  { src: "search_agent",  dst: "pubmed",        action: "invoke_tool" },
  { src: "pubmed",        dst: "search_agent",  action: "return" },
  { src: "memory_agent", dst: "llm",           action: "invoke_llm" },
  { src: "llm",           dst: "memory_agent", action: "return" },
  { src: "memory_agent", dst: "db",            action: "invoke_db" },
  { src: "db",            dst: "memory_agent", action: "return" },
];

const ACTORS_BY_ID = new Map(ACTORS.map((a) => [a.id, a]));

/** Collapse src↔dst directed pairs to one undirected edge (the canvas draws
 *  a single line per pair — direction is conveyed by the animated packet). */
const UNDIRECTED_EDGES: { a: ActorNode; b: ActorNode; key: string }[] = (() => {
  const seen = new Set<string>();
  const out: { a: ActorNode; b: ActorNode; key: string }[] = [];
  for (const e of TOPOLOGY) {
    const k = [e.src, e.dst].sort().join("|");
    if (seen.has(k)) continue;
    seen.add(k);
    const a = ACTORS_BY_ID.get(e.src)!;
    const b = ACTORS_BY_ID.get(e.dst)!;
    out.push({ a, b, key: k });
  }
  return out;
})();

// ---------------------------------------------------------------------------
// Styling
// ---------------------------------------------------------------------------

// Canvas dimensions + inner padding. Node positions are renormalised below
// so the leftmost and rightmost actors hug the padding rather than sitting
// wherever the raw ``pos`` values place them — this keeps the layout
// width-filling regardless of the specific normalised coordinates chosen
// in the ACTORS table.
const CANVAS_W = 1000;
const CANVAS_H = 580;
const PAD_X = 55;
const PAD_Y = 60;
const INNER_W = CANVAS_W - 2 * PAD_X;
const INNER_H = CANVAS_H - 2 * PAD_Y;
const NODE_R = 32;

function roleFill(role: Role, current: boolean): string {
  if (current) return "#ffffff";
  switch (role) {
    case "Agent (Human)": return "#fbbf24";
    case "Client":        return "#60a5fa";
    case "Server":        return "#a78bfa";
    case "Agent (LLM)":   return "#34d399";
    case "LLM":           return "#f472b6";
    case "Tool":          return "#22d3ee";
  }
}

function roleTextColor(role: Role, current: boolean): string {
  if (current) return "#101010";
  // Dark text on light fills, light text on dark.
  return role === "Tool" || role === "Client" ? "#101010" : "#101010";
}

/** Short label rendered inside the node circle — e.g. "HUMAN" for
 *  "Agent (Human)", "LLM" for "Agent (LLM)". Keeps the in-circle text
 *  readable at the current node radius. */
function roleShortLabel(role: Role): string {
  const m = role.match(/^Agent \((.+)\)$/);
  return (m ? m[1] : role).toUpperCase();
}

/** Colour the agent state-signature badge by quality.
 *  Hallucinated/rejected → red, refined/accepted → green, intermediate → muted. */
function signatureColor(sig: string): string {
  if (sig.includes("hallucinated") || sig.endsWith("!") || sig === "human:rejected")
    return "#f87171";
  if (sig.includes("refined") || sig === "orch:done" || sig === "memory:pra"
      || sig === "human:accepted")
    return "#4ade80";
  return "#94a3b8";
}

// ---------------------------------------------------------------------------
// Props & types shared with the rest of the UI
// ---------------------------------------------------------------------------

export interface WorkingStateView {
  task?: string;
  primary_endpoint?: string | null;
  effect_size?: number | null;
  sample_size?: number | null;
  status?: string;
}

export interface LastCallView {
  from: string;
  to: string;
  action: string;
  text: string;
}

interface Props {
  currentNode: string;
  lastCall: LastCallView | null;
  workingState: WorkingStateView | null;
  tick: number;
  done: boolean;
  /** Per-agent local-state signatures (e.g. "orch:pefn-", "search:endpoint:refined"). */
  orchState: string;
  searchState: string;
  memoryState: string;
  humanState: string;
  /** True when the current node is an agent deliberating its next tool call. */
  decisionPoint: boolean;
  /** The artefact being assembled — the demo's "goal". */
  artefact: FinalArtefactView | null;
}

// ---------------------------------------------------------------------------
// useTypewriter — reveals text one char at a time. No libraries.
// ---------------------------------------------------------------------------

function useTypewriter(text: string, cps = 80): string {
  const [out, setOut] = useState(text);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (!text) {
      setOut("");
      return;
    }
    setOut("");
    let i = 0;
    const tickMs = Math.max(8, Math.round(1000 / cps));
    timerRef.current = setInterval(() => {
      i += 1;
      setOut(text.slice(0, i));
      if (i >= text.length) {
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      }
    }, tickMs);
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [text, cps]);

  return out;
}

// ---------------------------------------------------------------------------
// Packet animation (source → target)
// ---------------------------------------------------------------------------

function usePacketPosition(
  from: [number, number] | null,
  to: [number, number] | null,
  durationMs = 400,
  tick = 0,
): [number, number] | null {
  const [pos, setPos] = useState<[number, number] | null>(null);
  const rafRef = useRef<number | null>(null);
  // Track the post-arrival hide timeout so a stale one doesn't clobber a
  // freshly-started animation when ticks arrive faster than durationMs + hide.
  const hideRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (hideRef.current !== null) {
      clearTimeout(hideRef.current);
      hideRef.current = null;
    }
    if (!from || !to) {
      setPos(null);
      return;
    }
    // Scale the animation to the inter-tick gap so the packet always reaches
    // its destination (or close to it) before the next hop starts. Cap on
    // both ends so very slow speeds don't feel sluggish and very fast speeds
    // still produce visible motion.
    const t0 = performance.now();
    const step = (now: number) => {
      const p = Math.min((now - t0) / durationMs, 1);
      setPos([from[0] + (to[0] - from[0]) * p, from[1] + (to[1] - from[1]) * p]);
      if (p < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        rafRef.current = null;
        hideRef.current = setTimeout(() => {
          setPos(null);
          hideRef.current = null;
        }, 120);
      }
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (hideRef.current !== null) {
        clearTimeout(hideRef.current);
        hideRef.current = null;
      }
    };
    // `tick` forces re-animation when the same (from,to) pair repeats.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [from?.[0], from?.[1], to?.[0], to?.[1], durationMs, tick]);

  return pos;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function GeneralisedCanvas({
  currentNode, lastCall, workingState, tick, done,
  orchState, searchState, memoryState, humanState, decisionPoint, artefact,
}: Props) {
  const agentStateById: Record<string, string> = {
    human:         humanState,
    orchestrator:  orchState,
    search_agent:  searchState,
    memory_agent: memoryState,
  };
  // Resolve positions for actors (cached; topology is static).
  //
  // Renormalise the raw ``pos`` values so the leftmost / topmost actor
  // sits at 0 and the rightmost / bottommost at 1 — *then* apply padding.
  // This makes the layout fill the canvas width regardless of what
  // normalised coordinates the ACTORS table happens to use, so far-left
  // (``human``) and far-right (``db`` / ``pubmed``) nodes hug the edges
  // symmetrically.
  const positions = useMemo(() => {
    const xs = ACTORS.map((a) => a.pos[0]);
    const ys = ACTORS.map((a) => a.pos[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const m = new Map<string, [number, number]>();
    for (const a of ACTORS) {
      const nx = (a.pos[0] - minX) / rangeX;
      const ny = (a.pos[1] - minY) / rangeY;
      m.set(a.id, [PAD_X + nx * INNER_W, PAD_Y + ny * INNER_H]);
    }
    return m;
  }, []);

  // The "packet" animates along lastCall's edge each tick.
  const packetFrom = lastCall ? positions.get(lastCall.from) ?? null : null;
  const packetTo = lastCall ? positions.get(lastCall.to) ?? null : null;
  const packetPos = usePacketPosition(packetFrom, packetTo, 220, tick);

  const activeEdgeKey = lastCall
    ? [lastCall.from, lastCall.to].sort().join("|")
    : null;

  // Text bubble attaches to the "destination" of the last hop (that's where
  // the payload has just arrived) during and after the packet's traversal.
  const bubbleNode = lastCall?.to ?? null;
  const bubblePos = bubbleNode ? positions.get(bubbleNode) ?? null : null;
  const bubbleText = useTypewriter(lastCall?.text ?? "", 120);

  return (
    <div className="generalised-canvas">
      <svg viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`} className="generalised-svg">
        <defs>
          <filter id="gen-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <marker
            id="gen-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#4a5065" />
          </marker>
        </defs>

        {/* Static topology edges */}
        {UNDIRECTED_EDGES.map(({ a, b, key }) => {
          const ap = positions.get(a.id)!;
          const bp = positions.get(b.id)!;
          const isActive = key === activeEdgeKey;
          return (
            <line
              key={key}
              x1={ap[0]}
              y1={ap[1]}
              x2={bp[0]}
              y2={bp[1]}
              stroke={isActive ? "#ffffff" : "#3a3f52"}
              strokeWidth={isActive ? 2 : 1.2}
              strokeOpacity={isActive ? 0.9 : 0.55}
            />
          );
        })}

        {/* Actor nodes */}
        {ACTORS.map((a) => {
          const [x, y] = positions.get(a.id)!;
          const current = a.id === currentNode;
          const sig = agentStateById[a.id];
          const isDecision = current && decisionPoint && a.kind === "agent";
          return (
            <g key={a.id}>
              {/* Decision-point ring: pulses when an agent is choosing a tool. */}
              {isDecision ? (
                <circle
                  cx={x}
                  cy={y}
                  r={NODE_R + 6}
                  fill="none"
                  stroke="#fbbf24"
                  strokeWidth={1.5}
                  strokeOpacity={0.65}
                  strokeDasharray="3,3"
                />
              ) : null}
              <circle
                cx={x}
                cy={y}
                r={NODE_R}
                fill={roleFill(a.role, current)}
                stroke={current ? "#ffffff" : "#101010"}
                strokeWidth={current ? 2.5 : 1}
                filter={current && done ? "url(#gen-glow)" : undefined}
              />
              {(() => {
                const parenMatch = a.role.match(/^Agent \((.+)\)$/);
                const fill = roleTextColor(a.role, current);
                if (parenMatch) {
                  // Two-line label: "AGENT" over "(HUMAN)" / "(LLM)".
                  // Explicit y offsets give predictable vertical centering
                  // across browsers (dominantBaseline + tspan dy doesn't).
                  return (
                    <>
                      <text
                        x={x}
                        y={y - 5}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fontSize="12"
                        fontWeight={600}
                        fill={fill}
                        letterSpacing="0.04em"
                      >
                        AGENT
                      </text>
                      <text
                        x={x}
                        y={y + 10}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fontSize="12"
                        fontWeight={600}
                        fill={fill}
                        letterSpacing="0.04em"
                      >
                        ({parenMatch[1].toUpperCase()})
                      </text>
                    </>
                  );
                }
                return (
                  <text
                    x={x}
                    y={y}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize="13"
                    fontWeight={600}
                    fill={fill}
                    letterSpacing="0.04em"
                  >
                    {roleShortLabel(a.role)}
                  </text>
                );
              })()}
              <text
                x={x}
                y={y + NODE_R + 18}
                textAnchor="middle"
                fontSize="13"
                fill="#cbd5e1"
              >
                {a.label.toUpperCase()}
              </text>
              {/* Local-state signature badge — agent nodes + human (also an agent). */}
              {sig && (a.kind === "agent" || a.id === "human") ? (
                <text
                  x={x}
                  y={y + NODE_R + 34}
                  textAnchor="middle"
                  fontSize="11"
                  fill={signatureColor(sig)}
                  fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                >
                  {sig}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* Travelling packet for the active hop */}
        {packetPos ? (
          <circle
            cx={packetPos[0]}
            cy={packetPos[1]}
            r={5.5}
            fill="#ffffff"
            opacity={0.95}
          />
        ) : null}

        {/* Text bubble at the destination actor. Computed layout is
            clamped so the bubble never extends past the canvas edges —
            otherwise a long response next to a corner node (e.g. the
            ``human`` on the far left) would spill out of view. */}
        {bubblePos && bubbleText ? (() => {
          const layout = computeBubbleLayout(bubbleText);
          const rawX = bubblePos[0];
          const rawY = bubblePos[1] - NODE_R - 20;
          const halfW = layout.width / 2;
          const edgeGap = 8;
          // Clamp horizontally so the bubble sits inside [edgeGap, CANVAS_W - edgeGap].
          const clampedX = Math.max(halfW + edgeGap,
                             Math.min(CANVAS_W - halfW - edgeGap, rawX));
          // If the bubble would extend above the canvas top, flip it
          // *below* the node (with `flipBelow=true` BubbleText anchors
          // the box downward from the translate origin).
          const wouldClipTop = rawY - layout.height < edgeGap;
          const clampedY = wouldClipTop
            ? bubblePos[1] + NODE_R + 20
            : rawY;
          return (
            <g transform={`translate(${clampedX}, ${clampedY})`}>
              <BubbleText text={bubbleText} layout={layout} flipBelow={wouldClipTop} />
            </g>
          );
        })() : null}
      </svg>

      <ArtefactPanel artefact={artefact} humanState={humanState} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// ArtefactPanel — the demo's headline goal: a 5-section recommendation
// being assembled section-by-section as tools are called. Each section is
// colour-coded by status (green = grounded, red = hallucinated, gray = missing).
// ---------------------------------------------------------------------------

const SECTION_LABELS: Record<string, string> = {
  indication:  "Indication",
  endpoint:    "Primary endpoint",
  effect_size: "Effect size",
  sample_size: "Sample size",
  confidence:  "Confidence",
};

function ArtefactPanel({
  artefact,
  humanState,
}: {
  artefact: FinalArtefactView | null;
  humanState: string;
}) {
  const sectionKeys = ["indication", "endpoint", "effect_size", "sample_size", "confidence"] as const;
  const sections = artefact
    ? sectionKeys.map((key) => ({ key, frag: artefact.sections[key] }))
    : [];
  const grounded = artefact?.grounded ?? 0;
  const total = artefact?.total ?? 5;
  const hallucinated = artefact?.hallucinated ?? 0;

  const verdict =
    humanState === "human:accepted" ? "ACCEPTED" :
    humanState === "human:rejected" ? "REJECTED" :
    humanState === "human:awaiting" ? "AWAITING REVIEW" :
    humanState === "human:queried"  ? "IN PROGRESS" :
    "DRAFTING";

  return (
    <div className="generalised-artefact">
      <div className="generalised-artefact-header">
        <div className="generalised-artefact-title">Response Artefact</div>
        <div className="generalised-artefact-meta">
          <span className="generalised-artefact-progress">
            {grounded}/{total} grounded
            {hallucinated > 0 ? `, ${hallucinated} hallucinated` : ""}
          </span>
          <span className={`generalised-artefact-verdict verdict-${verdict.toLowerCase().replace(/\s/g, "-")}`}>
            {verdict}
          </span>
        </div>
      </div>
      <div className="generalised-artefact-body">
        {sections.length === 0 ? (
          <div className="generalised-artefact-empty">
            Artefact will appear here as the system assembles a response.
          </div>
        ) : (
          sections.map(({ key, frag }) => (
            <div key={key} className={`generalised-artefact-section frag-${frag.status}`}>
              <div className="generalised-artefact-label">
                <span className="frag-dot" />
                {SECTION_LABELS[key]}
                {frag.source ? (
                  <span className="generalised-artefact-source">
                    via {frag.source}
                  </span>
                ) : null}
              </div>
              <div className="generalised-artefact-text">{frag.text}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// BubbleText — word-wrapped multi-line SVG bubble.
// ---------------------------------------------------------------------------

interface BubbleLayout {
  lines: string[];
  width: number;
  height: number;
  padX: number;
  padY: number;
  lineH: number;
  fontSize: number;
}

/** Word-wrap ``text`` and compute the bubble's pixel dimensions.
 *  Hoisted so the caller can clamp the bubble's position against
 *  the canvas bounds before rendering. */
function computeBubbleLayout(text: string): BubbleLayout {
  const maxChars = 58;
  const padX = 16;
  const padY = 12;
  const lineH = 20;
  const fontSize = 16;
  // Approximate mono character width at this fontSize. Generous so long
  // words that exceed ``maxChars`` don't overflow horizontally.
  const approxCharW = 9.0;

  const lines: string[] = [];
  let current = "";
  for (const word of text.split(/\s+/)) {
    if (!word) continue;
    if ((current + " " + word).trim().length > maxChars) {
      if (current) lines.push(current);
      current = word;
    } else {
      current = (current ? current + " " : "") + word;
    }
  }
  if (current) lines.push(current);

  const longest = lines.reduce((m, l) => Math.max(m, l.length), 0);
  const contentW = Math.min(Math.max(longest, 10), maxChars) * approxCharW;
  const width = contentW + padX * 2;
  const height = Math.max(1, lines.length) * lineH + padY * 2;
  return { lines, width, height, padX, padY, lineH, fontSize };
}

function BubbleText({
  text,
  layout,
  flipBelow = false,
}: {
  text: string;
  layout?: BubbleLayout;
  /** If true, the bubble is drawn *below* the group origin rather than
   *  above — used when the actor is too close to the canvas top. */
  flipBelow?: boolean;
}) {
  const L = layout ?? computeBubbleLayout(text);
  const rectY = flipBelow ? 0 : -L.height;
  return (
    <g>
      <rect
        x={-L.width / 2}
        y={rectY}
        width={L.width}
        height={L.height}
        rx={7}
        ry={7}
        fill="#0e1018"
        stroke="#3a3f52"
        strokeWidth={1}
        opacity={0.96}
      />
      {L.lines.map((ln, i) => (
        <text
          key={i}
          x={0}
          y={rectY + L.padY + L.lineH * (i + 0.8)}
          textAnchor="middle"
          fontSize={L.fontSize}
          fill="#e5e7eb"
          fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
        >
          {ln}
        </text>
      ))}
    </g>
  );
}
