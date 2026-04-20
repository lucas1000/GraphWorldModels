import { useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import type { GraphNode, GraphEdge, LegendEntry } from "../lib/grid";

// ---------------------------------------------------------------------------
// D3 node / link types (mutable — d3 writes x, y, etc.)
// ---------------------------------------------------------------------------

interface D3Node extends d3.SimulationNodeDatum {
  id: string;
  label: GraphNode["label"];
  tick?: number;
  room_id?: string;
  done?: boolean;
  stype?: string;
  name?: string;
  entity_type?: string;
  locked?: boolean;
  value?: string;
  atype?: string;
  direction?: string | null;
  target_id?: string | null;
  sensor?: string;
  goal_achieved?: boolean;
  room_x?: number;
  room_y?: number;
  role?: string;
  status?: string;
  agent?: string;
  signature?: string;
}

interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  rel_type: string;
  reward?: number;
  visit_count?: number;
  r_mean?: number;
  action?: string;
  weight?: number;
}

// ---------------------------------------------------------------------------
// Visual helpers
// ---------------------------------------------------------------------------

function nodeColor(d: D3Node): string {
  switch (d.label) {
    case "Entity":
      if (d.entity_type === "door") return "#f87171";
      if (d.entity_type === "box")  return "#a16207";
      if (d.name === "key") return "#fbbf24";
      if (d.name === "lamp") return "#fb923c";
      return "#22d3ee";
    case "Action":        return "#c084fc";
    case "Observation":   return "#5eead4";
    case "Room":          return "#ffffff";
    case "Waypoint":
      // Role-based colouring: goal = green, dead-ends = red, entries = violet,
      // box/door markers = warm; interior = muted.
      if (d.role === "goal")     return "#4ade80";
      if (d.role === "dead_end") return "#f87171";
      if (d.role === "box")      return "#a16207";
      if (d.role === "door")     return "#f59e0b";
      if (d.role === "entry")    return "#a78bfa";
      return "#8b8fa3";
    case "Object":        return "#34d399";
    case "Part":          return "#a78bfa";
    case "Attribute":     return "#fbbf24";
    case "AbstractState": return "#38bdf8";
    case "AbstractAction":return "#f472b6";
    case "Actor":
      // Role-based colouring for the generalised demo.
      switch (d.role) {
        case "Human":  return "#fbbf24";
        case "Client": return "#60a5fa";
        case "Server": return "#a78bfa";
        case "Agent":  return "#34d399";
        case "LLM":    return "#f472b6";
        case "Tool":   return "#22d3ee";
        default:       return "#8b8fa3";
      }
    case "WorkingState":  return "#fbbf24";
    case "AgentState": {
      const sig = d.signature ?? d.id;
      if (sig.includes("hallucinated") || sig.endsWith("!")) return "#f87171";
      if (sig.includes("refined") || sig === "orch:done" || sig === "memory:pra") return "#4ade80";
      if (sig.startsWith("search:")) return "#a78bfa";
      if (sig.startsWith("memory:")) return "#22d3ee";
      return "#8b8fa3";
    }
    case "State":
    default:
      if (d.done) return "#4ade80";
      if (d.stype === "initial" || d.tick === 0) return "#6c8cff";
      return "#8b8fa3";
  }
}

function nodeRadius(d: D3Node): number {
  switch (d.label) {
    case "Entity":         return 9;
    case "Room":           return 9;
    case "Waypoint":       return d.role === "goal" ? 10 : 8;
    case "State":          return d.stype === "initial" || d.done ? 8 : 6;
    case "Action":         return 5;
    case "Observation":    return 4;
    case "Object":         return 9;
    case "Part":           return 6;
    case "Attribute":      return 5;
    case "AbstractState":  return 8;
    case "AbstractAction": return 6;
    case "Actor":          return 9;
    case "WorkingState":   return 8;
    case "AgentState":     return 5;
    default:               return 5;
  }
}

function edgeColor(d: D3Link): string {
  switch (d.rel_type) {
    case "LEADS_TO":       return "#6c8cff";
    case "TRIGGERS":       return "#c084fc";
    case "PRODUCES":       return "#5eead4";
    case "HAS":            return "#4ade8066";
    case "CONCERNS":       return "#22d3ee44";
    case "TRANSITION":     return "#ffffff";
    case "IN_ROOM":        return "#fbbf24";
    case "ADJACENT":       return "#6b7280";
    case "PATH":           return "#a78bfa";
    case "CONTAINS":       return "#a16207";
    case "UNLOCKS":        return "#fbbf24";
    case "GUARDS":         return "#f87171";
    case "PART_OF":        return "#a78bfa";
    case "SUPPORTS":       return "#34d399";
    case "HAS_ATTRIBUTE":  return "#fbbf24";
    case "TRANSITIONS_TO": return "#38bdf8";
    case "INDUCES":        return "#f472b6";
    case "CALLS":          return "#6b7280";
    case "AT":             return "#94a3b8";
    case "OF":             return "#475569";
    case "TARGETS":        return "#c084fc";
    default:               return "#2e2e2e";
  }
}

function edgeWidth(d: D3Link): number {
  switch (d.rel_type) {
    case "LEADS_TO":    return 1.8;
    case "TRIGGERS":    return 1.0;
    case "PRODUCES":    return 1.0;
    case "HAS":         return 0.6;
    case "CONCERNS":    return 0.5;
    case "TRANSITION": {
      const count = d.visit_count ?? 1;
      return Math.min(1.0 + Math.log2(count + 1) * 1.2, 6);
    }
    case "IN_ROOM":         return 0.6;
    case "ADJACENT":        return 0.5;
    case "PATH":            return 1.0;
    case "CONTAINS":        return 1.5;
    case "UNLOCKS":         return 1.3;
    case "GUARDS":          return 1.3;
    case "PART_OF":         return 1.2;
    case "SUPPORTS":        return 1.2;
    case "HAS_ATTRIBUTE":   return 0.8;
    case "TRANSITIONS_TO": {
      const w = d.weight ?? 0.5;
      return Math.min(0.8 + w * 2.0, 3.5);
    }
    case "INDUCES":         return 0.8;
    case "CALLS":           return 0.7;
    case "AT":              return 0.6;
    case "OF":              return 0.6;
    case "TARGETS":         return 0.7;
    default:                return 0.5;
  }
}

function edgeOpacity(d: D3Link): number {
  switch (d.rel_type) {
    case "LEADS_TO":    return 0.8;
    case "TRIGGERS":    return 0.5;
    case "PRODUCES":    return 0.5;
    case "HAS":         return 0.3;
    case "CONCERNS":    return 0.25;
    case "TRANSITION": {
      const count = d.visit_count ?? 1;
      return Math.min(0.3 + count * 0.07, 1.0);
    }
    case "IN_ROOM":         return 0.35;
    case "ADJACENT":        return 0.15;
    case "PATH":            return 0.45;
    case "CONTAINS":        return 0.75;
    case "UNLOCKS":         return 0.7;
    case "GUARDS":          return 0.7;
    case "PART_OF":         return 0.7;
    case "SUPPORTS":        return 0.7;
    case "HAS_ATTRIBUTE":   return 0.5;
    case "TRANSITIONS_TO":  return 0.7;
    case "INDUCES":         return 0.55;
    case "CALLS":           return 0.25;
    case "AT":              return 0.45;
    case "OF":              return 0.35;
    case "TARGETS":         return 0.55;
    default:                return 0.3;
  }
}

function nodeLabel(d: D3Node): string {
  switch (d.label) {
    case "Entity":         return d.name ?? "?";
    case "Room":           return d.id;
    case "Waypoint":       return d.name ?? d.id;
    case "State":          return d.tick !== undefined ? `t${d.tick}` : "";
    case "Action":         return d.atype ?? "act";
    case "Observation":    return "obs";
    case "Object":         return d.name ?? d.id;
    case "Part":           return d.name ?? d.id;
    case "Attribute":      return `${d.name ?? ""}=${d.value ?? ""}`;
    case "AbstractState":  return d.name ?? d.id;
    case "AbstractAction": return d.name ?? d.id;
    case "Actor":          return d.name ?? d.id;
    case "WorkingState":   return d.status ? `ws:${d.status}` : "ws";
    case "AgentState":     return d.signature ?? d.id;
    default:               return "";
  }
}

function nodeTooltip(d: D3Node): string {
  switch (d.label) {
    case "Entity":
      return `Entity: ${d.name} (${d.entity_type})${d.locked ? " [locked]" : ""}`;
    case "Room":
      return `Room: ${d.id} (${d.room_x}, ${d.room_y})`;
    case "Waypoint":
      return `Waypoint: ${d.name ?? d.id}${d.role ? ` [${d.role}]` : ""}`;
    case "State":
      return `State t=${d.tick} ${d.room_id ?? ""}${d.done ? " [DONE]" : ""}`;
    case "Action": {
      let s = `Action t=${d.tick} ${d.atype ?? ""}`;
      if (d.direction) s += ` dir=${d.direction}`;
      if (d.target_id) s += ` target=${d.target_id}`;
      return s;
    }
    case "Observation":
      return `Observation t=${d.tick} sensor=${d.sensor ?? "?"}${d.goal_achieved ? " [goal!]" : ""}`;
    case "Object":         return `Object: ${d.name ?? d.id}`;
    case "Part":           return `Part: ${d.name ?? d.id}`;
    case "Attribute":      return `Attribute: ${d.name}=${d.value ?? ""}`;
    case "AbstractState":  return `AbstractState: ${d.name ?? d.id}`;
    case "AbstractAction": return `AbstractAction: ${d.name ?? d.id}`;
    case "Actor":          return `Actor (${d.role ?? "?"}): ${d.name ?? d.id}`;
    case "WorkingState":   return `WorkingState [${d.status ?? "?"}]`;
    case "AgentState":     return `AgentState ${d.signature ?? d.id}`;
    default:
      return d.id;
  }
}

// Arrow marker colors per relationship type
const MARKER_COLORS: Record<string, string> = {
  LEADS_TO:       "#6c8cff",
  TRIGGERS:       "#c084fc",
  PRODUCES:       "#5eead4",
  HAS:            "#4ade80",
  CONCERNS:       "#22d3ee",
  TRANSITION:     "#ffffff",
  PATH:           "#a78bfa",
  CONTAINS:       "#a16207",
  UNLOCKS:        "#fbbf24",
  GUARDS:         "#f87171",
  PART_OF:        "#a78bfa",
  SUPPORTS:       "#34d399",
  HAS_ATTRIBUTE:  "#fbbf24",
  TRANSITIONS_TO: "#38bdf8",
  INDUCES:        "#f472b6",
  AT:             "#94a3b8",
  OF:             "#475569",
  TARGETS:        "#c084fc",
};

// ---------------------------------------------------------------------------
// Stable key for D3 link data-join (handles both string and resolved-object
// source / target values that D3's forceLink produces).
// ---------------------------------------------------------------------------

function linkKey(d: D3Link): string {
  const s = typeof d.source === "object" ? (d.source as D3Node).id : d.source;
  const t = typeof d.target === "object" ? (d.target as D3Node).id : d.target;
  return `${s}|${t}|${d.rel_type}${d.rel_type === "TRANSITION" && d.action ? `|${d.action}` : ""}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  legend: LegendEntry[];
  onRefresh: () => void;
}

export function GraphView({ nodes, edges, legend, onRefresh }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<D3Node, D3Link> | null>(null);
  const gRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const d3NodesRef = useRef<D3Node[]>([]);
  const d3LinksRef = useRef<D3Link[]>([]);
  const initializedRef = useRef(false);

  // ---- Initialise SVG scaffolding once ------------------------------------
  const ensureInit = useCallback(() => {
    if (initializedRef.current) return;
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const { width, height } = svgEl.getBoundingClientRect();
    if (width === 0 || height === 0) return;

    initializedRef.current = true;
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    const g = svg.append("g");
    gRef.current = g;

    // Zoom / pan
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => g.attr("transform", event.transform)),
    );

    // Arrow markers
    const defs = svg.append("defs");
    for (const [relType, color] of Object.entries(MARKER_COLORS)) {
      defs
        .append("marker")
        .attr("id", `arrow-${relType}`)
        .attr("viewBox", "0 -4 8 8")
        .attr("refX", 16)
        .attr("refY", 0)
        .attr("markerWidth", 5)
        .attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-3L8,0L0,3")
        .attr("fill", color);
    }

    // Layer groups: links ➜ nodes ➜ labels
    g.append("g").attr("class", "links");
    g.append("g").attr("class", "nodes");
    g.append("g").attr("class", "labels");

    // Force simulation (starts empty)
    const forceSim = d3
      .forceSimulation<D3Node>([])
      .force(
        "link",
        d3
          .forceLink<D3Node, D3Link>([])
          .id((d) => d.id)
          .distance((d) => {
            if (d.rel_type === "TRIGGERS" || d.rel_type === "PRODUCES")
              return 25;
            if (d.rel_type === "LEADS_TO") return 50;
            if (d.rel_type === "TRANSITION") return 80;
            if (d.rel_type === "IN_ROOM") return 30;
            if (d.rel_type === "ADJACENT") return 50;
            if (d.rel_type === "PATH") return 45;
            if (d.rel_type === "CONTAINS") return 25;
            if (d.rel_type === "UNLOCKS" || d.rel_type === "GUARDS") return 40;
            return 60;
          }),
      )
      .force("charge", d3.forceManyBody().strength(-40))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide((d: D3Node) => nodeRadius(d) + 2))
      .on("tick", () => {
        const gg = gRef.current;
        if (!gg) return;
        gg.select(".links")
          .selectAll<SVGLineElement, D3Link>("line")
          .attr("x1", (d) => (d.source as any).x)
          .attr("y1", (d) => (d.source as any).y)
          .attr("x2", (d) => (d.target as any).x)
          .attr("y2", (d) => (d.target as any).y);
        gg.select(".nodes")
          .selectAll<SVGCircleElement, D3Node>("circle")
          .attr("cx", (d) => d.x!)
          .attr("cy", (d) => d.y!);
        gg.select(".labels")
          .selectAll<SVGTextElement, D3Node>("text")
          .attr("x", (d) => d.x!)
          .attr("y", (d) => d.y!);
      });

    simRef.current = forceSim;
  }, []);

  // ---- Incremental D3 update on every data change -------------------------
  const updateGraph = useCallback(() => {
    ensureInit();
    const g = gRef.current;
    const forceSim = simRef.current;
    if (!g || !forceSim) return;

    // ----- Nodes: preserve x/y/vx/vy for existing nodes -------------------
    const existingMap = new Map(d3NodesRef.current.map((n) => [n.id, n]));

    const d3Nodes: D3Node[] = nodes.map((n) => {
      const prev = existingMap.get(n.id);
      if (prev) {
        // Keep D3-managed simulation fields, update data properties
        const { x, y, vx, vy, fx, fy, index } = prev;
        return { ...n, x, y, vx, vy, fx, fy, index } as D3Node;
      }
      return { ...n } as D3Node;
    });

    const nodeIdSet = new Set(d3Nodes.map((n) => n.id));
    const d3Links: D3Link[] = edges
      .filter(
        (e) =>
          nodeIdSet.has(e.source as string) &&
          nodeIdSet.has(e.target as string),
      )
      .map((e) => ({ ...e }));

    d3NodesRef.current = d3Nodes;
    d3LinksRef.current = d3Links;

    // ----- Links (enter / update / exit) -----------------------------------
    const noArrow = new Set(["ADJACENT", "IN_ROOM"]);
    g.select(".links")
      .selectAll<SVGLineElement, D3Link>("line")
      .data(d3Links, linkKey)
      .join(
        (enter) =>
          enter
            .append("line")
            .attr("stroke", edgeColor)
            .attr("stroke-width", edgeWidth)
            .attr("stroke-opacity", edgeOpacity)
            .attr("stroke-dasharray", (d) =>
              d.rel_type === "ADJACENT" ? "3,3" : null,
            )
            .attr("marker-end", (d) =>
              noArrow.has(d.rel_type) ? null : `url(#arrow-${d.rel_type})`,
            ),
        (update) =>
          update
            .attr("stroke-width", edgeWidth)
            .attr("stroke-opacity", edgeOpacity),
        (exit) => exit.remove(),
      )
      .each(function (d) {
        // Tooltip for TRANSITION edges (refresh on every pass)
        const el = d3.select(this);
        el.selectAll("title").remove();
        if (d.rel_type === "TRANSITION") {
          el.append("title").text(
            `${d.action ?? "?"}: visits=${d.visit_count ?? 0}, r_mean=${(d.r_mean ?? 0).toFixed(2)}`,
          );
        }
      });

    // ----- Nodes (enter / update / exit) -----------------------------------
    g.select(".nodes")
      .selectAll<SVGCircleElement, D3Node>("circle")
      .data(d3Nodes, (d) => d.id)
      .join(
        (enter) => {
          const c = enter
            .append("circle")
            .attr("r", nodeRadius)
            .attr("fill", nodeColor)
            .attr("stroke", "#101010")
            .attr("stroke-width", 0.8);
          c.append("title").text(nodeTooltip);
          c.call(
            d3
              .drag<SVGCircleElement, D3Node>()
              .on("start", (event, d) => {
                if (!event.active) forceSim.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
              })
              .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
              })
              .on("end", (event, d) => {
                if (!event.active) forceSim.alphaTarget(0);
                d.fx = null;
                d.fy = null;
              }),
          );
          return c;
        },
        (update) => update.attr("fill", nodeColor).attr("r", nodeRadius),
        (exit) => exit.remove(),
      );

    // ----- Labels (enter / update / exit) ----------------------------------
    const labeled = d3Nodes.filter(
      (d) =>
        d.label === "State" ||
        d.label === "Entity" ||
        d.label === "Room" ||
        d.label === "Waypoint",
    );
    g.select(".labels")
      .selectAll<SVGTextElement, D3Node>("text")
      .data(labeled, (d) => d.id)
      .join(
        (enter) =>
          enter
            .append("text")
            .text(nodeLabel)
            .attr("font-size", (d) =>
              d.label === "Entity" || d.label === "Room" ? 9 : 7,
            )
            .attr("fill", "#e2e4ed")
            .attr("text-anchor", "middle")
            .attr("dy", (d) => -(nodeRadius(d) + 4))
            .attr("pointer-events", "none")
            .attr("font-family", "'Reddit Mono', monospace")
            .attr("font-weight", "300"),
        (update) => update,
        (exit) => exit.remove(),
      );

    // ----- Update force simulation -----------------------------------------
    forceSim.nodes(d3Nodes);
    (forceSim.force("link") as d3.ForceLink<D3Node, D3Link>).links(d3Links);

    // Use a gentle alpha for incremental additions so the layout doesn't
    // jump, but a larger alpha when many nodes are new (initial load / reset).
    const newNodeCount = d3Nodes.filter((n) => !existingMap.has(n.id)).length;
    const alpha =
      d3Nodes.length === 0
        ? 0
        : newNodeCount > d3Nodes.length * 0.5
          ? 1.0
          : Math.min(0.15 + newNodeCount * 0.03, 0.5);
    forceSim.alpha(alpha).restart();
  }, [nodes, edges, ensureInit]);

  // Re-render when data changes
  useEffect(() => {
    updateGraph();
  }, [updateGraph]);

  // Fetch graph from Neo4j on first mount
  useEffect(() => {
    onRefresh();
  }, []);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      simRef.current?.stop();
      simRef.current = null;
      gRef.current = null;
      d3NodesRef.current = [];
      d3LinksRef.current = [];
      initializedRef.current = false;
    };
  }, []);

  // Count breakdown for the info line
  const counts: Record<string, number> = {};
  for (const n of nodes) {
    counts[n.label] = (counts[n.label] || 0) + 1;
  }
  const countStr = Object.entries(counts)
    .map(([k, v]) => `${v} ${k}`)
    .join(", ");

  return (
    <div className="graph-view-panel">
      <h3>Graph View</h3>
      <div className="graph-view-toolbar">
        <button onClick={onRefresh}>Refresh</button>
      </div>
      {nodes.length === 0 ? (
        <div className="graph-view-empty">
          Click Refresh to load the graph, or step through the simulation first
        </div>
      ) : null}
      <svg ref={svgRef} className="graph-view-svg" />
      <div className="graph-view-legend">
        {legend.map((l) => {
          const glyph = l.shape === "line" ? "\u2014" : l.shape === "dash" ? "- -" : "\u25CF";
          return (
            <span key={`${l.label}-${l.shape}`}>
              <span style={{ color: l.color }}>{glyph}</span> {l.label}
            </span>
          );
        })}
      </div>
      {nodes.length > 0 ? (
        <div className="graph-view-info">
          {nodes.length} nodes ({countStr}), {edges.length} edges
        </div>
      ) : null}
    </div>
  );
}
