import { useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import type { RolloutData, RolloutTrace } from "../lib/grid";

interface Props {
  rolloutData: RolloutData | null;
}

/* ------------------------------------------------------------------ */
/*  Trie node — merges shared rollout path prefixes                   */
/* ------------------------------------------------------------------ */

interface TrieNode {
  room: string;
  count: number;
  avgReturn: number | null;
  /** True if any rollout passing through this node was flagged success. */
  hasSuccess: boolean;
  children: TrieNode[];
  x: number;
  y: number;
}

interface ActionNode {
  action: string;
  label: string;
  qValue: number | null;
  chosen: boolean;
  chosenByModel: boolean;
  known: boolean;         // has any rollout data
  trie: TrieNode | null;
  x: number;
  y: number;
}

/* ------------------------------------------------------------------ */
/*  Build a prefix trie from rollout paths                            */
/* ------------------------------------------------------------------ */

function buildTrie(rollouts: RolloutTrace[]): TrieNode | null {
  const paths = rollouts
    .filter(r => r.path.length > 1)
    .map(r => ({ rooms: r.path.slice(1), ret: r.expected_return, success: !!r.success }));

  if (paths.length === 0) return null;

  type Item = { rooms: string[]; ret: number | null; success: boolean };

  function build(items: Item[], depth: number): TrieNode[] {
    const groups = new Map<string, Item[]>();
    for (const item of items) {
      if (depth >= item.rooms.length) continue;
      const room = item.rooms[depth];
      if (!groups.has(room)) groups.set(room, []);
      groups.get(room)!.push(item);
    }
    const nodes: TrieNode[] = [];
    for (const [room, group] of groups) {
      const returns = group.map(g => g.ret).filter((r): r is number => r !== null);
      const avg = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : null;
      const hasSuccess = group.some(g => g.success);
      nodes.push({
        room,
        count: group.length,
        avgReturn: avg,
        hasSuccess,
        children: build(group, depth + 1),
        x: 0, y: 0,
      });
    }
    return nodes;
  }

  const topChildren = build(paths, 0);
  if (topChildren.length === 0) return null;
  if (topChildren.length === 1) return topChildren[0];

  const allReturns = paths.map(p => p.ret).filter((r): r is number => r !== null);
  const avgAll = allReturns.length > 0 ? allReturns.reduce((a, b) => a + b, 0) / allReturns.length : null;
  return {
    room: "?",
    count: paths.length,
    avgReturn: avgAll,
    hasSuccess: paths.some(p => p.success),
    children: topChildren,
    x: 0, y: 0,
  };
}

/* ------------------------------------------------------------------ */
/*  Layout                                                            */
/* ------------------------------------------------------------------ */

const LEVEL_W = 120;
const LEAF_H = 20;
const GROUP_GAP = 14;
const LEFT_PAD = 16;
const TOP_PAD = 20;
const NODE_R = 5;
const ROOT_R = 8;
const ACTION_RX = 8;
const ACTION_H = 18;

function leafCount(node: TrieNode): number {
  if (node.children.length === 0) return 1;
  return node.children.reduce((s, c) => s + leafCount(c), 0);
}

function actionLeafCount(a: ActionNode): number {
  return a.known && a.trie ? leafCount(a.trie) : 1;
}

function layoutTrie(node: TrieNode, x: number, yStart: number): void {
  node.x = x;
  if (node.children.length === 0) { node.y = yStart + LEAF_H / 2; return; }
  let cy = yStart;
  for (const child of node.children) {
    layoutTrie(child, x + LEVEL_W, cy);
    cy += leafCount(child) * LEAF_H;
  }
  node.y = (node.children[0].y + node.children[node.children.length - 1].y) / 2;
}

function flattenTrie(node: TrieNode, list: TrieNode[] = []): TrieNode[] {
  list.push(node);
  for (const c of node.children) flattenTrie(c, list);
  return list;
}

function abbrev(room: string): string {
  const first = room.split(" ")[0];
  return first.length > 8 ? first.slice(0, 7) + "." : first;
}

function dirLabel(action: string): string {
  // Spatial / scene: "move_north" -> "NORTH"
  if (action.startsWith("move_")) {
    return action.replace("move_", "").toUpperCase();
  }
  // Generalised: shelf-key style names like "invoke_regulatory_db",
  // "delegate_search_endpoint", "invoke_llm_synth", "respond".
  // Strip the leading verb_ prefix when it's a long noun, and abbreviate.
  const stripped = action
    .replace(/^invoke_/, "")
    .replace(/^delegate_/, "→ ")
    .replace(/_/g, " ");
  return stripped.toUpperCase();
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export function RolloutPanel({ rolloutData }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  const render = useCallback(() => {
    const svg = d3.select(svgRef.current);
    if (!svgRef.current || !rolloutData) return;
    svg.selectAll("*").remove();

    const currentRoom = rolloutData.current_room;
    const evals = rolloutData.evaluations;
    if (evals.length === 0) return;

    // Build action nodes — ALL directions, including unknown
    const actions: ActionNode[] = evals.map(ev => ({
      action: ev.action,
      label: dirLabel(ev.action),
      qValue: ev.q_value,
      chosen: ev.chosen,
      chosenByModel: ev.chosen_by_model,
      known: ev.rollouts.length > 0,
      trie: ev.rollouts.length > 0 ? buildTrie(ev.rollouts) : null,
      x: 0,
      y: 0,
    }));

    // Sort: model-chosen first, then viz-chosen, then known, then by q_value
    actions.sort((a, b) => {
      if (a.chosenByModel !== b.chosenByModel) return a.chosenByModel ? -1 : 1;
      if (a.chosen !== b.chosen) return a.chosen ? -1 : 1;
      if (a.known !== b.known) return a.known ? -1 : 1;
      return (b.qValue ?? -Infinity) - (a.qValue ?? -Infinity);
    });

    const hasModelChoice = actions.some(a => a.chosenByModel);
    const isHighlighted = (a: ActionNode) => hasModelChoice ? a.chosenByModel : a.chosen;

    // Layout
    const totalLeaves = actions.reduce((s, a) => s + actionLeafCount(a), 0);
    const totalHeight = totalLeaves * LEAF_H + (actions.length - 1) * GROUP_GAP + TOP_PAD * 2;
    const rootX = LEFT_PAD;
    const rootY = totalHeight / 2;

    let curY = TOP_PAD;
    for (const a of actions) {
      const groupH = actionLeafCount(a) * LEAF_H;
      a.x = rootX + LEVEL_W;
      a.y = curY + groupH / 2;
      if (a.trie) layoutTrie(a.trie, a.x + LEVEL_W, curY);
      curY += groupH + GROUP_GAP;
    }

    // Color scale — absolute, anchored at zero. Saturation tuned to the
    // typical |Q| range across all three demos (per-step rewards are
    // ±0.1–0.6; rollouts that brush the goal reach ~±2). Saturating early
    // keeps small-but-real returns clearly red/green instead of near-grey.
    const colorScale = (val: number | null): string => {
      if (val === null) return "#5a5e72";
      if (val > 0) {
        const t = Math.min(val / 0.5, 1);
        return d3.interpolateRgb("#8b8fa3", "#4ade80")(t);
      }
      const t = Math.min(-val / 0.5, 1);
      return d3.interpolateRgb("#8b8fa3", "#f87171")(t);
    };

    // ViewBox — pad horizontally so centered labels on the leftmost/rightmost
    // nodes (e.g. the root room name) aren't clipped at the default zoom.
    const SVG_PAD_X = 20;
    let maxX = rootX + LEVEL_W * 2 + 40;
    for (const a of actions) {
      if (a.trie) for (const n of flattenTrie(a.trie)) { if (n.x + 70 > maxX) maxX = n.x + 70; }
    }
    const vh = Math.max(totalHeight, 80);
    svg.attr("viewBox", `${-SVG_PAD_X} 0 ${maxX + SVG_PAD_X * 2} ${vh}`)
       .attr("preserveAspectRatio", "xMinYMid meet");

    const g = svg.append("g");
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom as any);

    const linkPath = (x1: number, y1: number, x2: number, y2: number) => {
      const mx = (x1 + x2) / 2;
      return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    };

    // --- Edges ---
    for (const a of actions) {
      const hi = isHighlighted(a);
      g.append("path")
        .attr("d", linkPath(rootX, rootY, a.x, a.y))
        .attr("fill", "none")
        .attr("stroke", hi ? "#6c8cff" : "#5a5e72")
        .attr("stroke-width", hi ? 2 : 1)
        .attr("opacity", hi ? 0.8 : a.known ? 0.35 : 0.15);
    }

    function drawTrieEdges(px: number, py: number, node: TrieNode, hi: boolean) {
      const w = node.hasSuccess ? 2.2 : hi ? 1 + Math.min(node.count * 0.4, 2.5) : 0.8;
      const stroke = node.hasSuccess ? "#4ade80" : colorScale(node.avgReturn);
      g.append("path")
        .attr("d", linkPath(px, py, node.x, node.y))
        .attr("fill", "none")
        .attr("stroke", stroke)
        .attr("stroke-width", w)
        .attr("opacity", node.hasSuccess ? 0.9 : hi ? 0.7 : 0.2);
      for (const child of node.children) drawTrieEdges(node.x, node.y, child, hi);
    }
    for (const a of actions) { if (a.trie) drawTrieEdges(a.x, a.y, a.trie, isHighlighted(a)); }

    // --- Nodes ---

    // Root
    g.append("circle").attr("cx", rootX).attr("cy", rootY).attr("r", ROOT_R)
      .attr("fill", "#6c8cff").attr("stroke", "#fff").attr("stroke-width", 1.5);
    g.append("text").attr("x", rootX).attr("y", rootY - ROOT_R - 4)
      .attr("text-anchor", "middle").attr("fill", "#c8cad0").attr("font-size", 10).attr("font-weight", 500)
      .text(abbrev(currentRoom));

    // Action nodes
    for (const a of actions) {
      const hi = isHighlighted(a);
      const opacity = hi ? 1 : a.known ? 0.55 : 0.25;

      const qText = a.known ? (a.qValue !== null ? a.qValue.toFixed(1) : "?") : "?";
      const label = `${a.label} ${qText}`;
      const pillW = label.length * 6.5 + 12;

      // Pill fill carries Q-value sign: a faint tint of the colorScale
      // colour layered behind the highlight (model/best). Unknown actions
      // stay flat dark so they read as "no data".
      const qTint = a.known ? colorScale(a.qValue) : "#222222";
      const fillColor = a.chosenByModel ? "#2a4a2a" : hi ? "#3a4a7a" : qTint;
      const strokeColor = a.chosenByModel ? "#4ade80" : hi ? "#6c8cff" : a.known ? qTint : "#2e2e2e";

      g.append("rect")
        .attr("x", a.x - pillW / 2).attr("y", a.y - ACTION_H / 2)
        .attr("width", pillW).attr("height", ACTION_H)
        .attr("rx", ACTION_RX).attr("ry", ACTION_RX)
        .attr("fill", fillColor)
        .attr("fill-opacity", a.chosenByModel || hi ? 1 : a.known ? 0.35 : 1)
        .attr("stroke", strokeColor)
        .attr("stroke-width", hi ? 1.5 : 1)
        .attr("opacity", opacity)
        .attr("stroke-dasharray", a.known ? "none" : "3,2");

      g.append("text")
        .attr("x", a.x).attr("y", a.y + 3.5)
        .attr("text-anchor", "middle")
        .attr("fill", hi ? "#fff" : a.known ? "#8b8fa3" : "#5a5e72")
        .attr("font-size", 10).attr("font-weight", hi ? 500 : 300)
        .attr("opacity", opacity)
        .text(label);

      if (a.chosenByModel) {
        g.append("text").attr("x", a.x).attr("y", a.y - ACTION_H / 2 - 4)
          .attr("text-anchor", "middle").attr("fill", "#4ade80").attr("font-size", 8).attr("font-weight", 500)
          .text("MODEL");
      }
    }

    // Trie room nodes
    function drawTrieNodes(node: TrieNode, hi: boolean) {
      const opacity = node.hasSuccess ? 1 : hi ? 0.9 : 0.3;
      const r = NODE_R + Math.min(node.count * 0.5, 3);
      const isLeaf = node.children.length === 0;
      const fill = node.hasSuccess ? "#4ade80" : colorScale(node.avgReturn);
      const stroke = node.hasSuccess ? "#86efac" : hi ? "#fff" : "none";

      g.append("circle").attr("cx", node.x).attr("cy", node.y).attr("r", r)
        .attr("fill", fill)
        .attr("stroke", stroke).attr("stroke-width", node.hasSuccess ? 1.4 : hi ? 0.8 : 0)
        .attr("opacity", opacity);

      const labelText = isLeaf && node.avgReturn !== null
        ? `${abbrev(node.room)} ${node.avgReturn >= 0 ? "+" : ""}${node.avgReturn.toFixed(1)}`
        : abbrev(node.room);
      g.append("text").attr("x", node.x + r + 3).attr("y", node.y + 3)
        .attr("fill", hi ? "#c8cad0" : "#5a5e72").attr("font-size", 9).attr("font-weight", 300)
        .attr("opacity", opacity).text(labelText);

      if (node.count > 1 && !isLeaf) {
        g.append("text").attr("x", node.x).attr("y", node.y - r - 2)
          .attr("text-anchor", "middle").attr("fill", "#5a5e72").attr("font-size", 8)
          .attr("opacity", opacity).text(`×${node.count}`);
      }
      for (const child of node.children) drawTrieNodes(child, hi);
    }
    for (const a of actions) { if (a.trie) drawTrieNodes(a.trie, isHighlighted(a)); }
  }, [rolloutData]);

  useEffect(() => { render(); }, [render]);

  if (!rolloutData) {
    return (
      <div className="rollout-panel">
        <h3>Rollout</h3>
        <div className="rollout-empty">Waiting for first step...</div>
      </div>
    );
  }

  const evals = rolloutData.evaluations;
  const knownCount = evals.filter(ev => ev.rollouts.length > 0).length;
  const totalRollouts = evals.reduce((s, ev) => s + ev.rollouts.length, 0);
  const modelEv = evals.find(ev => ev.chosen_by_model);
  const bestEv = evals.find(ev => ev.chosen);

  return (
    <div className="rollout-panel">
      <h3>Rollout</h3>
      <div className="rollout-info">
        {knownCount}/{evals.length} known · {totalRollouts} simulations
        {modelEv
          ? ` · Model: ${dirLabel(modelEv.action)}`
          : bestEv
            ? ` · Best: ${dirLabel(bestEv.action)}`
            : ""}
      </div>
      <svg ref={svgRef} className="rollout-svg" />
    </div>
  );
}
