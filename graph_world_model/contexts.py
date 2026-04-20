"""Context registry — per-demo STATE fields, Cypher presets, legend, and view.

Each demo context (spatial, scene, generalised) has its own subgraph partition
in Neo4j (via the ``context`` property on every node/edge), its own set of
STATE panel fields, its own Cypher console presets, and its own legend.

The registry is the single source of truth shared between backend and frontend:
the backend attaches a context's metadata to every ``init`` message and the
frontend renders its STATE panel, Cypher presets, and legend directly from
that payload.

Scene and generalised entries are placeholders. Their simulations currently
produce a tiny hand-authored subgraph on reset and do not advance beyond
tick 0 — the plumbing is exercised end-to-end while the domain design is
deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class StateField:
    """One row of the STATE panel."""
    key: str
    label: str
    kind: str = "text"  # "text" | "number" | "list" | "reward"


@dataclass
class PresetQuery:
    label: str
    query: str


@dataclass
class LegendEntry:
    label: str
    color: str
    shape: str = "dot"  # "dot" | "line" | "dash"


@dataclass
class ContextDef:
    key: str
    label: str
    central_view: str  # "grid" | "scene" | "generalised"
    state_fields: list[StateField]
    presets: list[PresetQuery]
    legend: list[LegendEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "central_view": self.central_view,
            "state_fields": [f.__dict__ for f in self.state_fields],
            "presets": [p.__dict__ for p in self.presets],
            "legend": [e.__dict__ for e in self.legend],
        }


# ---------------------------------------------------------------------------
# Spatial environment (the existing demo)
# ---------------------------------------------------------------------------


SPATIAL = ContextDef(
    key="spatial",
    label="Spatial Environment",
    central_view="grid",
    state_fields=[
        StateField("tick", "Tick", "number"),
        StateField("room", "Room"),
        StateField("inventory", "Inventory", "list"),
        StateField("action", "Action"),
        StateField("reward", "Step Reward", "reward"),
        StateField("total_reward", "Total Reward", "reward"),
        StateField("done", "Done"),
    ],
    presets=[
        # --- existing ---
        PresetQuery(
            "Episode path",
            "MATCH p=(s:State {context:'spatial', tick:0})-[:LEADS_TO*]->(e:State {context:'spatial', done:true}) "
            "RETURN [n IN nodes(p) | n.room_id] AS path",
        ),
        PresetQuery(
            "Key locations",
            "MATCH (s:State {context:'spatial'})-[:HAS]->(e:Entity {name:'key', context:'spatial'}) "
            "RETURN s.tick AS tick, s.room_id AS room",
        ),
        PresetQuery(
            "Locked door obs",
            "MATCH (o:Observation {context:'spatial'})-[:CONCERNS]->(e:Entity {locked:true, context:'spatial'}) "
            "RETURN o.tick AS tick, e.name AS entity",
        ),
        PresetQuery(
            "Shortest path",
            "MATCH p=shortestPath((a:State {room_id:'Kitchen', context:'spatial'})"
            "-[:LEADS_TO*]->(b:State {room_id:'Garden', context:'spatial'})) "
            "RETURN [n IN nodes(p) | n.room_id] AS rooms",
        ),
        PresetQuery(
            "State count",
            "MATCH (s:State {context:'spatial'}) RETURN count(s) AS states",
        ),
        # --- learned transition dynamics ---
        PresetQuery(
            "Predicted next (Kitchen N)",
            "MATCH (r:Room {id:'Kitchen', context:'spatial'})-[t:TRANSITION {action:'move_north'}]->(next:Room) "
            "RETURN next.id AS next_room, t.visit_count AS visits, t.r_mean AS avg_reward "
            "ORDER BY t.visit_count DESC",
        ),
        PresetQuery(
            "Workshop transitions",
            "MATCH (r:Room {id:'Workshop', context:'spatial'})-[t:TRANSITION]->(next:Room) "
            "RETURN t.action AS action, next.id AS next_room, t.visit_count AS visits, t.r_mean AS r_mean "
            "ORDER BY t.action, t.visit_count DESC",
        ),
        # --- model confidence & exploration coverage ---
        PresetQuery(
            "Top transitions",
            "MATCH (a:Room {context:'spatial'})-[t:TRANSITION]->(b:Room) "
            "RETURN a.id AS from_room, t.action AS action, b.id AS to_room, t.visit_count AS visits "
            "ORDER BY t.visit_count DESC LIMIT 20",
        ),
        PresetQuery(
            "Blind spots",
            "MATCH (r:Room {context:'spatial'})-[:ADJACENT]->(neighbor:Room) "
            "WHERE NOT EXISTS { MATCH (r)-[:TRANSITION]->(neighbor) } "
            "RETURN r.id AS room, neighbor.id AS unexplored_neighbor",
        ),
        PresetQuery(
            "Exploration %",
            "MATCH (a:Room {context:'spatial'})-[:TRANSITION]->(b:Room) "
            "WITH count(DISTINCT [a.id, b.id]) AS learned_pairs "
            "MATCH (a:Room {context:'spatial'})-[:ADJACENT]->(b:Room) "
            "WITH learned_pairs, count(*) AS possible_pairs "
            "RETURN learned_pairs, possible_pairs, "
            "round(100.0 * learned_pairs / possible_pairs) AS pct_explored",
        ),
        PresetQuery(
            "Room familiarity",
            "MATCH (r:Room {context:'spatial'})-[t:TRANSITION]->(:Room) "
            "RETURN r.id AS room, count(t) AS known_actions, sum(t.visit_count) AS total_visits "
            "ORDER BY total_visits DESC",
        ),
        # --- path planning ---
        PresetQuery(
            "Shortest (learned)",
            "MATCH path = shortestPath("
            "(a:Room {id:'Kitchen', context:'spatial'})-[:TRANSITION*]->(b:Room {id:'Garden', context:'spatial'})) "
            "RETURN [r IN nodes(path) | r.id] AS route, length(path) AS hops",
        ),
        PresetQuery(
            "Routes \u22645 hops",
            "MATCH path = (a:Room {id:'Kitchen', context:'spatial'})-[:TRANSITION*1..5]->"
            "(b:Room {id:'Garden', context:'spatial'}) "
            "RETURN [r IN nodes(path) | r.id] AS route, length(path) AS hops "
            "ORDER BY hops LIMIT 5",
        ),
        PresetQuery(
            "Action sequence",
            "MATCH path = shortestPath("
            "(a:Room {id:'Kitchen', context:'spatial'})-[:TRANSITION*]->(b:Room {id:'Garden', context:'spatial'})) "
            "RETURN [rel IN relationships(path) | rel.action] AS action_sequence, length(path) AS hops",
        ),
        # --- reward structure ---
        PresetQuery(
            "Rewarding steps",
            "MATCH (a:Room {context:'spatial'})-[t:TRANSITION]->(b:Room) "
            "WHERE t.r_mean <> -0.1 "
            "RETURN a.id AS from_room, t.action AS action, b.id AS to_room, "
            "t.r_mean AS avg_reward, t.visit_count AS visits",
        ),
        PresetQuery(
            "Path reward",
            "MATCH path = shortestPath("
            "(a:Room {id:'Kitchen', context:'spatial'})-[:TRANSITION*]->(b:Room {id:'Garden', context:'spatial'})) "
            "WITH relationships(path) AS edges UNWIND edges AS e "
            "RETURN sum(e.r_mean) AS expected_total_reward, count(e) AS steps",
        ),
        PresetQuery(
            "Top rewards",
            "MATCH (a:Room {context:'spatial'})-[t:TRANSITION]->(b:Room) "
            "RETURN a.id AS from_room, t.action, b.id AS to_room, t.r_mean "
            "ORDER BY t.r_mean DESC LIMIT 5",
        ),
        # --- trajectory analysis ---
        PresetQuery(
            "Trajectory",
            "MATCH (s:State {context:'spatial'}) "
            "RETURN s.tick AS tick, s.room_id AS room, s.inventory AS inventory, s.done AS done "
            "ORDER BY s.tick",
        ),
        PresetQuery(
            "Room visits",
            "MATCH (s:State {context:'spatial'})-[:IN_ROOM]->(r:Room) "
            "RETURN r.id AS room, count(s) AS visits ORDER BY visits DESC",
        ),
        PresetQuery(
            "First-visit tick",
            "MATCH (s:State {context:'spatial'})-[:IN_ROOM]->(r:Room) "
            "WITH r.id AS room, min(s.tick) AS first_visit "
            "RETURN room, first_visit ORDER BY first_visit",
        ),
        PresetQuery(
            "State trajectory",
            "MATCH (s:State {context:'spatial'})-[l:LEADS_TO]->(n:State) "
            "RETURN s.tick AS tick, s.room_id AS from_room, l.via_action AS action, "
            "n.room_id AS to_room, l.reward AS reward ORDER BY s.tick",
        ),
        # --- entity interactions ---
        PresetQuery(
            "Key pickup",
            "MATCH (s:State {context:'spatial'})-[:TRIGGERS]->"
            "(a:Action {type:'pick_up', target_id:'key_1'}) "
            "RETURN s.tick AS tick, s.room_id AS room",
        ),
        PresetQuery(
            "Holding key",
            "MATCH (s:State {context:'spatial'}) WHERE 'key_1' IN s.inventory "
            "RETURN s.tick AS tick, s.room_id AS room ORDER BY s.tick",
        ),
        PresetQuery(
            "Empty-handed rooms",
            "MATCH (s:State {context:'spatial'})-[:IN_ROOM]->(r:Room) "
            "WHERE size(s.inventory) = 0 "
            "RETURN DISTINCT r.id AS room, count(s) AS ticks_visited "
            "ORDER BY ticks_visited DESC",
        ),
        # --- cross-cutting ---
        PresetQuery(
            "Model vs adjacency",
            "MATCH (a:Room {context:'spatial'})-[:ADJACENT]->(b:Room) "
            "WITH a, b, EXISTS { MATCH (a)-[:TRANSITION]->(b) } AS learned "
            "RETURN a.id AS from_room, b.id AS to_room, learned ORDER BY learned, a.id",
        ),
        PresetQuery(
            "Agent room path",
            "MATCH (initial:State {tick:0, context:'spatial'}) "
            "MATCH path = (initial)-[:LEADS_TO*]->(end:State) "
            "WHERE NOT (end)-[:LEADS_TO]->() "
            "RETURN [n IN nodes(path) | n.room_id] AS room_sequence, length(path) AS steps",
        ),
        PresetQuery(
            "Reachable \u22643 (Kitchen)",
            "MATCH (a:Room {id:'Kitchen', context:'spatial'})-[:TRANSITION*1..3]->(b:Room) "
            "RETURN DISTINCT b.id AS reachable_room ORDER BY b.id",
        ),
    ],
    legend=[
        LegendEntry("State",       "#8b8fa3", "dot"),
        LegendEntry("Action",      "#c084fc", "dot"),
        LegendEntry("Observation", "#5eead4", "dot"),
        LegendEntry("Room",        "#ffffff", "dot"),
        LegendEntry("Leads to",    "#6c8cff", "line"),
        LegendEntry("Adjacent",    "#6b7280", "dash"),
        LegendEntry("Transition",  "#ffffff", "line"),
    ],
)


# ---------------------------------------------------------------------------
# Scene graph — enlarged Vault interior puzzle
# ---------------------------------------------------------------------------


SCENE = ContextDef(
    key="scene",
    label="Scene Graph",
    central_view="scene",
    state_fields=[
        StateField("tick",         "Tick", "number"),
        StateField("waypoint",     "Waypoint"),
        StateField("has_key",      "Has Key"),
        StateField("door",         "Door"),
        StateField("phase",        "Phase"),
        StateField("action",       "Action"),
        StateField("reward",       "Step Reward",  "reward"),
        StateField("total_reward", "Total Reward", "reward"),
        StateField("done",         "Done"),
    ],
    presets=[
        PresetQuery(
            "Winning escape paths",
            "MATCH p=(s:State {context:'scene', tick:0})-[:LEADS_TO*]->(e:State {context:'scene', done:true}) "
            "RETURN [n IN nodes(p) | n.room_id] AS waypoints, length(p) AS steps",
        ),
        PresetQuery(
            "Key-pickup timing",
            "MATCH (s:State {context:'scene'}) "
            "WHERE 'key_v' IN s.inventory "
            "RETURN s.tick AS tick, s.room_id AS at "
            "ORDER BY s.tick LIMIT 5",
        ),
        PresetQuery(
            "Per-cell visit counts",
            "MATCH (s:State {context:'scene'})-[:IN_ROOM]->(w:Waypoint {context:'scene'}) "
            "RETURN w.cell AS cell, count(s) AS visits "
            "ORDER BY visits DESC",
        ),
        PresetQuery(
            "Dead-end waypoints reached",
            "MATCH (s:State {context:'scene'})-[:IN_ROOM]->(w:Waypoint {context:'scene', role:'dead_end'}) "
            "RETURN w.name AS dead_end, count(s) AS times "
            "ORDER BY times DESC",
        ),
        PresetQuery(
            "Locked-door collisions",
            "MATCH (s:State {context:'scene'})-[:TRIGGERS]->(a:Action {context:'scene'}) "
            "WHERE s.room_id = 'vp_vault_e' AND a.direction = 'east' AND NOT 'key_v' IN s.inventory "
            "RETURN s.tick AS tick",
        ),
        PresetQuery(
            "What does the door guard?",
            "MATCH (d:Entity {context:'scene', id:'door_v'})-[:GUARDS]->(w:Waypoint {context:'scene'}) "
            "RETURN w.name AS blocks, w.cell AS cell",
        ),
        PresetQuery(
            "Best-return transitions",
            "MATCH (a:Waypoint {context:'scene'})-[t:TRANSITION {context:'scene'}]->(b:Waypoint {context:'scene'}) "
            "RETURN a.name AS from, t.action AS action, b.name AS to, "
            "       t.visit_count AS visits, t.r_mean AS r_mean "
            "ORDER BY t.r_mean DESC LIMIT 10",
        ),
        PresetQuery(
            "All scene waypoints",
            "MATCH (w:Waypoint {context:'scene'}) "
            "RETURN w.cell AS cell, w.id AS id, w.name AS name, w.role AS role "
            "ORDER BY w.cell, w.y, w.x",
        ),
    ],
    legend=[
        LegendEntry("Waypoint",    "#a78bfa", "dot"),
        LegendEntry("Box",         "#a16207", "dot"),
        LegendEntry("Key",         "#fbbf24", "dot"),
        LegendEntry("Door",        "#f87171", "dot"),
        LegendEntry("State",       "#8b8fa3", "dot"),
        LegendEntry("Action",      "#c084fc", "dot"),
        LegendEntry("Observation", "#5eead4", "dot"),
        LegendEntry("Path",        "#a78bfa", "line"),
        LegendEntry("Contains",    "#a16207", "line"),
        LegendEntry("Unlocks",     "#fbbf24", "line"),
        LegendEntry("Guards",      "#f87171", "line"),
        LegendEntry("Transition",  "#ffffff", "line"),
        LegendEntry("Leads to",    "#6c8cff", "line"),
    ],
)


# ---------------------------------------------------------------------------
# Generalised environment (placeholder)
# ---------------------------------------------------------------------------


GENERALISED = ContextDef(
    key="generalised",
    label="Generalised Environment",
    central_view="generalised",
    state_fields=[
        StateField("tick",             "Tick",             "number"),
        StateField("current_node",     "Current Node"),
        StateField("last_call",        "Last Call"),
        StateField("human_state",      "Human"),
        StateField("orch_state",       "Orchestrator"),
        StateField("search_state",     "Search Agent"),
        StateField("memory_state",     "Memory Agent"),
        StateField("artefact",         "Artefact"),
        StateField("primary_endpoint", "Endpoint"),
        StateField("sample_size",      "Sample Size",      "number"),
        StateField("reward",           "Step Reward",      "reward"),
        StateField("total_reward",     "Total Reward",     "reward"),
        StateField("done",             "Done"),
    ],
    presets=[
        # --- episode trajectory ---
        PresetQuery(
            "Full trajectory",
            "MATCH (s:State {context:'generalised'})-[l:LEADS_TO]->(n:State {context:'generalised'}) "
            "RETURN s.tick AS tick, s.room_id AS from_state, l.call_type AS call, "
            "n.room_id AS to_state, l.reward AS reward ORDER BY s.tick",
        ),
        PresetQuery(
            "Visited agent states",
            "MATCH (s:State {context:'generalised'}) "
            "WHERE s.room_id STARTS WITH 'orch:' OR s.room_id STARTS WITH 'search:' OR s.room_id STARTS WITH 'memory:' "
            "RETURN s.room_id AS state, count(*) AS visits "
            "ORDER BY visits DESC",
        ),
        PresetQuery(
            "Working state",
            "MATCH (w:WorkingState {context:'generalised'}) "
            "RETURN w.task AS task, w.primary_endpoint AS endpoint, "
            "w.effect_size AS effect_size, w.sample_size AS sample_size, w.status AS status",
        ),
        # --- learned causal effects of tool calls (the headline preset) ---
        PresetQuery(
            "Tool effects (orchestrator)",
            "MATCH (a:Room {context:'generalised'})-[t:TRANSITION]->(b:Room {context:'generalised'}) "
            "WHERE a.id STARTS WITH 'orch:' "
            "RETURN a.id AS from_state, t.action AS tool_call, b.id AS to_state, "
            "t.visit_count AS visits, t.r_mean AS avg_reward "
            "ORDER BY t.r_mean DESC",
        ),
        PresetQuery(
            "Tool effects (search)",
            "MATCH (a:Room {context:'generalised'})-[t:TRANSITION]->(b:Room {context:'generalised'}) "
            "WHERE a.id STARTS WITH 'search:' "
            "RETURN a.id AS from_state, t.action AS tool_call, b.id AS to_state, "
            "t.visit_count AS visits, t.r_mean AS avg_reward "
            "ORDER BY t.r_mean DESC",
        ),
        PresetQuery(
            "Tool effects (memory)",
            "MATCH (a:Room {context:'generalised'})-[t:TRANSITION]->(b:Room {context:'generalised'}) "
            "WHERE a.id STARTS WITH 'memory:' "
            "RETURN a.id AS from_state, t.action AS tool_call, b.id AS to_state, "
            "t.visit_count AS visits, t.r_mean AS avg_reward "
            "ORDER BY t.r_mean DESC",
        ),
        # --- causal failures ---
        PresetQuery(
            "Hallucinated transitions",
            "MATCH (a:Room {context:'generalised'})-[t:TRANSITION]->(b:Room {context:'generalised'}) "
            "WHERE b.id CONTAINS 'hallucinated' OR b.id ENDS WITH '!' "
            "RETURN a.id AS from_state, t.action AS tool_call, b.id AS to_state, "
            "t.visit_count AS visits, t.r_mean AS avg_reward "
            "ORDER BY t.visit_count DESC",
        ),
        PresetQuery(
            "Penalty transitions",
            "MATCH (a:Room {context:'generalised'})-[t:TRANSITION]->(b:Room {context:'generalised'}) "
            "WHERE t.r_mean < 0 "
            "RETURN a.id AS from_state, t.action AS tool_call, b.id AS to_state, "
            "t.r_mean AS avg_reward, t.visit_count AS visits "
            "ORDER BY t.r_mean ASC",
        ),
        # --- topology & structure ---
        PresetQuery(
            "Call topology",
            "MATCH (a:Actor {context:'generalised'})-[c:CALLS]->(b:Actor {context:'generalised'}) "
            "RETURN a.id AS from_actor, c.action AS call_type, b.id AS to_actor "
            "ORDER BY a.id",
        ),
        PresetQuery(
            "Agent toolshelves",
            "MATCH (a:Actor {context:'generalised'})-[:CALLS]->(b:Actor {context:'generalised'}) "
            "WHERE a.kind = 'agent' "
            "RETURN a.id AS agent, collect(b.id) AS callable "
            "ORDER BY a.id",
        ),
        PresetQuery(
            "Hops to LLM per agent",
            "MATCH (s:State {context:'generalised'})-[:TRIGGERS]->(a:Action {context:'generalised'}) "
            "WHERE a.target_id = 'llm' "
            "RETURN s.room_id AS from_state, count(*) AS llm_calls "
            "ORDER BY llm_calls DESC",
        ),
    ],
    legend=[
        LegendEntry("Human",            "#fbbf24", "dot"),
        LegendEntry("Client",           "#60a5fa", "dot"),
        LegendEntry("Server",           "#a78bfa", "dot"),
        LegendEntry("Agent",            "#34d399", "dot"),
        LegendEntry("LLM",              "#f472b6", "dot"),
        LegendEntry("Tool",             "#22d3ee", "dot"),
        LegendEntry("Agent state",      "#94a3b8", "dot"),
        LegendEntry("Hallucinated",     "#f87171", "dot"),
        LegendEntry("Refined / done",   "#4ade80", "dot"),
        LegendEntry("State (per tick)", "#8b8fa3", "dot"),
        LegendEntry("Action",           "#c084fc", "dot"),
        LegendEntry("Observation",      "#5eead4", "dot"),
        LegendEntry("Calls (topology)", "#6b7280", "dash"),
        LegendEntry("Leads to",         "#6c8cff", "line"),
        LegendEntry("Transition",       "#ffffff", "line"),
        LegendEntry("At (state\u2192actor)", "#94a3b8", "line"),
        LegendEntry("Targets (action\u2192actor)", "#c084fc", "line"),
        LegendEntry("Of (agent state\u2192actor)", "#475569", "line"),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CONTEXT_REGISTRY: dict[str, ContextDef] = {
    "spatial": SPATIAL,
    "scene": SCENE,
    "generalised": GENERALISED,
}


def get_context(key: str) -> ContextDef:
    if key not in CONTEXT_REGISTRY:
        raise ValueError(f"Unknown context: {key}")
    return CONTEXT_REGISTRY[key]


# ---------------------------------------------------------------------------
# Legend mapping used by the frontend graph renderer
# ---------------------------------------------------------------------------
#
# The GraphView uses node.label and edge.rel_type to colour primitives. Each
# context tells the view which node labels and relationship types exist in
# that subgraph so it can pick colours from the context's legend.

NODE_STYLE: dict[str, dict[str, str]] = {
    # spatial (existing)
    "State":          {"color": "#8b8fa3"},
    "Action":         {"color": "#c084fc"},
    "Observation":    {"color": "#5eead4"},
    "Room":           {"color": "#ffffff"},
    "Entity":         {"color": "#22d3ee"},
    # scene
    "Object":         {"color": "#34d399"},
    "Part":           {"color": "#a78bfa"},
    "Attribute":      {"color": "#fbbf24"},
    # generalised (placeholder Abstract* kept for backwards compat with any
    # leftover data — real episode nodes are :Actor + :WorkingState)
    "AbstractState":  {"color": "#38bdf8"},
    "AbstractAction": {"color": "#f472b6"},
    "Actor":          {"color": "#34d399"},
    "WorkingState":   {"color": "#fbbf24"},
}

EDGE_STYLE: dict[str, dict[str, Any]] = {
    # spatial (existing)
    "LEADS_TO":       {"color": "#6c8cff", "width": 1.8, "opacity": 0.8},
    "TRIGGERS":       {"color": "#c084fc", "width": 1.0, "opacity": 0.5},
    "PRODUCES":       {"color": "#22d3ee", "width": 1.0, "opacity": 0.5},
    "HAS":            {"color": "#4ade80", "width": 0.6, "opacity": 0.3},
    "CONCERNS":       {"color": "#22d3ee", "width": 0.5, "opacity": 0.25},
    "TRANSITION":     {"color": "#ffffff", "width": 1.5, "opacity": 0.6},
    "IN_ROOM":        {"color": "#fbbf24", "width": 0.6, "opacity": 0.35},
    "ADJACENT":       {"color": "#6b7280", "width": 0.5, "opacity": 0.15, "dashed": True},
    # scene
    "PART_OF":        {"color": "#a78bfa", "width": 1.2, "opacity": 0.7},
    "SUPPORTS":       {"color": "#34d399", "width": 1.2, "opacity": 0.7},
    "HAS_ATTRIBUTE":  {"color": "#fbbf24", "width": 0.8, "opacity": 0.5},
    # generalised (placeholder TRANSITIONS_TO / INDUCES kept for backwards
    # compat; real generalised edges are :CALLS on the static topology)
    "TRANSITIONS_TO": {"color": "#38bdf8", "width": 1.5, "opacity": 0.7},
    "INDUCES":        {"color": "#f472b6", "width": 1.0, "opacity": 0.6},
    "CALLS":          {"color": "#6b7280", "width": 0.8, "opacity": 0.35, "dashed": True},
}
