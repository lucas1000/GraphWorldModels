"""Graphical UI — FastAPI WebSocket server + Vite subprocess launcher."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


def _json_safe(obj: Any) -> Any:
    """Make Neo4j return values JSON-serialisable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # neo4j.graph.Node, Relationship, Path, spatial/temporal types
    if hasattr(obj, "items"):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "__iter__"):
        return [_json_safe(v) for v in obj]
    return str(obj)

from .world import GridWorld, ActionType, ROOM_NAMES, GRID_SIZE
from .graph_store import GraphStore
from .policies import RandomPolicy, BFSPolicy, ModelBasedPolicy, Policy
from .world_model import WorldModel
from .contexts import CONTEXT_REGISTRY, get_context
from .vault_world import (
    VaultWorld,
    build_vault_subgraph,
    VaultBFSPolicy,
    VaultModelBasedPolicy,
    VAULT_LABELS,
    KEY_ID as VAULT_KEY_ID,
    DOOR_ID as VAULT_DOOR_ID,
)
from .generalised_world import (
    GeneralisedWorld,
    GeneralisedPolicy,
    build_generalised_subgraph,
    ensure_agent_state,
    GeneralisedAction,
    WorkingState,
    CANNED_TASK,
    ACTORS as GENERALISED_ACTORS,
    agent_shelf,
)
import uuid as _uuid


VALID_CONTEXTS = ("spatial", "scene", "generalised")


# ---------------------------------------------------------------------------
# Simulation state container
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    world: GridWorld
    policy: Policy
    graph: GraphStore
    current_state: Any = None
    total_reward: float = 0.0
    steps: int = 0
    max_steps: int = 500
    tps: float = 4.0  # ticks per second
    paused: bool = True
    done: bool = False
    last_action: Any = None
    last_reward: float = 0.0
    prev_room: str | None = None  # room before the last step (for rollouts)
    seed: int = 42
    policy_name: str = "random"
    # World model
    model_active: bool = False
    model_threshold: int = 2
    world_model: WorldModel | None = None
    # Active demo context (drives which subgraph this sim reads/writes)
    context: str = "spatial"
    # Generalised-context bookkeeping (None for spatial/scene)
    working_state: WorkingState | None = None
    last_call: GeneralisedAction | None = None

    def _make_policy(self) -> Policy:
        # Pick the policy classes appropriate to the active context. Spatial
        # uses the default (spatial-target) variants; scene uses vault-aware
        # subclasses with waypoint targets; generalised picks a scripted or
        # random call-graph policy and skips the world-model planner.
        if self.context == "generalised":
            if self.model_active:
                # Model-based: agent decisions consult learned per-signature
                # TRANSITION dynamics. min_visits=1 because the demo wants the
                # model to use whatever evidence it has, even if sparse.
                wm = WorldModel(self.graph, min_visits=1)
                self.world_model = wm
                return GeneralisedPolicy(seed=self.seed, mode="model", world_model=wm)
            self.world_model = None
            mode = "random" if self.policy_name == "random" else "scripted"
            return GeneralisedPolicy(seed=self.seed, mode=mode)
        if self.context == "scene":
            bfs_cls: type[Policy] = VaultBFSPolicy
            model_cls: type[Policy] = VaultModelBasedPolicy
        elif self.context == "spatial":
            bfs_cls = BFSPolicy
            model_cls = ModelBasedPolicy
        else:
            self.world_model = None
            return RandomPolicy(seed=self.seed)

        if self.model_active:
            wm = WorldModel(self.graph, min_visits=self.model_threshold)
            self.world_model = wm
            return model_cls(wm, self.graph, seed=self.seed)
        elif self.policy_name == "bfs":
            self.world_model = None
            return bfs_cls(self.graph, seed=self.seed)
        else:
            self.world_model = None
            return RandomPolicy(seed=self.seed)

    def reset_episode(self, seed: int | None = None, policy_name: str | None = None) -> None:
        if seed is not None:
            self.seed = seed
        if policy_name is not None:
            self.policy_name = policy_name

        self.graph.set_context(self.context)
        if self.context == "generalised":
            # Wipe the per-tick log so the graph view starts clean (matches
            # the spatial/scene "reset clears everything visible" behaviour),
            # but KEEP AgentState signature Rooms and the learned TRANSITION /
            # OF edges. Those are the world model's accumulated evidence —
            # without them every fresh episode would start the rollout panel
            # at zero data and never light up red/green. AgentStates aren't
            # rendered in the graph view unless a current State references
            # them via IN_ROOM (see `_fetch_graph_data`), so this preserves
            # learning without re-cluttering the post-reset view.
            self.graph.run_write(
                """
                MATCH (n {context: $context})
                WHERE n:State OR n:Action OR n:Observation OR n:WorkingState
                DETACH DELETE n
                """
            )
        else:
            self.graph.clear_context(self.context)

        # Common reset fields
        self.total_reward = 0.0
        self.steps = 0
        self.done = False
        self.paused = True
        self.last_action = None
        self.last_reward = 0.0
        self.prev_room = None
        self.working_state = None
        self.last_call = None

        if self.context == "spatial":
            self.world = GridWorld(seed=self.seed)
            self.policy = self._make_policy()
            self.graph.create_rooms()
        elif self.context == "scene":
            self.world = VaultWorld(seed=self.seed)
            self.policy = self._make_policy()
            build_vault_subgraph(self.graph)
        elif self.context == "generalised":
            self.world = GeneralisedWorld(seed=self.seed)
            self.policy = self._make_policy()
            self.working_state = build_generalised_subgraph(self.graph)
        else:
            raise ValueError(f"Unknown context: {self.context}")

        # Shared episode bootstrap (spatial + scene). Waypoint nodes are
        # dual-labeled as :Room so link_state_room and record_transition
        # work unchanged for the vault subgraph.
        #
        # For generalised, the per-tick scaffolding (initial State, AgentState
        # signature Rooms, WorkingState document) is created lazily on the
        # first step so a fresh reset shows only the static actor topology.
        self.current_state = self.world.get_state()
        if self.context != "generalised":
            self.graph.create_state(self.current_state)
            self.graph.link_state_room(self.current_state)
            for entity in self.world.get_entities():
                self.graph.create_entity(entity)
            self.graph.link_state_entities(
                self.current_state,
                [e.id for e in self.world._entities_here()],
            )

    def switch_context(self, context: str) -> None:
        """Swap to a different demo context. Current context's graph is left
        intact in Neo4j — only the active subgraph changes."""
        if context not in VALID_CONTEXTS:
            raise ValueError(f"Unknown context: {context}")
        self.context = context
        self.reset_episode()

    def toggle_model(self, active: bool, threshold: int | None = None) -> None:
        """Toggle world model on/off and optionally update threshold."""
        self.model_active = active
        if threshold is not None:
            self.model_threshold = threshold
        # Rebuild the policy with updated model settings
        self.policy = self._make_policy()
        self.policy.reset()


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        data = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
sim: SimulationState | None = None
sim_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spatial_state_values(s: SimulationState, action_repr: dict | None) -> dict:
    """Map spatial sim state to the StateField keys defined in contexts.py."""
    st = s.current_state
    action_label = "-"
    if action_repr is not None:
        t = action_repr.get("type")
        d = action_repr.get("direction")
        tgt = action_repr.get("target_id")
        if t == "move" and d:
            action_label = f"Move({d})"
        elif t == "pick_up" and tgt:
            action_label = f"PickUp({tgt})"
        elif t == "drop" and tgt:
            action_label = f"Drop({tgt})"
        else:
            action_label = "Observe(local)"
    return {
        "tick": st.tick,
        "room": st.room_id,
        "inventory": list(st.inventory),
        "action": action_label,
        "reward": s.last_reward,
        "total_reward": s.total_reward,
        "done": "Yes" if s.done else "No",
    }


def _vault_state_values(s: SimulationState, action_repr: dict | None) -> dict:
    """Map vault sim state to the StateField keys defined for the SCENE context."""
    st = s.current_state
    door = s.world.entities.get(VAULT_DOOR_ID)
    has_key = VAULT_KEY_ID in st.inventory
    door_locked = bool(door.locked) if door else True
    if s.done:
        phase = "escaped"
    elif not has_key:
        phase = "seeking key"
    elif door_locked:
        phase = "seeking door"
    else:
        phase = "exiting"
    action_label = "-"
    if action_repr is not None:
        t = action_repr.get("type")
        d = action_repr.get("direction")
        tgt = action_repr.get("target_id")
        if t == "move" and d:
            action_label = f"Move({d})"
        elif t == "pick_up" and tgt:
            action_label = f"PickUp({tgt})"
        elif t == "drop" and tgt:
            action_label = f"Drop({tgt})"
        else:
            action_label = "Observe(local)"
    return {
        "tick": st.tick,
        "waypoint": VAULT_LABELS.get(st.room_id, st.room_id),
        "has_key": "Yes" if has_key else "No",
        "door": "Unlocked" if not door_locked else "Locked",
        "phase": phase,
        "action": action_label,
        "reward": s.last_reward,
        "total_reward": s.total_reward,
        "done": "Yes" if s.done else "No",
    }


def _generalised_state_values(s: SimulationState) -> dict:
    """Map generalised sim state to the StateField keys for the GENERALISED context.

    In the agent-tool model, ``current_node`` is the *physical* packet position
    (an actor id, not a state signature), and we additionally surface each
    agent's local state signature so the UI can show what the agent "knows".
    """
    world = s.world if isinstance(s.world, GeneralisedWorld) else None
    ws = s.working_state or WorkingState(task=CANNED_TASK)
    current_node = world.current_node if world else "human"
    orch_sig = world.orch_state.signature() if world else "orch:-----"
    search_sig = world.search_state.signature() if world else "search:idle"
    memory_sig = world.memory_state.signature() if world else "memory:idle"
    human_sig = world.human_state.signature() if world else "human:drafting"
    artefact_summary = (
        f"{world.artefact.grounded_count()}/5 grounded"
        + (f", {world.artefact.hallucinated_count()} hallucinated"
           if world and world.artefact.hallucinated_count() else "")
        if world else "—"
    )
    if s.last_call is not None:
        last_call_label = f"{s.last_call.src} → {s.last_call.dst} ({s.last_call.call_type})"
    else:
        last_call_label = "—"
    actor = GENERALISED_ACTORS.get(current_node)
    return {
        "tick": s.current_state.tick if s.current_state else 0,
        "current_node": actor.label if actor else current_node,
        "last_call": last_call_label,
        "human_state":   human_sig,
        "orch_state":    orch_sig,
        "search_state":  search_sig,
        "memory_state": memory_sig,
        "artefact":      artefact_summary,
        "primary_endpoint": ws.primary_endpoint or "—",
        "sample_size": ws.sample_size if ws.sample_size is not None else "—",
        "status": ws.status,
        "reward": s.last_reward,
        "total_reward": s.total_reward,
        "done": "Yes" if s.done else "No",
    }


def build_tick_message(s: SimulationState) -> dict:
    # Generalised context drives a real call-graph episode but doesn't fit the
    # spatial/scene action shape (no Action dataclass, no entities, no model).
    # Build a separate payload tailored to it.
    if s.context == "generalised":
        return _build_generalised_tick_message(s)

    # Spatial + scene both drive a real episode — share the payload shape.
    entities = []
    for e in s.world.entities.values():
        entities.append({
            "id": e.id,
            "name": e.name,
            "type": e.type,
            "x": e.x,
            "y": e.y,
            "pickable": e.pickable,
            "locked": e.locked,
            "held": e.id in s.world.inventory,
        })

    action_repr = None
    if s.last_action is not None:
        action_repr = {
            "type": s.last_action.type.value,
            "direction": s.last_action.direction.value if s.last_action.direction else None,
            "target_id": s.last_action.target_id,
        }

    # Get planning data from model-based policy
    planned_path: list[str] = []
    using_model = False
    action_evaluations: list[dict] = []
    if isinstance(s.policy, ModelBasedPolicy):
        planned_path = s.policy.planned_path
        using_model = s.policy.using_model
        # Serialize action evaluations so the UI can show what the model considered
        plan = s.policy._last_plan
        if plan and plan.evaluations:
            for ev in plan.evaluations:
                action_evaluations.append({
                    "action": ev.action,
                    "q_value": ev.q_value if ev.q_value != float("-inf") else None,
                    "immediate_next": ev.immediate_transition,
                    "immediate_reward": ev.immediate_reward,
                    "n_rollouts": len(ev.rollouts),
                    "chosen": ev.action == plan.chosen_action,
                })

    # Get model stats if model is active
    model_stats = None
    if s.model_active:
        try:
            model_stats = _json_safe(s.graph.get_model_stats())
        except Exception:
            model_stats = None

    if s.context == "scene":
        state_values = _vault_state_values(s, action_repr)
    else:
        state_values = _spatial_state_values(s, action_repr)

    return {
        "type": "tick",
        "context": s.context,
        "tick": s.current_state.tick,
        "agent": {
            "x": s.current_state.agent_x,
            "y": s.current_state.agent_y,
            "room": s.current_state.room_id,
        },
        "entities": entities,
        "inventory": list(s.current_state.inventory),
        "action": action_repr,
        "reward": s.last_reward,
        "total_reward": s.total_reward,
        "done": s.done,
        "last_cypher": s.graph.last_cypher.strip()[:500],
        "model_active": s.model_active,
        "model_threshold": s.model_threshold,
        "planned_path": planned_path,
        "using_model": using_model,
        "action_evaluations": action_evaluations,
        "model_stats": model_stats,
        "rollout_data": _compute_rollouts(s),
        "state_values": state_values,
    }


def _build_generalised_tick_message(s: SimulationState) -> dict:
    """Tick payload for the generalised context.

    Generalised has no spatial agent, no entities, and no Action-dataclass
    based policy, so this builds a minimal payload carrying the call-graph
    state (current actor, last hop, working-state fields) and reuses the
    shared rollout machinery.
    """
    last_call_payload = None
    if s.last_call is not None:
        last_call_payload = {
            "from": s.last_call.src,
            "to": s.last_call.dst,
            "action": s.last_call.call_type,
            "text": s.last_call.text,
        }
    working_state_payload = (
        (s.working_state or WorkingState(task=CANNED_TASK)).to_dict()
    )
    artefact_payload = (
        s.world.artefact.to_dict()
        if isinstance(s.world, GeneralisedWorld) else None
    )
    human_state_payload = (
        s.world.human_state.signature()
        if isinstance(s.world, GeneralisedWorld) else None
    )
    # World-model stats mirror the spatial/scene payload so the header
    # badge ("PLANNING | N edges") works identically in all three demos.
    model_stats = None
    if s.model_active:
        try:
            model_stats = _json_safe(s.graph.get_model_stats())
        except Exception:
            model_stats = None
    using_model_flag = (
        s.model_active
        and isinstance(s.policy, GeneralisedPolicy)
        and s.policy.mode == "model"
    )
    # ``agent.room`` historically carries the actor id (used by the canvas);
    # the state snapshot's room_id now carries the agent *state signature*
    # for world-model keying, so we route the physical actor id through
    # ``agent.room`` instead.
    physical_actor = (
        s.world.current_node if isinstance(s.world, GeneralisedWorld)
        else (s.current_state.room_id if s.current_state else "human")
    )
    return {
        "type": "tick",
        "context": s.context,
        "tick": s.current_state.tick if s.current_state else 0,
        "agent": {"x": 0, "y": 0, "room": physical_actor},
        "entities": [],
        "inventory": [],
        "action": None,
        "reward": s.last_reward,
        "total_reward": s.total_reward,
        "done": s.done,
        "last_cypher": s.graph.last_cypher.strip()[:500],
        "model_active": s.model_active,
        "model_threshold": s.model_threshold,
        "planned_path": [],
        "using_model": using_model_flag,
        "action_evaluations": [],
        "model_stats": model_stats,
        "rollout_data": _compute_rollouts(s),
        "state_values": _generalised_state_values(s),
        "last_call": last_call_payload,
        "working_state": working_state_payload,
        "artefact": artefact_payload,
        "human_state": human_state_payload,
    }


def _link_state_to_actor(graph, state_id: str, actor_id: str) -> None:
    """Create an :AT edge from a State to the Actor where the packet was
    located when that state was sampled. Bridges the per-tick subgraph to
    the static actor topology so they aren't disjoint in the graph view.
    """
    graph.run_write(
        """
        MATCH (st:State {id: $sid, context: $context}),
              (a:Actor  {id: $aid, context: $context})
        MERGE (st)-[rel:AT {context: $context}]->(a)
        """,
        sid=state_id,
        aid=actor_id,
    )


async def _do_generalised_step(s: SimulationState) -> dict:
    """Execute one call-graph hop and persist it to Neo4j.

    Writes a ``State`` / ``Action`` / ``LEADS_TO`` / ``IN_ROOM`` per tick,
    and a ``TRANSITION`` edge *only when the hop changed some agent's local
    state signature* (``StepOutcome.actor_transition``) — so TRANSITION
    edges key on per-agent state signatures (e.g. ``search:endpoint:fresh``
    → ``search:endpoint:grounded`` via ``invoke_regulatory_db``). The shared
    ``WorldModel`` then produces causally meaningful rollouts at agent
    decision points.
    """
    call: GeneralisedAction = s.policy.act(s.world, s.current_state)
    # Capture packet location BEFORE the step so the prev State's :AT edge
    # points to where the call originated, not where it ended up.
    prev_actor = s.world.current_node
    result = s.world.step(call)

    # Lazy-create the initial State on the first step so a fresh reset
    # leaves only the static actor topology in the graph view.
    if s.current_state.tick == 0:
        s.graph.create_state(s.current_state)
        if ":" in s.current_state.room_id:
            ensure_agent_state(s.graph, s.current_state.room_id)
            s.graph.link_state_room(s.current_state)
        _link_state_to_actor(s.graph, s.current_state.id, prev_actor)

    s.graph.create_state(result.state)
    if ":" in result.state.room_id:
        ensure_agent_state(s.graph, result.state.room_id)
    s.graph.link_state_room(result.state)
    _link_state_to_actor(s.graph, result.state.id, s.world.current_node)

    action_id = str(_uuid.uuid4())
    s.graph.run_write(
        """
        MATCH (st:State {id: $sid, context: $context}),
              (tgt:Actor {id: $target, context: $context})
        CREATE (a:Action {
            id:        $aid,
            tick:      $tick,
            type:      'call',
            call_type: $ct,
            target_id: $target,
            text:      $text,
            cost:      1,
            context:   $context
        })
        CREATE (st)-[rel:TRIGGERS {context: $context}]->(a)
        CREATE (a)-[tgt_rel:TARGETS {context: $context}]->(tgt)
        """,
        sid=s.current_state.id,
        aid=action_id,
        tick=s.current_state.tick,
        ct=result.call.call_type,
        target=result.call.dst,
        text=result.call.text,
    )

    # Persist the Observation that the step produced (payload text returned
    # to the caller). Matches the spatial/scene Action -[:PRODUCES]-> Observation
    # pattern so the per-tick subgraph has the same shape across all demos.
    obs = result.observation
    s.graph.run_write(
        """
        MATCH (a:Action {id: $aid, context: $context})
        CREATE (o:Observation {
            id:             $oid,
            tick:           $tick,
            sensor:         $sensor,
            goal_achieved:  $goal,
            context:        $context
        })
        CREATE (a)-[rel:PRODUCES {context: $context}]->(o)
        """,
        aid=action_id,
        oid=obs.id,
        tick=obs.tick,
        sensor=obs.sensor,
        goal=obs.goal_achieved,
    )

    s.graph.run_write(
        """
        MATCH (s1:State {id: $fid, context: $context}),
              (s2:State {id: $tid, context: $context})
        CREATE (s1)-[:LEADS_TO {
            via_action:  'call',
            call_type:   $ct,
            reward:      $r,
            probability: 1.0,
            context:     $context
        }]->(s2)
        """,
        fid=s.current_state.id,
        tid=result.state.id,
        ct=result.call.call_type,
        r=result.reward,
    )

    transition_delta: dict | None = None
    if result.actor_transition is not None:
        agent_id, from_sig, to_sig, shelf_key = result.actor_transition
        ensure_agent_state(s.graph, from_sig)
        ensure_agent_state(s.graph, to_sig)
        transition = s.graph.record_transition(
            from_room=from_sig,
            to_room=to_sig,
            action=shelf_key,
            reward=result.reward,
        )
        transition_delta = {
            "source": from_sig,
            "target": to_sig,
            "rel_type": "TRANSITION",
            "action": shelf_key,
            "agent": agent_id,
            "visit_count": _json_safe(transition["visit_count"]),
            "r_mean": _json_safe(transition["r_mean"]),
        }

    # Persist WorkingState mutations so cypher-console queries see them.
    # MERGE so the document is created on first step rather than on reset
    # (keeps the post-reset graph view limited to the actor topology).
    ws = result.working_state
    s.graph.run_write(
        """
        MERGE (w:WorkingState {id: 'working_state', context: $context})
        SET w.task             = $task,
            w.primary_endpoint = $pe,
            w.effect_size      = $es,
            w.sample_size      = $ss,
            w.status           = $status
        """,
        task=ws.task,
        pe=ws.primary_endpoint,
        es=ws.effect_size,
        ss=ws.sample_size,
        status=ws.status,
    )

    graph_delta = _build_generalised_graph_delta(
        s.current_state, result, action_id, transition_delta,
        prev_actor=prev_actor,
        new_actor=s.world.current_node,
        obs_id=obs.id,
        obs_tick=obs.tick,
        obs_sensor=obs.sensor,
        obs_goal=obs.goal_achieved,
    )

    s.prev_room = s.current_state.room_id
    s.last_call = result.call
    s.last_action = None
    s.last_reward = result.reward
    s.total_reward += result.reward
    s.current_state = result.state
    s.steps += 1
    s.done = result.done
    s.working_state = result.working_state

    msg = _build_generalised_tick_message(s)
    msg["graph_delta"] = graph_delta
    return msg


def _build_generalised_graph_delta(
    prev_state,
    result,
    action_id: str,
    transition_delta: dict | None,
    *,
    prev_actor: str,
    new_actor: str,
    obs_id: str,
    obs_tick: int,
    obs_sensor: str,
    obs_goal: bool,
) -> dict:
    """Incremental nodes/edges produced by one generalised hop."""
    nodes = [
        {
            "id": result.state.id,
            "label": "State",
            "tick": result.state.tick,
            "room_id": result.state.room_id,
            "done": result.state.done,
            "stype": result.state.type,
        },
        {
            "id": action_id,
            "label": "Action",
            "tick": prev_state.tick,
            "atype": "call",
            "call_type": result.call.call_type,
            "direction": None,
            "target_id": result.call.dst,
        },
        {
            "id": obs_id,
            "label": "Observation",
            "tick": obs_tick,
            "sensor": obs_sensor,
            "goal_achieved": obs_goal,
        },
    ]
    edges = [
        {
            "source": prev_state.id,
            "target": result.state.id,
            "rel_type": "LEADS_TO",
            "reward": result.reward,
            "call_type": result.call.call_type,
        },
        {
            "source": prev_state.id,
            "target": action_id,
            "rel_type": "TRIGGERS",
        },
        {
            "source": result.state.id,
            "target": result.state.room_id,
            "rel_type": "IN_ROOM",
        },
        {
            "source": result.state.id,
            "target": new_actor,
            "rel_type": "AT",
        },
        {
            "source": action_id,
            "target": result.call.dst,
            "rel_type": "TARGETS",
        },
        {
            "source": action_id,
            "target": obs_id,
            "rel_type": "PRODUCES",
        },
    ]
    # On the very first step also surface the initial state's edges so the
    # graph view doesn't show it floating until the next periodic fetch.
    if prev_state.tick == 0:
        edges.append({
            "source": prev_state.id,
            "target": prev_actor,
            "rel_type": "AT",
        })
        if ":" in prev_state.room_id:
            edges.append({
                "source": prev_state.id,
                "target": prev_state.room_id,
                "rel_type": "IN_ROOM",
            })
    if transition_delta is not None:
        edges.append(transition_delta)
    return {"nodes": nodes, "edges": edges}


def _compute_rollouts(s: SimulationState) -> dict:
    """Compute rollouts to visualise in the Rollout panel.

    Always runs with min_visits=1 from the *previous* room (pre-move),
    because that is the room where outgoing transitions have just been
    recorded.  The current room (post-move) typically has no outgoing
    transitions yet on a first visit.

    When the model-based policy is active, the model's chosen action
    is flagged with ``chosen_by_model``.
    """
    # Use the pre-move room — that's where outgoing transitions exist.
    # Fall back to current room on the very first tick (no prev_room yet).
    room = s.prev_room or s.current_state.room_id

    # Per-context candidate action set:
    #  - spatial / scene: the four cardinal moves (let evaluate_action handle
    #    unknown ones — it returns -inf and the panel marks them blank)
    #  - generalised: the agent tool shelf. We key rollouts on the *previous*
    #    agent-state signature (s.prev_room) — matching spatial/scene —
    #    because TRANSITION edges were just recorded outgoing from it. The
    #    current agent signature is typically brand-new on arrival and would
    #    always start grey.
    if s.context == "generalised":
        world = s.world if isinstance(s.world, GeneralisedWorld) else None
        if world is None:
            return {"current_room": room, "evaluations": []}
        # Only compute rollouts when the room is an agent state signature
        # (prefix ∈ {orch, search, memory}). Infra/tool hops have no shelf.
        if not room or ":" not in room:
            return {"current_room": room, "evaluations": [], "decision_point": False}
        prefix = room.split(":", 1)[0]
        agent_id_by_prefix = {
            "orch": "orchestrator",
            "search": "search_agent",
            "memory": "memory_agent",
        }
        agent_id = agent_id_by_prefix.get(prefix)
        if agent_id is None:
            return {"current_room": room, "evaluations": [], "decision_point": False}
        all_directions = list(agent_shelf(agent_id))
        if not all_directions:
            return {"current_room": room, "evaluations": [], "decision_point": False}
    else:
        all_directions = ["move_north", "move_south", "move_east", "move_west"]

    viz_model = WorldModel(s.graph, min_visits=1)

    # Which action did the actual model policy choose (if active)?
    policy_chosen: str | None = None
    if isinstance(s.policy, ModelBasedPolicy):
        plan = s.policy._last_plan
        if plan and plan.chosen_action:
            policy_chosen = plan.chosen_action
    elif (
        isinstance(s.policy, GeneralisedPolicy)
        and s.policy.mode == "model"
        and s.policy.world_model is not None
    ):
        plan = s.policy.world_model.last_plan
        if plan and plan.chosen_action:
            policy_chosen = plan.chosen_action

    evaluations: list[dict] = []
    best_q = float("-inf")
    best_action: str | None = None

    for action_key in all_directions:
        ev = viz_model.evaluate_action(room, action_key, n_rollouts=6, horizon=5)
        q = ev.q_value
        if q > best_q:
            best_q = q
            best_action = ev.action

        rollouts = []
        for ro in ev.rollouts:
            rollouts.append({
                "path": ro.path,
                "expected_return": (
                    ro.expected_return
                    if ro.expected_return != float("-inf")
                    else None
                ),
                "steps": ro.steps,
                "success": ro.success,
            })

        evaluations.append({
            "action": ev.action,
            "q_value": q if q != float("-inf") else None,
            "immediate_next": ev.immediate_transition,
            "immediate_reward": ev.immediate_reward,
            "n_rollouts": len(ev.rollouts),
            "rollouts": rollouts,
        })

    for e in evaluations:
        e["chosen"] = e["action"] == best_action and best_q != float("-inf")
        e["chosen_by_model"] = e["action"] == policy_chosen

    return {
        "current_room": room,
        "evaluations": evaluations,
        "decision_point": True,
    }


def _context_metadata_payload(context_key: str) -> dict:
    """Legend / presets / state_fields for the given context (for init msgs)."""
    return get_context(context_key).to_dict()


def _build_graph_delta(
    prev_state: Any,
    action: Any,
    new_state: Any,
    obs: Any,
    reward: float,
    step_info: dict,
    visible_entity_ids: list[str],
) -> dict:
    """Build the incremental graph nodes/edges produced by a single step."""
    action_id = step_info["action_id"]
    action_key = step_info["action_key"]
    transition = step_info["transition"]

    nodes: list[dict] = [
        {
            "id": new_state.id,
            "label": "State",
            "tick": new_state.tick,
            "room_id": new_state.room_id,
            "done": new_state.done,
            "stype": new_state.type,
        },
        {
            "id": action_id,
            "label": "Action",
            "tick": prev_state.tick,
            "atype": action.type.value,
            "direction": action.direction.value if action.direction else None,
            "target_id": action.target_id,
        },
        {
            "id": obs.id,
            "label": "Observation",
            "tick": obs.tick,
            "sensor": obs.sensor,
            "goal_achieved": obs.goal_achieved,
        },
    ]

    edges: list[dict] = [
        {
            "source": prev_state.id,
            "target": new_state.id,
            "rel_type": "LEADS_TO",
            "reward": reward,
        },
        {
            "source": prev_state.id,
            "target": action_id,
            "rel_type": "TRIGGERS",
        },
        {
            "source": action_id,
            "target": obs.id,
            "rel_type": "PRODUCES",
        },
        {
            "source": prev_state.room_id,
            "target": new_state.room_id,
            "rel_type": "TRANSITION",
            "action": action_key,
            "visit_count": _json_safe(transition["visit_count"]),
            "r_mean": _json_safe(transition["r_mean"]),
        },
        {
            "source": new_state.id,
            "target": new_state.room_id,
            "rel_type": "IN_ROOM",
        },
    ]
    for eid in visible_entity_ids:
        edges.append({"source": new_state.id, "target": eid, "rel_type": "HAS"})
    for eid in obs.visible_entities:
        edges.append({"source": obs.id, "target": eid, "rel_type": "CONCERNS"})

    return {"nodes": nodes, "edges": edges}


def _build_init_graph(s: SimulationState) -> dict:
    """Build the initial graph snapshot for the active context.

    Spatial is built from the live Python sim state (rooms, entities,
    first state). Non-spatial contexts have already written their static
    subgraph into Neo4j on reset, so we read it back filtered by context.
    """
    if s.context != "spatial":
        return _fetch_graph_data(s.graph, context=s.context)

    nodes: list[dict] = []
    edges: list[dict] = []

    for (x, y), name in ROOM_NAMES.items():
        nodes.append({"id": name, "label": "Room", "room_x": x, "room_y": y})

    for e in s.world.entities.values():
        nodes.append({
            "id": e.id,
            "label": "Entity",
            "name": e.name,
            "entity_type": e.type,
            "locked": e.locked,
        })

    st = s.current_state
    nodes.append({
        "id": st.id,
        "label": "State",
        "tick": st.tick,
        "room_id": st.room_id,
        "done": st.done,
        "stype": st.type,
    })

    for eid in [e.id for e in s.world._entities_here()]:
        edges.append({"source": st.id, "target": eid, "rel_type": "HAS"})

    # IN_ROOM: initial state → its room
    edges.append({"source": st.id, "target": st.room_id, "rel_type": "IN_ROOM"})

    # ADJACENT: static grid adjacency (one direction per pair)
    for (x, y), name in ROOM_NAMES.items():
        for dx, dy in [(1, 0), (0, 1)]:
            neighbor = ROOM_NAMES.get((x + dx, y + dy))
            if neighbor:
                edges.append({"source": name, "target": neighbor, "rel_type": "ADJACENT"})

    return {"type": "graph_data", "nodes": nodes, "edges": edges, "error": None}


async def do_step(s: SimulationState) -> dict:
    """Execute one simulation step and return the tick message."""
    if s.context == "generalised":
        return await _do_generalised_step(s)
    action = s.policy.act(s.world, s.current_state)
    result = s.world.step(action)

    visible_ids = [e.id for e in s.world._entities_here()]
    step_info = s.graph.write_step(
        prev_state=s.current_state,
        action=action,
        new_state=result.state,
        obs=result.observation,
        reward=result.reward,
        visible_entity_ids=visible_ids,
    )

    graph_delta = _build_graph_delta(
        s.current_state, action, result.state, result.observation,
        result.reward, step_info, visible_ids,
    )

    s.prev_room = s.current_state.room_id
    s.last_action = action
    s.last_reward = result.reward
    s.total_reward += result.reward
    s.current_state = result.state
    s.steps += 1
    s.done = result.done

    msg = build_tick_message(s)
    msg["graph_delta"] = graph_delta
    return msg


def _fetch_graph_data(graph: GraphStore, context: str | None = None) -> dict:
    """Query Neo4j for all nodes/edges in the given context (or active one)."""
    ctx = context if context is not None else graph.context
    node_rows = graph.run_write(
        """
        MATCH (n {context: $context})
        RETURN n.id AS id,
               CASE
                 WHEN n:Actor          THEN 'Actor'
                 WHEN n:AgentState     THEN 'AgentState'
                 WHEN n:WorkingState   THEN 'WorkingState'
                 WHEN n:Waypoint       THEN 'Waypoint'
                 WHEN n:State          THEN 'State'
                 WHEN n:Entity         THEN 'Entity'
                 WHEN n:Action         THEN 'Action'
                 WHEN n:Observation    THEN 'Observation'
                 WHEN n:Object         THEN 'Object'
                 WHEN n:Part           THEN 'Part'
                 WHEN n:Attribute      THEN 'Attribute'
                 WHEN n:AbstractState  THEN 'AbstractState'
                 WHEN n:AbstractAction THEN 'AbstractAction'
                 WHEN n:Room           THEN 'Room'
                 ELSE labels(n)[0]
               END AS label,
               CASE
                 WHEN n:Actor          THEN {name: n.name, role: n.role, room_x: n.x, room_y: n.y}
                 WHEN n:AgentState     THEN {agent: n.agent, signature: n.signature}
                 WHEN n:WorkingState   THEN {task: n.task, primary_endpoint: n.primary_endpoint, effect_size: n.effect_size, sample_size: n.sample_size, status: n.status}
                 WHEN n:Waypoint       THEN {name: n.name, room_x: n.x, room_y: n.y, role: n.role, cell: n.cell}
                 WHEN n:State          THEN {tick: n.tick, room_id: n.room_id, done: n.done, stype: n.type}
                 WHEN n:Entity         THEN {name: n.name, entity_type: n.type, locked: n.locked}
                 WHEN n:Action         THEN {tick: n.tick, atype: n.type, direction: n.direction, target_id: n.target_id, call_type: n.call_type, text: n.text}
                 WHEN n:Observation    THEN {tick: n.tick, sensor: n.sensor, goal_achieved: n.goal_achieved}
                 WHEN n:Room           THEN {room_x: n.x, room_y: n.y}
                 WHEN n:Object         THEN {name: n.name, entity_type: n.type}
                 WHEN n:Part           THEN {name: n.name}
                 WHEN n:Attribute      THEN {name: n.name, value: n.value}
                 WHEN n:AbstractState  THEN {name: n.name}
                 WHEN n:AbstractAction THEN {name: n.name}
                 ELSE {}
               END AS props
        """,
        context=ctx,
    )

    edge_rows = graph.run_write(
        """
        MATCH (a {context: $context})-[r {context: $context}]->(b {context: $context})
        RETURN a.id AS source, b.id AS target, type(r) AS rel_type,
               CASE WHEN type(r) = 'LEADS_TO'       THEN r.reward       ELSE null END AS reward,
               CASE WHEN type(r) = 'TRANSITION'     THEN r.visit_count  ELSE null END AS visit_count,
               CASE WHEN type(r) = 'TRANSITION'     THEN r.r_mean       ELSE null END AS r_mean,
               CASE WHEN type(r) = 'TRANSITION'     THEN r.action
                    WHEN type(r) = 'TRANSITIONS_TO' THEN r.action
                    ELSE null END                                       AS action,
               CASE WHEN type(r) = 'TRANSITIONS_TO' THEN r.weight       ELSE null END AS weight
        """,
        context=ctx,
    )

    nodes = []
    for row in node_rows:
        node = {"id": row["id"], "label": row["label"]}
        if row.get("props"):
            node.update(_json_safe(row["props"]))
        nodes.append(node)

    edges = []
    for row in edge_rows:
        edge = {
            "source": row["source"],
            "target": row["target"],
            "rel_type": row["rel_type"],
        }
        if row.get("reward") is not None:
            edge["reward"] = _json_safe(row["reward"])
        if row.get("visit_count") is not None:
            edge["visit_count"] = _json_safe(row["visit_count"])
        if row.get("r_mean") is not None:
            edge["r_mean"] = _json_safe(row["r_mean"])
        if row.get("action") is not None:
            edge["action"] = _json_safe(row["action"])
        if row.get("weight") is not None:
            edge["weight"] = _json_safe(row["weight"])
        edges.append(edge)

    if ctx == "generalised":
        # The world model persists AgentState scaffolding + TRANSITION / OF
        # edges across resets so rollouts have evidence to score. Only the
        # subset referenced by the current episode's State chain should be
        # visible in the graph view — everything else is model bookkeeping.
        referenced = {e["target"] for e in edges if e["rel_type"] == "IN_ROOM"}
        visible_ids = {n["id"] for n in nodes
                       if n["label"] != "AgentState" or n["id"] in referenced}
        nodes = [n for n in nodes if n["id"] in visible_ids]
        edges = [e for e in edges
                 if e["source"] in visible_ids and e["target"] in visible_ids
                 and e["rel_type"] not in ("TRANSITION", "OF")
                 or (e["rel_type"] in ("TRANSITION", "OF")
                     and e["source"] in visible_ids and e["target"] in visible_ids
                     and (e["source"] in referenced or e["target"] in referenced))]

    return {"type": "graph_data", "nodes": nodes, "edges": edges, "context": ctx, "error": None}


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

async def simulation_loop() -> None:
    global sim
    while True:
        if sim is None:
            await asyncio.sleep(0.1)
            continue
        if sim.paused or sim.done or sim.steps >= sim.max_steps:
            await asyncio.sleep(0.05)
            continue

        msg = await do_step(sim)
        await manager.broadcast(msg)
        # Push the full graph snapshot so the D3 view updates every tick
        await manager.broadcast(_fetch_graph_data(sim.graph))
        await asyncio.sleep(1.0 / sim.tps)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_task
    sim_task = asyncio.create_task(simulation_loop())
    yield
    if sim_task:
        sim_task.cancel()
        try:
            await sim_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_init_message(s: SimulationState) -> dict:
    """Assemble the 'init' payload: tick fields + init_graph + context metadata."""
    msg = build_tick_message(s)
    msg["type"] = "init"
    msg["context"] = s.context
    msg["init_graph"] = _build_init_graph(s)
    msg["context_meta"] = _context_metadata_payload(s.context)
    return msg


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    # Send initial state immediately
    if sim is not None:
        await ws.send_text(json.dumps(_build_init_message(sim)))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            await handle_client_message(ws, msg)
    except WebSocketDisconnect:
        manager.disconnect(ws)


async def handle_client_message(ws: WebSocket, msg: dict) -> None:
    global sim
    if sim is None:
        return

    msg_type = msg.get("type")

    if msg_type == "play":
        sim.paused = False
        await manager.broadcast({"type": "status", "paused": False})

    elif msg_type == "pause":
        sim.paused = True
        await manager.broadcast({"type": "status", "paused": True})

    elif msg_type == "step":
        if not sim.done and sim.steps < sim.max_steps:
            tick_msg = await do_step(sim)
            await manager.broadcast(tick_msg)
            await manager.broadcast(_fetch_graph_data(sim.graph))

    elif msg_type == "reset":
        seed = msg.get("seed", sim.seed)
        policy_name = msg.get("policy", sim.policy_name)
        sim.reset_episode(seed=seed, policy_name=policy_name)
        await manager.broadcast(_build_init_message(sim))

    elif msg_type == "environment_switch":
        view = msg.get("view", "spatial")
        if view not in VALID_CONTEXTS:
            print(f"[environment_switch] unknown context: {view!r}")
            return
        sim.switch_context(view)
        await manager.broadcast(_build_init_message(sim))

    elif msg_type == "speed":
        tps = msg.get("tps", 4)
        sim.tps = max(0.5, min(30.0, float(tps)))
        await manager.broadcast({"type": "speed", "tps": sim.tps})

    elif msg_type == "graph":
        try:
            graph_data = _fetch_graph_data(sim.graph)
            await ws.send_text(json.dumps(graph_data))
        except Exception as exc:
            await ws.send_text(json.dumps({
                "type": "graph_data",
                "nodes": [],
                "edges": [],
                "error": str(exc),
            }))

    elif msg_type == "model_toggle":
        active = msg.get("active", False)
        threshold = msg.get("threshold")
        sim.toggle_model(active, threshold)
        # Send updated tick so UI reflects new state
        tick_msg = build_tick_message(sim)
        await manager.broadcast(tick_msg)
        await manager.broadcast({
            "type": "model_status",
            "model_active": sim.model_active,
            "model_threshold": sim.model_threshold,
        })

    elif msg_type == "cypher":
        query = msg.get("query", "")
        print(f"[cypher] received query: {query!r}")
        try:
            rows = sim.graph.run_cypher(query)
            print(f"[cypher] got {len(rows)} rows")
            if rows:
                columns = list(rows[0].keys())
                row_data = [
                    [_json_safe(row.get(c)) for c in columns]
                    for row in rows[:100]
                ]
            else:
                columns = []
                row_data = []
            response = json.dumps({
                "type": "cypher_result",
                "columns": columns,
                "rows": row_data,
                "error": None,
            })
            print(f"[cypher] sending response ({len(response)} bytes)")
            await ws.send_text(response)
        except Exception as exc:
            print(f"[cypher] ERROR: {exc}")
            await ws.send_text(json.dumps({
                "type": "cypher_result",
                "columns": [],
                "rows": [],
                "error": str(exc),
            }))


# ---------------------------------------------------------------------------
# Vite subprocess management
# ---------------------------------------------------------------------------

def find_ui_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "ui"


def start_vite(ui_dir: Path, port: int = 5173) -> subprocess.Popen | None:
    """Start Vite dev server as a subprocess."""
    if not (ui_dir / "node_modules").exists():
        click.echo("Installing frontend dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=str(ui_dir),
            check=True,
        )

    click.echo(f"Starting Vite dev server on port {port}...")
    proc = subprocess.Popen(
        ["npx", "vite", "--port", str(port), "--strictPort"],
        cwd=str(ui_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--policy",
    type=click.Choice(["random", "bfs"]),
    default="random",
    help="Agent policy.",
)
@click.option("--model-threshold", default=2, type=int, help="min_visits for world model transitions.")
@click.option("--neo4j-uri", default="bolt://localhost:7687")
@click.option("--neo4j-user", default="neo4j")
@click.option("--neo4j-password", default="password")
@click.option("--max-steps", default=500, type=int)
@click.option("--port", default=8765, type=int, help="Backend server port.")
@click.option("--vite-port", default=5175, type=int, help="Vite dev server port.")
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser.")
@click.option(
    "--serve-static",
    is_flag=True,
    help="Serve pre-built static files instead of running Vite.",
)
def main(
    seed: int,
    policy: str,
    model_threshold: int,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_steps: int,
    port: int,
    vite_port: int,
    no_browser: bool,
    serve_static: bool,
) -> None:
    """Run the graph world model with a graphical browser UI."""
    global sim

    # Connect to Neo4j (starts in spatial context by default)
    graph = GraphStore(uri=neo4j_uri, user=neo4j_user, password=neo4j_password, context="spatial")
    graph.setup_schema()
    graph.clear()

    # Initialize simulation via the context-aware reset path so scene/
    # generalised share the same setup machinery.
    world = GridWorld(seed=seed)
    sim = SimulationState(
        world=world,
        policy=RandomPolicy(seed=seed),  # placeholder, rebuilt by reset_episode
        graph=graph,
        max_steps=max_steps,
        seed=seed,
        policy_name=policy,
        model_active=False,
        model_threshold=model_threshold,
        context="spatial",
    )
    sim.reset_episode()

    # Static file serving or Vite dev server
    ui_dir = find_ui_dir()
    vite_proc: subprocess.Popen | None = None

    if serve_static:
        dist_dir = ui_dir / "dist"
        if not dist_dir.exists():
            click.echo(f"Error: {dist_dir} does not exist. Run 'npm run build' in {ui_dir} first.")
            sys.exit(1)
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")
        frontend_url = f"http://localhost:{port}"
    else:
        vite_proc = start_vite(ui_dir, port=vite_port)
        frontend_url = f"http://localhost:{vite_port}"

    click.echo(f"\n  Graph World Model — Graphical UI")
    click.echo(f"  Backend:  http://localhost:{port}")
    click.echo(f"  Frontend: {frontend_url}")
    click.echo(f"  Policy:   {policy}   Seed: {seed}\n")

    if not no_browser:
        # Small delay so Vite has time to start
        import threading
        def open_browser():
            import time
            time.sleep(2)
            webbrowser.open(frontend_url)
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        if vite_proc:
            click.echo("Shutting down Vite...")
            vite_proc.terminate()
            try:
                vite_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vite_proc.kill()
        graph.close()
        click.echo("Done.")


if __name__ == "__main__":
    main()
