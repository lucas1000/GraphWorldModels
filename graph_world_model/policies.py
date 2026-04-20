"""Pluggable agent policies for the grid world."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .world import (
    Action,
    ActionType,
    Direction,
    GridWorld,
    StateSnapshot,
    room_name,
    GRID_SIZE,
)
from .graph_store import GraphStore
from .world_model import WorldModel, PlanningResult


class Policy(ABC):
    @abstractmethod
    def act(self, world: GridWorld, state: StateSnapshot) -> Action:
        ...

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------

class RandomPolicy(Policy):
    """Selects a random valid action each tick."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, world: GridWorld, state: StateSnapshot) -> Action:
        actions = world.valid_actions()
        return self.rng.choice(actions)


# ---------------------------------------------------------------------------
# BFSPolicy — navigates via the graph world model
# ---------------------------------------------------------------------------

class BFSPolicy(Policy):
    """Uses the graph store's shortest-path query to navigate toward the goal.

    Strategy:
    1. If key is not in inventory, navigate to the key's room.
    2. If key is in inventory, navigate to the Garden.
    3. Pick up items when standing on them.
    4. Falls back to random when no graph path exists yet.
    """

    # Class-level targets — subclasses override for non-spatial contexts.
    KEY_ROOM = room_name(2, 2)   # Workshop — where the key starts
    GOAL_ROOM = room_name(4, 4)  # Garden
    KEY_ID = "key_1"             # inventory id checked to switch targets

    def __init__(self, graph: GraphStore, seed: int | None = None) -> None:
        self.graph = graph
        self.rng = random.Random(seed)
        self._path: list[str] = []
        self._path_idx: int = 0

    def reset(self) -> None:
        self._path = []
        self._path_idx = 0

    def act(self, world: GridWorld, state: StateSnapshot) -> Action:
        # Pick up any pickable item in the current room
        for e in world._entities_here():
            if e.pickable and e.id not in world.inventory:
                return Action(type=ActionType.PICK_UP, target_id=e.id)

        target_room = self.GOAL_ROOM if self.KEY_ID in state.inventory else self.KEY_ROOM
        current_room = state.room_id

        if current_room == target_room:
            # Already there — might need to wait or observe
            return Action(type=ActionType.OBSERVE, scope="local")

        # Try graph-based navigation
        if not self._path or self._path_idx >= len(self._path):
            self._path = self.graph.query_shortest_path(current_room, target_room)
            self._path_idx = 1  # skip index 0 (current room)

        if self._path and self._path_idx < len(self._path):
            next_room = self._path[self._path_idx]
            direction = self._direction_to(state.agent_x, state.agent_y, next_room)
            if direction:
                self._path_idx += 1
                return Action(type=ActionType.MOVE, direction=direction)

        # Fallback: greedy move toward target coordinates
        return self._greedy_move(state, target_room, world)

    def _direction_to(self, ax: int, ay: int, target_room: str) -> Direction | None:
        """Return the direction to move from (ax, ay) toward target_room."""
        rx, ry = self._room_coords(target_room)
        if rx is None or ry is None:
            return None
        dx, dy = rx - ax, ry - ay
        if dx == 1 and dy == 0:
            return Direction.EAST
        if dx == -1 and dy == 0:
            return Direction.WEST
        if dy == 1 and dx == 0:
            return Direction.SOUTH
        if dy == -1 and dx == 0:
            return Direction.NORTH
        return None

    def _greedy_move(self, state: StateSnapshot, target_room: str, world: GridWorld) -> Action:
        """Move greedily toward the target room's coordinates (axis-priority
        heuristic — original spatial behaviour)."""
        tx, ty = self._room_coords(target_room)
        if tx is None:
            return self.rng.choice(world.valid_actions())
        dx = tx - state.agent_x
        dy = ty - state.agent_y
        if abs(dx) >= abs(dy):
            direction = Direction.EAST if dx > 0 else Direction.WEST
        else:
            direction = Direction.SOUTH if dy > 0 else Direction.NORTH
        return Action(type=ActionType.MOVE, direction=direction)

    @classmethod
    def _room_coords(cls, name: str) -> tuple[int | None, int | None]:
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if room_name(i, j) == name:
                    return i, j
        return None, None


# ---------------------------------------------------------------------------
# ModelBasedPolicy — plans using the learned world model
# ---------------------------------------------------------------------------

class ModelBasedPolicy(Policy):
    """Picks actions by running imagined rollouts through the learned world model.

    Every step:
    1. Enumerate valid move actions (move_north, move_south, etc.)
    2. For each, ask the model: "if I do this, what sequence of rooms do I
       visit and what reward do I accumulate?" (imagined rollout)
    3. Pick the action with the highest expected return.
    4. If the model has no data for any action, fall back to greedy heuristic.

    This is visibly different from random/BFS because the agent's choices
    are driven by learned transition probabilities and reward estimates —
    the same dynamics the graph view shows on TRANSITION edges.
    """

    # Class-level targets — subclasses override for non-spatial contexts.
    KEY_ROOM = room_name(2, 2)   # Workshop
    GOAL_ROOM = room_name(4, 4)  # Garden
    KEY_ID = "key_1"

    # Direction strings matching the action keys recorded in TRANSITION edges
    DIRECTION_ACTIONS = {
        Direction.NORTH: "move_north",
        Direction.SOUTH: "move_south",
        Direction.EAST:  "move_east",
        Direction.WEST:  "move_west",
    }

    def __init__(
        self,
        world_model: WorldModel,
        graph: GraphStore,
        seed: int | None = None,
    ) -> None:
        self.model = world_model
        self.graph = graph
        self.rng = random.Random(seed)
        self._last_plan: PlanningResult | None = None

    def reset(self) -> None:
        self._last_plan = None

    @property
    def planned_path(self) -> list[str]:
        """Best rollout path from the last planning step (for UI overlay)."""
        if not self._last_plan or not self._last_plan.used_model:
            return []
        # Find the best evaluation's best rollout
        best = self._last_plan.chosen_action
        for ev in self._last_plan.evaluations:
            if ev.action == best and ev.rollouts:
                # Return the rollout with highest return
                best_ro = max(ev.rollouts, key=lambda r: r.expected_return)
                return best_ro.path
        return []

    @property
    def using_model(self) -> bool:
        """Whether the last action came from model planning."""
        return self._last_plan is not None and self._last_plan.used_model

    def act(self, world: GridWorld, state: StateSnapshot) -> Action:
        # Always pick up items when standing on them
        for e in world._entities_here():
            if e.pickable and e.id not in world.inventory:
                return Action(type=ActionType.PICK_UP, target_id=e.id)

        current_room = state.room_id
        target_room = self.GOAL_ROOM if self.KEY_ID in state.inventory else self.KEY_ROOM

        if current_room == target_room:
            return Action(type=ActionType.OBSERVE, scope="local")

        # Build candidate action list from valid moves
        valid = world.valid_actions()
        move_actions: list[tuple[str, Direction]] = []
        for a in valid:
            if a.type == ActionType.MOVE and a.direction:
                action_key = self.DIRECTION_ACTIONS[a.direction]
                move_actions.append((action_key, a.direction))

        if not move_actions:
            return self.rng.choice(valid)

        # ---- Ask the world model to evaluate each action ----
        candidate_keys = [key for key, _ in move_actions]
        plan = self.model.best_action(
            current_room, candidate_keys,
            n_rollouts=6, horizon=5,
        )
        self._last_plan = plan

        if plan.used_model and plan.chosen_action:
            # Model chose an action — convert back to a Direction
            for key, direction in move_actions:
                if key == plan.chosen_action:
                    return Action(type=ActionType.MOVE, direction=direction)

        # ---- Greedy fallback: move toward target coords ----
        return self._greedy_toward(state, target_room, world)

    def _greedy_toward(
        self, state: StateSnapshot, target_room: str, world: GridWorld,
    ) -> Action:
        """Move greedily toward the target room's grid coordinates
        (axis-priority heuristic — original spatial behaviour)."""
        tx, ty = self._room_coords(target_room)
        if tx is None:
            return self.rng.choice(world.valid_actions())
        dx = tx - state.agent_x
        dy = ty - state.agent_y
        if abs(dx) >= abs(dy):
            direction = Direction.EAST if dx > 0 else Direction.WEST
        else:
            direction = Direction.SOUTH if dy > 0 else Direction.NORTH
        return Action(type=ActionType.MOVE, direction=direction)

    @classmethod
    def _room_coords(cls, name: str) -> tuple[int | None, int | None]:
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if room_name(i, j) == name:
                    return i, j
        return None, None
