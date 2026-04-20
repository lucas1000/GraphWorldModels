"""Pure-Python grid world environment with named rooms, items, and a goal."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ActionType(Enum):
    MOVE = "move"
    PICK_UP = "pick_up"
    DROP = "drop"
    OBSERVE = "observe"


class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass(frozen=True)
class Action:
    type: ActionType
    direction: Optional[Direction] = None
    target_id: Optional[str] = None
    scope: Optional[str] = None  # for Observe

    def __repr__(self) -> str:
        if self.type == ActionType.MOVE:
            return f"Move({self.direction.value})"
        if self.type == ActionType.PICK_UP:
            return f"PickUp({self.target_id})"
        if self.type == ActionType.DROP:
            return f"Drop({self.target_id})"
        return f"Observe({self.scope or 'local'})"


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    id: str
    name: str
    type: str  # "item", "door"
    x: int
    y: int
    pickable: bool = False
    locked: bool = False

    def copy(self) -> Entity:
        return Entity(
            id=self.id, name=self.name, type=self.type,
            x=self.x, y=self.y, pickable=self.pickable, locked=self.locked,
        )


# ---------------------------------------------------------------------------
# Room grid
# ---------------------------------------------------------------------------

# 5x5 named rooms laid out as (x, y) -> name
ROOM_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "Kitchen",       (1, 0): "Pantry",        (2, 0): "Dining Room",  (3, 0): "Lounge",       (4, 0): "Balcony",
    (0, 1): "Hallway",       (1, 1): "Living Room",   (2, 1): "Study",        (3, 1): "Music Room",   (4, 1): "Sunroom",
    (0, 2): "Basement",      (1, 2): "Storage",       (2, 2): "Workshop",     (3, 2): "Gallery",      (4, 2): "Conservatory",
    (0, 3): "Cellar",        (1, 3): "Wine Room",     (2, 3): "Armory",       (3, 3): "Trophy Room",  (4, 3): "Chapel",
    (0, 4): "Dungeon",       (1, 4): "Tunnel",        (2, 4): "Crypt",        (3, 4): "Vault",        (4, 4): "Garden",
}

GRID_SIZE = 5


def room_name(x: int, y: int) -> str:
    return ROOM_NAMES.get((x, y), f"Room({x},{y})")


# ---------------------------------------------------------------------------
# Observation / State snapshots
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    id: str
    tick: int
    sensor: str
    visible_entities: list[str]
    distances: list[float]
    goal_achieved: bool


@dataclass
class StateSnapshot:
    """Immutable snapshot of the world at a given tick."""
    id: str
    tick: int
    room_id: str
    agent_x: int
    agent_y: int
    inventory: list[str]
    done: bool
    type: str = "step"  # "initial", "step", "terminal"


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    state: StateSnapshot
    observation: Observation
    reward: float
    done: bool


# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------

class GridWorld:
    """A 5x5 named-room grid with items and a goal.

    Goal: reach the Garden (4, 4) while holding the key.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.tick = 0
        self.agent_x = 0
        self.agent_y = 0
        self.inventory: list[str] = []
        self.done = False

        # Entities -- placed at fixed positions for reproducibility
        self.entities: dict[str, Entity] = {}
        self._place_default_entities()

    # -- setup ---------------------------------------------------------------

    def _place_default_entities(self) -> None:
        key = Entity(id="key_1", name="key", type="item", x=2, y=2, pickable=True)
        lamp = Entity(id="lamp_1", name="lamp", type="item", x=1, y=1, pickable=True)
        door = Entity(id="door_1", name="locked_door", type="door", x=3, y=4, pickable=False, locked=True)
        for e in (key, lamp, door):
            self.entities[e.id] = e

    # -- helpers -------------------------------------------------------------

    def _current_room(self) -> str:
        return room_name(self.agent_x, self.agent_y)

    def _entities_here(self) -> list[Entity]:
        return [e for e in self.entities.values() if e.x == self.agent_x and e.y == self.agent_y]

    def _make_snapshot(self, stype: str = "step") -> StateSnapshot:
        return StateSnapshot(
            id=str(uuid.uuid4()),
            tick=self.tick,
            room_id=self._current_room(),
            agent_x=self.agent_x,
            agent_y=self.agent_y,
            inventory=list(self.inventory),
            done=self.done,
            type=stype,
        )

    def _make_observation(self) -> Observation:
        here = self._entities_here()
        return Observation(
            id=str(uuid.uuid4()),
            tick=self.tick,
            sensor="local",
            visible_entities=[e.id for e in here],
            distances=[0.0] * len(here),
            goal_achieved=self.done,
        )

    # -- public interface ----------------------------------------------------

    def get_state(self) -> StateSnapshot:
        stype = "initial" if self.tick == 0 else "step"
        return self._make_snapshot(stype)

    def get_entities(self) -> list[Entity]:
        return [e.copy() for e in self.entities.values()]

    def valid_actions(self) -> list[Action]:
        actions: list[Action] = []
        # Movement
        if self.agent_y > 0:
            actions.append(Action(type=ActionType.MOVE, direction=Direction.NORTH))
        if self.agent_y < GRID_SIZE - 1:
            actions.append(Action(type=ActionType.MOVE, direction=Direction.SOUTH))
        if self.agent_x > 0:
            actions.append(Action(type=ActionType.MOVE, direction=Direction.WEST))
        if self.agent_x < GRID_SIZE - 1:
            actions.append(Action(type=ActionType.MOVE, direction=Direction.EAST))
        # Pick up
        for e in self._entities_here():
            if e.pickable and e.id not in self.inventory:
                actions.append(Action(type=ActionType.PICK_UP, target_id=e.id))
        # Drop
        for eid in self.inventory:
            actions.append(Action(type=ActionType.DROP, target_id=eid))
        # Observe is always available
        actions.append(Action(type=ActionType.OBSERVE, scope="local"))
        return actions

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is finished; call reset().")

        reward = -0.1  # small step penalty to encourage efficiency

        if action.type == ActionType.MOVE:
            reward += self._handle_move(action)
        elif action.type == ActionType.PICK_UP:
            reward += self._handle_pickup(action)
        elif action.type == ActionType.DROP:
            reward += self._handle_drop(action)
        # Observe is a no-op beyond generating the observation

        self.tick += 1

        # Check goal: agent in Garden with key
        if self.agent_x == 4 and self.agent_y == 4 and "key_1" in self.inventory:
            self.done = True
            reward += 10.0

        stype = "terminal" if self.done else "step"
        state = self._make_snapshot(stype)
        obs = self._make_observation()
        return StepResult(state=state, observation=obs, reward=reward, done=self.done)

    def reset(self, seed: int | None = None) -> StateSnapshot:
        self.__init__(seed=seed)  # type: ignore[misc]
        return self.get_state()

    # -- action handlers -----------------------------------------------------

    def _handle_move(self, action: Action) -> float:
        assert action.direction is not None
        dx, dy = {
            Direction.NORTH: (0, -1),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
            Direction.EAST: (1, 0),
        }[action.direction]
        nx, ny = self.agent_x + dx, self.agent_y + dy

        # Check locked door blocks passage
        door = self.entities.get("door_1")
        if door and door.locked and (nx, ny) == (door.x, door.y) and "key_1" not in self.inventory:
            return -0.5  # penalty for walking into locked door

        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            # Unlock door if agent has key and moves onto it
            if door and door.locked and (nx, ny) == (door.x, door.y) and "key_1" in self.inventory:
                door.locked = False
            self.agent_x, self.agent_y = nx, ny
        return 0.0

    def _handle_pickup(self, action: Action) -> float:
        eid = action.target_id
        if eid and eid in self.entities:
            entity = self.entities[eid]
            if entity.pickable and entity.x == self.agent_x and entity.y == self.agent_y:
                self.inventory.append(eid)
                return 1.0
        return 0.0

    def _handle_drop(self, action: Action) -> float:
        eid = action.target_id
        if eid and eid in self.inventory:
            self.inventory.remove(eid)
            entity = self.entities.get(eid)
            if entity:
                entity.x, entity.y = self.agent_x, self.agent_y
        return 0.0
