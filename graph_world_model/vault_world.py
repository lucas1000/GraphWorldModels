"""Scene-graph demo — higher-detail view of the spatial grid's southeast corner.

Four adjacent rooms form a T-shape:

::

                 ┌──────────┐
                 │  Trophy  │     (key inside)
                 │    N     │
                 │  W C E   │
                 │    S     │
                 └────┬─────┘
                      │
     ┌──────────┐  ┌──┴───────┐  ┌──────────┐
     │  Crypt   │  │  Vault   │  │  Garden  │
     │    N     │  │    N     │  │    N     │
     │  W C E───┼──┤W  C  E───╫──┤W  C  E   │
     │    S     │  │    S     │  │    S     │
     └──────────┘  └──────────┘  └──────────┘
                                 (door guards
                                 Vault.E↔Garden.W)

Each room has N/S/E/W entry/exit waypoints around a Center. Moves are
cardinal only — corners are walls, so diagonal traversal is impossible.
Crypt, Vault, and Garden are in a row; Trophy Room sits north of Vault,
connected via Vault.N ↔ Trophy.S.

The agent starts at Crypt.Center. The key sits at **Trophy.Center** —
off the direct east-to-Garden path. A locked door bars passage between
Vault.E and Garden.W. Winning sequence: agent must head east to Vault,
**detour north** to Trophy Room for the key, double back to Vault, then
continue east through the door to Garden.

Waypoints carry both ``:Waypoint`` and ``:Room`` labels so the spatial
world-model's ``:Room`` queries work unchanged.
"""

from __future__ import annotations

import random
import uuid
from collections import deque
from typing import Optional

from .graph_store import GraphStore
from .policies import BFSPolicy, ModelBasedPolicy
from .world import (
    Action,
    ActionType,
    Direction,
    Entity,
    Observation,
    StateSnapshot,
    StepResult,
)


# Waypoint grid. Crypt/Vault/Garden share row y=0..2; Trophy Room sits
# above Vault at y=-3..-1. All four rooms are 3x3 with N/C/S and W/C/E
# pairs only (corners are not waypoints).
VAULT_LAYOUT: dict[tuple[int, int], str] = {
    # Crypt cell (west of hub)
    (1, 0): "vp_crypt_n",
    (0, 1): "vp_crypt_w",
    (1, 1): "vp_crypt_c",
    (2, 1): "vp_crypt_e",
    (1, 2): "vp_crypt_s",
    # Vault cell (hub)
    (4, 0): "vp_vault_n",
    (3, 1): "vp_vault_w",
    (4, 1): "vp_vault_c",
    (5, 1): "vp_vault_e",
    (4, 2): "vp_vault_s",
    # Garden cell (east of hub, behind locked door)
    (7, 0): "vp_garden_n",
    (6, 1): "vp_garden_w",
    (7, 1): "vp_garden_c",
    (8, 1): "vp_garden_e",
    (7, 2): "vp_garden_s",
    # Trophy Room (north of hub — key lives here, off the shortest path)
    (4, -3): "vp_trophy_n",
    (3, -2): "vp_trophy_w",
    (4, -2): "vp_trophy_c",
    (5, -2): "vp_trophy_e",
    (4, -1): "vp_trophy_s",
}

VAULT_LABELS: dict[str, str] = {
    "vp_crypt_n":   "Crypt · N",
    "vp_crypt_w":   "Crypt · W",
    "vp_crypt_c":   "Crypt · Center",
    "vp_crypt_e":   "Crypt · E",
    "vp_crypt_s":   "Crypt · S",
    "vp_vault_n":   "Vault · N",
    "vp_vault_w":   "Vault · W",
    "vp_vault_c":   "Vault · Center",
    "vp_vault_e":   "Vault · E",
    "vp_vault_s":   "Vault · S",
    "vp_garden_n":  "Garden · N",
    "vp_garden_w":  "Garden · W",
    "vp_garden_c":  "Garden · Center",
    "vp_garden_e":  "Garden · E",
    "vp_garden_s":  "Garden · S",
    "vp_trophy_n":  "Trophy · N",
    "vp_trophy_w":  "Trophy · W",
    "vp_trophy_c":  "Trophy · Center",
    "vp_trophy_e":  "Trophy · E",
    "vp_trophy_s":  "Trophy · S",
}

VAULT_ROLES: dict[str, str] = {
    "vp_crypt_c":  "start",
    "vp_crypt_e":  "passage",   # east into Vault
    "vp_vault_w":  "passage",
    "vp_vault_c":  "hub",
    "vp_vault_n":  "passage",   # north to Trophy
    "vp_trophy_s": "passage",
    "vp_trophy_c": "key",
    "vp_vault_e":  "passage",   # east through the door
    "vp_garden_w": "passage",
    "vp_garden_c": "goal",
    # Everything else is a dead-end leaf.
    "vp_crypt_n":  "dead_end",
    "vp_crypt_s":  "dead_end",
    "vp_crypt_w":  "dead_end",
    "vp_vault_s":  "dead_end",
    "vp_garden_n": "dead_end",
    "vp_garden_s": "dead_end",
    "vp_garden_e": "dead_end",
    "vp_trophy_n": "dead_end",
    "vp_trophy_w": "dead_end",
    "vp_trophy_e": "dead_end",
}

VAULT_CELLS: dict[str, str] = {wp: wp.split("_")[1] for wp in VAULT_LABELS}

START_POSITION: tuple[int, int] = (1, 1)   # vp_crypt_c
KEY_POSITION:   tuple[int, int] = (4, -2)  # vp_trophy_c  (off the direct path)
GOAL_POSITION:  tuple[int, int] = (7, 1)   # vp_garden_c

# Door blocks the Vault.E ↔ Garden.W passage in either direction.
DOOR_SIDE_A: tuple[int, int] = (5, 1)   # vp_vault_e
DOOR_SIDE_B: tuple[int, int] = (6, 1)   # vp_garden_w
DOOR_PASSAGE: frozenset = frozenset((DOOR_SIDE_A, DOOR_SIDE_B))

KEY_ID  = "key_v"
DOOR_ID = "door_v"

VAULT_COORDS: dict[str, tuple[int, int]] = {wid: pos for pos, wid in VAULT_LAYOUT.items()}


def waypoint_id(x: int, y: int) -> Optional[str]:
    return VAULT_LAYOUT.get((x, y))


_CARDINAL: list[tuple[Direction, tuple[int, int]]] = [
    (Direction.NORTH, (0, -1)),
    (Direction.SOUTH, (0,  1)),
    (Direction.WEST,  (-1, 0)),
    (Direction.EAST,  (1,  0)),
]


def _neighbours(x: int, y: int) -> list[tuple[int, int]]:
    """Cardinal neighbours that exist in the vault layout."""
    return [(x + dx, y + dy) for _, (dx, dy) in _CARDINAL
            if (x + dx, y + dy) in VAULT_LAYOUT]


def _shortest_first_move(
    start: tuple[int, int],
    target: tuple[int, int],
) -> Optional[Direction]:
    """BFS on the waypoint graph. Returns the direction of the first move
    on the shortest path from ``start`` to ``target`` (or ``None`` if no
    path exists or start == target).

    Used by vault-aware policies as a non-oscillating greedy fallback —
    Manhattan-distance greedy breaks on dead-end ties (e.g. going north
    into ``vp_crypt_n`` when the key is at ``vp_trophy_c``).
    """
    if start == target or start not in VAULT_LAYOUT or target not in VAULT_LAYOUT:
        return None

    seen: set[tuple[int, int]] = {start}
    # Seed BFS with each valid first move so we can tag descendants by
    # which initial direction led to them.
    queue: deque[tuple[tuple[int, int], Direction]] = deque()
    for direction, (dx, dy) in _CARDINAL:
        nxt = (start[0] + dx, start[1] + dy)
        if nxt in VAULT_LAYOUT:
            if nxt == target:
                return direction
            seen.add(nxt)
            queue.append((nxt, direction))

    while queue:
        pos, first = queue.popleft()
        for _, (dx, dy) in _CARDINAL:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in VAULT_LAYOUT and nxt not in seen:
                if nxt == target:
                    return first
                seen.add(nxt)
                queue.append((nxt, first))
    return None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class VaultWorld:
    """Four-room scene episode. Public interface matches :class:`GridWorld`."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.tick = 0
        self.agent_x, self.agent_y = START_POSITION
        self.inventory: list[str] = []
        self.done = False

        self.entities: dict[str, Entity] = {}
        self._place_entities()

    def _place_entities(self) -> None:
        kx, ky = KEY_POSITION
        dx, dy = DOOR_SIDE_A
        key = Entity(id=KEY_ID, name="key", type="item",
                     x=kx, y=ky, pickable=True)
        door = Entity(id=DOOR_ID, name="vault_door", type="door",
                      x=dx, y=dy, pickable=False, locked=True)
        for e in (key, door):
            self.entities[e.id] = e

    def _current_waypoint(self) -> str:
        return waypoint_id(self.agent_x, self.agent_y) or "unknown"

    def _entities_here(self) -> list[Entity]:
        return [e for e in self.entities.values()
                if e.x == self.agent_x and e.y == self.agent_y]

    def _make_snapshot(self, stype: str = "step") -> StateSnapshot:
        return StateSnapshot(
            id=str(uuid.uuid4()),
            tick=self.tick,
            room_id=self._current_waypoint(),
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

    def get_state(self) -> StateSnapshot:
        stype = "initial" if self.tick == 0 else "step"
        return self._make_snapshot(stype)

    def get_entities(self) -> list[Entity]:
        return [e.copy() for e in self.entities.values()]

    def valid_actions(self) -> list[Action]:
        actions: list[Action] = []
        for d, (dx, dy) in _CARDINAL:
            if waypoint_id(self.agent_x + dx, self.agent_y + dy) is not None:
                actions.append(Action(type=ActionType.MOVE, direction=d))
        for e in self._entities_here():
            if e.pickable and e.id not in self.inventory:
                actions.append(Action(type=ActionType.PICK_UP, target_id=e.id))
        for eid in self.inventory:
            actions.append(Action(type=ActionType.DROP, target_id=eid))
        actions.append(Action(type=ActionType.OBSERVE, scope="local"))
        return actions

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is finished; call reset().")

        reward = -0.05
        if action.type == ActionType.MOVE:
            reward += self._handle_move(action)
        elif action.type == ActionType.PICK_UP:
            reward += self._handle_pickup(action)
        elif action.type == ActionType.DROP:
            reward += self._handle_drop(action)

        self.tick += 1
        stype = "terminal" if self.done else "step"
        state = self._make_snapshot(stype)
        obs = self._make_observation()
        return StepResult(state=state, observation=obs, reward=reward, done=self.done)

    def reset(self, seed: int | None = None) -> StateSnapshot:
        self.__init__(seed=seed)  # type: ignore[misc]
        return self.get_state()

    def _handle_move(self, action: Action) -> float:
        assert action.direction is not None
        dx, dy = dict(_CARDINAL)[action.direction]
        nx, ny = self.agent_x + dx, self.agent_y + dy

        if waypoint_id(nx, ny) is None:
            return -0.1

        # Door interaction: Vault.E ↔ Garden.W requires the key; first
        # successful crossing unlocks the door (bonus reward).
        crossing = frozenset(((self.agent_x, self.agent_y), (nx, ny)))
        if crossing == DOOR_PASSAGE:
            door = self.entities[DOOR_ID]
            if door.locked and KEY_ID not in self.inventory:
                return -0.5
            bonus = 0.0
            if door.locked:
                door.locked = False
                bonus = 2.0
            self.agent_x, self.agent_y = nx, ny
            return bonus

        self.agent_x, self.agent_y = nx, ny
        if (nx, ny) == GOAL_POSITION:
            self.done = True
            return 10.0
        return 0.0

    def _handle_pickup(self, action: Action) -> float:
        eid = action.target_id
        if not eid or eid not in self.entities:
            return 0.0
        ent = self.entities[eid]
        if ent.pickable and ent.x == self.agent_x and ent.y == self.agent_y:
            self.inventory.append(eid)
            return 1.0
        return 0.0

    def _handle_drop(self, action: Action) -> float:
        eid = action.target_id
        if eid and eid in self.inventory:
            self.inventory.remove(eid)
            e = self.entities.get(eid)
            if e:
                e.x, e.y = self.agent_x, self.agent_y
        return 0.0


# ---------------------------------------------------------------------------
# Subgraph builder
# ---------------------------------------------------------------------------

def build_vault_subgraph(graph: GraphStore) -> None:
    """Write the 4-room scene subgraph (waypoints, PATH edges, entity nodes
    with CONTAINS/UNLOCKS/GUARDS relationships) into Neo4j.

    Waypoints are dual-labeled ``Waypoint:Room`` so ``:Room`` queries in
    the shared world-model code path find them. Caller clears context first.
    """
    for (x, y), wid in VAULT_LAYOUT.items():
        role = VAULT_ROLES.get(wid, "dead_end")
        cell = VAULT_CELLS.get(wid, "")
        name = VAULT_LABELS.get(wid, wid)
        graph.run_write(
            """
            MERGE (w:Waypoint:Room {id: $id, context: $context})
            SET w.name = $name,
                w.x    = $x,
                w.y    = $y,
                w.role = $role,
                w.cell = $cell
            """,
            id=wid, name=name, x=x, y=y, role=role, cell=cell,
        )

    # PATH edges — both directions so Cypher traversals are natural.
    for (x, y), wid in VAULT_LAYOUT.items():
        for nx, ny in _neighbours(x, y):
            neighbour = VAULT_LAYOUT[(nx, ny)]
            graph.run_write(
                """
                MATCH (a:Waypoint {id: $a, context: $context}),
                      (b:Waypoint {id: $b, context: $context})
                MERGE (a)-[rel:PATH]->(b)
                SET rel.context = $context
                """,
                a=wid, b=neighbour,
            )

    kx, ky = KEY_POSITION
    dx, dy = DOOR_SIDE_A
    entities_to_write = [
        dict(id=KEY_ID,  name="key",        type="item", x=kx, y=ky, pickable=True,  locked=False),
        dict(id=DOOR_ID, name="vault_door", type="door", x=dx, y=dy, pickable=False, locked=True),
    ]
    for e in entities_to_write:
        graph.run_write(
            """
            MERGE (n:Entity {id: $id, context: $context})
            SET n.name     = $name,
                n.type     = $type,
                n.x        = $x,
                n.y        = $y,
                n.pickable = $pickable,
                n.locked   = $locked
            """,
            **e,
        )

    # Key HELD_AT Trophy.Center (initial location).
    graph.run_write(
        """
        MATCH (k:Entity {id: $kid, context: $context}),
              (w:Waypoint {id: 'vp_trophy_c', context: $context})
        MERGE (k)-[r:HELD_AT]->(w)
        SET r.context = $context
        """,
        kid=KEY_ID,
    )
    # Key UNLOCKS Door — static capability edge.
    graph.run_write(
        """
        MATCH (k:Entity {id: $kid, context: $context}),
              (d:Entity {id: $did, context: $context})
        MERGE (k)-[r:UNLOCKS]->(d)
        SET r.context = $context
        """,
        kid=KEY_ID, did=DOOR_ID,
    )
    for side in ("vp_vault_e", "vp_garden_w"):
        graph.run_write(
            """
            MATCH (d:Entity {id: $did, context: $context}),
                  (w:Waypoint {id: $wid, context: $context})
            MERGE (d)-[r:GUARDS]->(w)
            SET r.context = $context
            """,
            did=DOOR_ID, wid=side,
        )


# ---------------------------------------------------------------------------
# Vault-aware policies
# ---------------------------------------------------------------------------

def _bfs_move_toward(
    policy,
    state: StateSnapshot,
    target_room: str,
    world,
) -> Action:
    """Pick the first move on the BFS shortest path through the waypoint
    graph toward ``target_room``. Proper graph search, not Manhattan —
    avoids oscillating between a center and an adjacent dead-end.
    """
    target_pos = VAULT_COORDS.get(target_room)
    start_pos = (state.agent_x, state.agent_y)
    if target_pos is None:
        return policy.rng.choice(world.valid_actions())
    direction = _shortest_first_move(start_pos, target_pos)
    if direction is None:
        return policy.rng.choice(world.valid_actions())
    return Action(type=ActionType.MOVE, direction=direction)


class VaultBFSPolicy(BFSPolicy):
    """BFS policy targeting vault waypoints."""
    KEY_ROOM = "vp_trophy_c"   # key lives in Trophy Room, off the direct path
    GOAL_ROOM = "vp_garden_c"
    KEY_ID = "key_v"

    @classmethod
    def _room_coords(cls, name: str) -> tuple[int | None, int | None]:
        pos = VAULT_COORDS.get(name)
        if pos is None:
            return None, None
        return pos

    def _greedy_move(self, state, target_room, world):
        return _bfs_move_toward(self, state, target_room, world)


class VaultModelBasedPolicy(ModelBasedPolicy):
    """Model-based policy targeting vault waypoints with BFS fallback."""
    KEY_ROOM = "vp_trophy_c"
    GOAL_ROOM = "vp_garden_c"
    KEY_ID = "key_v"

    @classmethod
    def _room_coords(cls, name: str) -> tuple[int | None, int | None]:
        pos = VAULT_COORDS.get(name)
        if pos is None:
            return None, None
        return pos

    def _greedy_toward(self, state, target_room, world):
        return _bfs_move_toward(self, state, target_room, world)
