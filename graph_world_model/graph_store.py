"""Neo4j graph store — writes world-model state as a growing graph.

Every node and relationship is tagged with a ``context`` property so that
multiple demo subgraphs (spatial, scene, generalised) can coexist in a single
Neo4j instance without interleaving. The active context is held on the store
and automatically applied to every write.
"""

from __future__ import annotations

import uuid
from typing import Any

from neo4j import GraphDatabase, Driver, Session

from .world import StateSnapshot, Action, Observation, Entity, ActionType, ROOM_NAMES, room_name


class GraphStore:
    """Thin wrapper around the Neo4j Python driver.

    Every public method runs parameterised Cypher — no string formatting.
    All writes tag nodes/edges with the active ``context`` so subgraphs for
    different demo contexts stay isolated in the same database.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        context: str = "spatial",
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._last_cypher: str = ""  # exposed for the demo UI
        self._context: str = context

    def close(self) -> None:
        self._driver.close()

    @property
    def last_cypher(self) -> str:
        return self._last_cypher

    @property
    def context(self) -> str:
        return self._context

    def set_context(self, context: str) -> None:
        self._context = context

    # -- helpers -------------------------------------------------------------

    def _run(self, cypher: str, **params: Any) -> list[dict]:
        self._last_cypher = cypher
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    # -- schema setup --------------------------------------------------------

    def setup_schema(self) -> None:
        # Composite (id, context) uniqueness so the same id can exist in
        # multiple contexts without collision. Safe even if only spatial ids
        # are present today.
        constraints = [
            "CREATE CONSTRAINT state_id IF NOT EXISTS FOR (s:State) REQUIRE (s.id, s.context) IS UNIQUE",
            "CREATE CONSTRAINT action_id IF NOT EXISTS FOR (a:Action) REQUIRE (a.id, a.context) IS UNIQUE",
            "CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE (o.id, o.context) IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE (e.id, e.context) IS UNIQUE",
            "CREATE CONSTRAINT room_id IF NOT EXISTS FOR (r:Room) REQUIRE (r.id, r.context) IS UNIQUE",
            "CREATE INDEX state_tick IF NOT EXISTS FOR (s:State) ON (s.tick)",
            "CREATE INDEX node_context IF NOT EXISTS FOR (n:State) ON (n.context)",
        ]
        for stmt in constraints:
            try:
                self._run(stmt)
            except Exception:
                # Tolerate a pre-existing single-property constraint on id
                # (older databases). Drop-and-retry is risky, so we skip.
                pass

    def clear(self) -> None:
        """Delete every node across all contexts (boot-time wipe)."""
        self._run("MATCH (n) DETACH DELETE n")

    def clear_context(self, context: str | None = None) -> None:
        """Delete every node/edge in the given context (defaults to active)."""
        ctx = context if context is not None else self._context
        self._run("MATCH (n {context: $ctx}) DETACH DELETE n", ctx=ctx)

    # -- entity persistence --------------------------------------------------

    def create_entity(self, entity: Entity) -> None:
        self._run(
            """
            MERGE (e:Entity {id: $id, context: $context})
            SET e.name = $name,
                e.type = $type,
                e.x    = $x,
                e.y    = $y,
                e.pickable = $pickable,
                e.locked   = $locked
            """,
            id=entity.id,
            name=entity.name,
            type=entity.type,
            x=entity.x,
            y=entity.y,
            pickable=entity.pickable,
            locked=entity.locked,
            context=self._context,
        )

    # -- state ---------------------------------------------------------------

    def create_state(self, state: StateSnapshot) -> None:
        self._run(
            """
            CREATE (s:State {
                id:        $id,
                tick:      $tick,
                type:      $type,
                room_id:   $room_id,
                agent_x:   $agent_x,
                agent_y:   $agent_y,
                inventory: $inventory,
                done:      $done,
                context:   $context
            })
            """,
            id=state.id,
            tick=state.tick,
            type=state.type,
            room_id=state.room_id,
            agent_x=state.agent_x,
            agent_y=state.agent_y,
            inventory=state.inventory,
            done=state.done,
            context=self._context,
        )

    def link_state_room(self, state: StateSnapshot) -> None:
        """Create an IN_ROOM edge from a State to its Room."""
        self._run(
            """
            MATCH (s:State {id: $sid, context: $context}),
                  (r:Room {id: $rid, context: $context})
            MERGE (s)-[rel:IN_ROOM]->(r)
            SET rel.context = $context
            """,
            sid=state.id,
            rid=state.room_id,
            context=self._context,
        )

    def link_state_entities(self, state: StateSnapshot, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            self._run(
                """
                MATCH (s:State {id: $sid, context: $context}),
                      (e:Entity {id: $eid, context: $context})
                MERGE (s)-[rel:HAS]->(e)
                SET rel.context = $context
                """,
                sid=state.id,
                eid=eid,
                context=self._context,
            )

    # -- action --------------------------------------------------------------

    def record_action(self, state: StateSnapshot, action: Action) -> str:
        action_id = str(uuid.uuid4())
        self._run(
            """
            MATCH (s:State {id: $sid, context: $context})
            CREATE (a:Action {
                id:        $aid,
                tick:      $tick,
                type:      $atype,
                direction: $direction,
                target_id: $target_id,
                cost:      1,
                context:   $context
            })
            CREATE (s)-[rel:TRIGGERS {context: $context}]->(a)
            """,
            sid=state.id,
            aid=action_id,
            tick=state.tick,
            atype=action.type.value,
            direction=action.direction.value if action.direction else None,
            target_id=action.target_id,
            context=self._context,
        )
        return action_id

    # -- observation ---------------------------------------------------------

    def attach_observation(self, action_id: str, obs: Observation) -> None:
        self._run(
            """
            MATCH (a:Action {id: $aid, context: $context})
            CREATE (o:Observation {
                id:               $oid,
                tick:             $tick,
                sensor:           $sensor,
                visible_entities: $visible,
                distances:        $distances,
                goal_achieved:    $goal,
                context:          $context
            })
            CREATE (a)-[rel:PRODUCES {context: $context}]->(o)
            """,
            aid=action_id,
            oid=obs.id,
            tick=obs.tick,
            sensor=obs.sensor,
            visible=obs.visible_entities,
            distances=obs.distances,
            goal=obs.goal_achieved,
            context=self._context,
        )
        for eid in obs.visible_entities:
            self._run(
                """
                MATCH (o:Observation {id: $oid, context: $context}),
                      (e:Entity {id: $eid, context: $context})
                MERGE (o)-[rel:CONCERNS]->(e)
                SET rel.context = $context
                """,
                oid=obs.id,
                eid=eid,
                context=self._context,
            )

    # -- transition ----------------------------------------------------------

    def add_transition(
        self,
        from_state: StateSnapshot,
        to_state: StateSnapshot,
        action: Action,
        reward: float,
    ) -> None:
        self._run(
            """
            MATCH (s1:State {id: $fid, context: $context}),
                  (s2:State {id: $tid, context: $context})
            CREATE (s1)-[:LEADS_TO {
                via_action:  $via,
                reward:      $reward,
                probability: 1.0,
                context:     $context
            }]->(s2)
            """,
            fid=from_state.id,
            tid=to_state.id,
            via=action.type.value,
            reward=reward,
            context=self._context,
        )

    # -- room-level aggregate transitions (world model) ----------------------

    def create_rooms(self) -> None:
        """Create Room nodes and ADJACENT edges for the grid layout."""
        for (x, y), name in ROOM_NAMES.items():
            self._run(
                """
                MERGE (r:Room {id: $id, context: $context})
                SET r.x = $x, r.y = $y
                """,
                id=name,
                x=x,
                y=y,
                context=self._context,
            )
        for (x, y), name in ROOM_NAMES.items():
            for dx, dy in [(1, 0), (0, 1)]:
                neighbor = ROOM_NAMES.get((x + dx, y + dy))
                if neighbor:
                    self._run(
                        """
                        MATCH (a:Room {id: $a, context: $context}),
                              (b:Room {id: $b, context: $context})
                        MERGE (a)-[rel:ADJACENT]->(b)
                        SET rel.context = $context
                        """,
                        a=name,
                        b=neighbor,
                        context=self._context,
                    )

    def record_transition(
        self,
        from_room: str,
        to_room: str,
        action: str,
        reward: float,
    ) -> dict:
        """MERGE and increment an aggregate TRANSITION edge between Room nodes.

        Returns the updated edge properties: ``{visit_count, r_mean}``.
        """
        rows = self._run(
            """
            MATCH (a:Room {id: $fr, context: $context}),
                  (b:Room {id: $tr, context: $context})
            MERGE (a)-[t:TRANSITION {action: $act, context: $context}]->(b)
            ON CREATE SET t.visit_count = 1,
                          t.r_sum       = $reward,
                          t.r_mean      = $reward
            ON MATCH  SET t.r_sum       = t.r_sum + $reward,
                          t.visit_count = t.visit_count + 1,
                          t.r_mean      = t.r_sum / t.visit_count
            RETURN t.visit_count AS visit_count, t.r_mean AS r_mean
            """,
            fr=from_room,
            tr=to_room,
            act=action,
            reward=reward,
            context=self._context,
        )
        if rows:
            return rows[0]
        return {"visit_count": 1, "r_mean": reward}

    def query_transitions(
        self,
        room_id: str,
        action: str,
        min_visits: int = 1,
    ) -> list[dict]:
        """Return [{next_room, visit_count, r_mean}] for a (room, action) pair.

        Self-loops (next_room == room_id) are excluded. They represent "this
        action at this state didn't change anything" — informative for the
        reward landscape but actively harmful to model-based planning: a
        rollout that treats the self-loop as a viable outcome will get
        trapped there as the least-bad option whenever the real progress
        actions haven't been observed yet. ``known_actions`` already
        filters them the same way, so the two stay consistent.
        """
        return self._run(
            """
            MATCH (r:Room {id: $rid, context: $context})
                  -[t:TRANSITION {action: $act, context: $context}]->(next:Room)
            WHERE t.visit_count >= $min AND next.id <> r.id
            RETURN next.id       AS next_room,
                   t.visit_count AS visit_count,
                   t.r_mean      AS r_mean
            ORDER BY t.visit_count DESC
            """,
            rid=room_id,
            act=action,
            min=min_visits,
            context=self._context,
        )

    def query_all_transitions(self, min_visits: int = 1) -> list[dict]:
        """Return all aggregate TRANSITION edges above threshold (for UI)."""
        return self._run(
            """
            MATCH (a:Room {context: $context})-[t:TRANSITION {context: $context}]->(b:Room {context: $context})
            WHERE t.visit_count >= $min
            RETURN a.id          AS from_room,
                   b.id          AS to_room,
                   t.action      AS action,
                   t.visit_count AS visit_count,
                   t.r_mean      AS r_mean
            ORDER BY t.visit_count DESC
            """,
            min=min_visits,
            context=self._context,
        )

    def get_model_stats(self) -> dict:
        """Summary statistics of the world model's learned transitions."""
        rows = self._run(
            """
            MATCH (:Room {context: $context})-[t:TRANSITION {context: $context}]->(:Room {context: $context})
            RETURN count(t)            AS edge_count,
                   sum(t.visit_count)  AS total_visits,
                   avg(t.visit_count)  AS avg_visits,
                   max(t.visit_count)  AS max_visits,
                   min(t.visit_count)  AS min_visits
            """,
            context=self._context,
        )
        if rows:
            return rows[0]
        return {"edge_count": 0, "total_visits": 0, "avg_visits": 0, "max_visits": 0, "min_visits": 0}

    # -- queries -------------------------------------------------------------

    def query_shortest_path(self, from_room: str, to_room: str) -> list[str]:
        rows = self._run(
            """
            MATCH (a:State {room_id: $fr, context: $context}),
                  (b:State {room_id: $tr, context: $context}),
                  p = shortestPath((a)-[:LEADS_TO*]->(b))
            RETURN [n IN nodes(p) | n.room_id] AS rooms
            LIMIT 1
            """,
            fr=from_room,
            tr=to_room,
            context=self._context,
        )
        if rows:
            return rows[0]["rooms"]
        return []

    def get_state_history(self) -> list[dict]:
        return self._run(
            "MATCH (s:State {context: $context}) RETURN s.tick AS tick, "
            "s.room_id AS room, s.inventory AS inventory, s.done AS done ORDER BY s.tick",
            context=self._context,
        )

    def replay_from_tick(self, tick: int) -> list[dict]:
        return self._run(
            """
            MATCH p = (s:State {tick: $tick, context: $context})
                      -[:LEADS_TO*]->(end:State {context: $context})
            WHERE end.done = true
            RETURN [n IN nodes(p) | {tick: n.tick, room: n.room_id}] AS trajectory
            LIMIT 1
            """,
            tick=tick,
            context=self._context,
        )

    def get_full_episode_path(self) -> list[dict]:
        return self._run(
            """
            MATCH p = (s:State {tick: 0, context: $context})
                      -[:LEADS_TO*]->(e:State {done: true, context: $context})
            RETURN [n IN nodes(p) | {tick: n.tick, room: n.room_id, inventory: n.inventory}] AS path
            LIMIT 1
            """,
            context=self._context,
        )

    def get_states_with_item(self, item_name: str) -> list[dict]:
        return self._run(
            """
            MATCH (s:State {context: $context})-[:HAS]->(e:Entity {name: $name, context: $context})
            RETURN s.tick AS tick, s.room_id AS room
            ORDER BY s.tick
            """,
            name=item_name,
            context=self._context,
        )

    def get_locked_door_observations(self) -> list[dict]:
        return self._run(
            """
            MATCH (o:Observation {context: $context})-[:CONCERNS]->(e:Entity {locked: true, context: $context})
            RETURN o.tick AS tick, e.name AS entity
            ORDER BY o.tick
            """,
            context=self._context,
        )

    def run_cypher(self, cypher: str) -> list[dict]:
        """Run arbitrary Cypher — used by the interactive REPL.

        Note: context filtering is the user's responsibility here. Preset
        queries embed the context in their WHERE clauses.
        """
        return self._run(cypher)

    # -- generic write escape hatch -----------------------------------------

    def run_write(self, cypher: str, **params: Any) -> list[dict]:
        """Run an arbitrary parameterised write (used by context-specific stubs).

        The caller is responsible for including ``context: $context`` where
        appropriate; ``$context`` is auto-injected if not supplied.
        """
        if "context" not in params:
            params["context"] = self._context
        return self._run(cypher, **params)

    # -- write convenience ---------------------------------------------------

    def write_step(
        self,
        prev_state: StateSnapshot,
        action: Action,
        new_state: StateSnapshot,
        obs: Observation,
        reward: float,
        visible_entity_ids: list[str],
    ) -> dict:
        """Record a full step: new state, action, observation, transition."""
        self.create_state(new_state)
        self.link_state_room(new_state)
        self.link_state_entities(new_state, visible_entity_ids)
        action_id = self.record_action(prev_state, action)
        self.attach_observation(action_id, obs)
        self.add_transition(prev_state, new_state, action, reward)
        if action.type == ActionType.MOVE and action.direction:
            action_key = f"move_{action.direction.value}"
        else:
            action_key = action.type.value
        transition = self.record_transition(
            from_room=prev_state.room_id,
            to_room=new_state.room_id,
            action=action_key,
            reward=reward,
        )
        return {"action_id": action_id, "action_key": action_key, "transition": transition}
