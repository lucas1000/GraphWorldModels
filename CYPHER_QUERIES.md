# Cypher Query Guide

A reference of Cypher queries you can run in the **Cypher Console** (right panel) to demonstrate the world model's predictive power and inspect the agent's learned representation.

The graph uses the following schema:

| Node      | Key properties |
|-----------|----------------|
| `State`       | `id`, `tick`, `room_id`, `agent_x`, `agent_y`, `inventory`, `done`, `type` |
| `Action`      | `id`, `tick`, `type`, `direction`, `target_id` |
| `Observation` | `id`, `tick`, `sensor`, `visible_entities`, `goal_achieved` |
| `Entity`      | `id`, `name`, `type`, `x`, `y`, `pickable`, `locked` |
| `Room`        | `id` (name), `x`, `y` |

| Relationship  | Source → Target     | Properties |
|---------------|---------------------|------------|
| `LEADS_TO`    | State → State       | `via_action`, `reward`, `probability` |
| `TRIGGERS`    | State → Action      | — |
| `PRODUCES`    | Action → Observation | — |
| `HAS`         | State → Entity      | — |
| `CONCERNS`    | Observation → Entity | — |
| `IN_ROOM`     | State → Room        | — |
| `ADJACENT`    | Room → Room         | static grid layout |
| `TRANSITION`  | Room → Room         | `action`, `visit_count`, `r_sum`, `r_mean` (learned world model) |

---

## 1. Learned transition dynamics — T(s' | s, a)

### Where does the model predict I'll go from Kitchen heading north?

```cypher
MATCH (r:Room {id: 'Kitchen'})-[t:TRANSITION {action: 'move_north'}]->(next:Room)
RETURN next.id AS next_room,
       t.visit_count AS visits,
       t.r_mean AS avg_reward
ORDER BY t.visit_count DESC
```

### Everything the model has learned about a given room

```cypher
MATCH (r:Room {id: 'Workshop'})-[t:TRANSITION]->(next:Room)
RETURN t.action AS action,
       next.id AS next_room,
       t.visit_count AS visits,
       t.r_mean AS r_mean
ORDER BY t.action, t.visit_count DESC
```

---

## 2. Model confidence & exploration coverage

### Most-confident learned transitions

```cypher
MATCH (a:Room)-[t:TRANSITION]->(b:Room)
RETURN a.id AS from_room,
       t.action AS action,
       b.id AS to_room,
       t.visit_count AS visits
ORDER BY t.visit_count DESC
LIMIT 20
```

### Blind spots — grid neighbours the agent has never transitioned to

```cypher
MATCH (r:Room)-[:ADJACENT]->(neighbor:Room)
WHERE NOT EXISTS {
  MATCH (r)-[:TRANSITION]->(neighbor)
}
RETURN r.id AS room, neighbor.id AS unexplored_neighbor
```

### Percentage of the grid that has been explored

```cypher
MATCH (a:Room)-[:TRANSITION]->(b:Room)
WITH count(DISTINCT [a.id, b.id]) AS learned_pairs
MATCH (a:Room)-[:ADJACENT]->(b:Room)
WITH learned_pairs, count(*) AS possible_pairs
RETURN learned_pairs,
       possible_pairs,
       round(100.0 * learned_pairs / possible_pairs) AS pct_explored
```

### Rooms ranked by how well the agent knows them

```cypher
MATCH (r:Room)-[t:TRANSITION]->(:Room)
RETURN r.id AS room,
       count(t) AS known_actions,
       sum(t.visit_count) AS total_visits
ORDER BY total_visits DESC
```

---

## 3. Path planning — what the model would predict

### Shortest predicted path Kitchen → Garden through learned transitions

```cypher
MATCH path = shortestPath(
  (a:Room {id: 'Kitchen'})-[:TRANSITION*]->(b:Room {id: 'Garden'})
)
RETURN [r IN nodes(path) | r.id] AS route,
       length(path) AS hops
```

### All viable routes up to the model's planning horizon (5 hops)

```cypher
MATCH path = (a:Room {id: 'Kitchen'})-[:TRANSITION*1..5]->(b:Room {id: 'Garden'})
RETURN [r IN nodes(path) | r.id] AS route,
       length(path) AS hops
ORDER BY hops
LIMIT 5
```

### Sequence of actions the model would recommend

```cypher
MATCH path = shortestPath(
  (a:Room {id: 'Kitchen'})-[:TRANSITION*]->(b:Room {id: 'Garden'})
)
RETURN [rel IN relationships(path) | rel.action] AS action_sequence,
       length(path) AS hops
```

---

## 4. Reward structure — R(s, a)

### Transitions that carry non-trivial reward

```cypher
MATCH (a:Room)-[t:TRANSITION]->(b:Room)
WHERE t.r_mean <> -0.1
RETURN a.id AS from_room,
       t.action AS action,
       b.id AS to_room,
       t.r_mean AS avg_reward,
       t.visit_count AS visits
```

### Expected total reward along the shortest learned path

```cypher
MATCH path = shortestPath(
  (a:Room {id: 'Kitchen'})-[:TRANSITION*]->(b:Room {id: 'Garden'})
)
WITH relationships(path) AS edges
UNWIND edges AS e
RETURN sum(e.r_mean) AS expected_total_reward,
       count(e) AS steps
```

### Top-reward transitions the model has discovered

```cypher
MATCH (a:Room)-[t:TRANSITION]->(b:Room)
RETURN a.id AS from_room, t.action, b.id AS to_room, t.r_mean
ORDER BY t.r_mean DESC
LIMIT 5
```

---

## 5. Trajectory analysis — what actually happened

### Full episode trajectory ordered by tick

```cypher
MATCH (s:State)
RETURN s.tick AS tick,
       s.room_id AS room,
       s.inventory AS inventory,
       s.done AS done
ORDER BY s.tick
```

### Room visit frequency (where did the agent spend its time?)

```cypher
MATCH (s:State)-[:IN_ROOM]->(r:Room)
RETURN r.id AS room,
       count(s) AS visits
ORDER BY visits DESC
```

### Tick at which each room was first discovered

```cypher
MATCH (s:State)-[:IN_ROOM]->(r:Room)
WITH r.id AS room, min(s.tick) AS first_visit
RETURN room, first_visit
ORDER BY first_visit
```

### Complete state → state trajectory with actions and rewards

```cypher
MATCH (s:State)-[l:LEADS_TO]->(n:State)
RETURN s.tick AS tick,
       s.room_id AS from_room,
       l.via_action AS action,
       n.room_id AS to_room,
       l.reward AS reward
ORDER BY s.tick
```

---

## 6. Entity interactions

### When and where was the key picked up?

```cypher
MATCH (s:State)-[:TRIGGERS]->(a:Action {type: 'pick_up', target_id: 'key_1'})
RETURN s.tick AS tick,
       s.room_id AS room
```

### All observations of the locked door

```cypher
MATCH (o:Observation)-[:CONCERNS]->(e:Entity {name: 'locked_door'})
RETURN o.tick AS tick
ORDER BY o.tick
```

### Every state in which the agent was holding the key

```cypher
MATCH (s:State)
WHERE 'key_1' IN s.inventory
RETURN s.tick AS tick,
       s.room_id AS room
ORDER BY s.tick
```

### Rooms the agent visited while holding nothing

```cypher
MATCH (s:State)-[:IN_ROOM]->(r:Room)
WHERE size(s.inventory) = 0
RETURN DISTINCT r.id AS room, count(s) AS ticks_visited
ORDER BY ticks_visited DESC
```

---

## 7. Cross-cutting demonstrations

### Learned model vs. ground-truth adjacency (are they the same?)

```cypher
MATCH (a:Room)-[:ADJACENT]->(b:Room)
WITH a, b, EXISTS { MATCH (a)-[:TRANSITION]->(b) } AS learned
RETURN a.id AS from_room,
       b.id AS to_room,
       learned
ORDER BY learned, a.id
```

### Agent's path through the graph as a sequence of rooms visited

```cypher
MATCH (initial:State {tick: 0})
MATCH path = (initial)-[:LEADS_TO*]->(end:State)
WHERE NOT (end)-[:LEADS_TO]->()
RETURN [n IN nodes(path) | n.room_id] AS room_sequence,
       length(path) AS steps
```

### Rooms the model can reach from Kitchen in at most 3 steps

```cypher
MATCH (a:Room {id: 'Kitchen'})-[:TRANSITION*1..3]->(b:Room)
RETURN DISTINCT b.id AS reachable_room
ORDER BY b.id
```

---

## Tips

- The model only uses transitions with `visit_count >= threshold` (default 2). Set the Threshold slider to 1 to see every observed transition.
- `TRANSITION` edges are *directed and action-qualified* — there can be up to 4 edges between the same pair of rooms (one per direction).
- `LEADS_TO` captures the exact state-level trajectory; `TRANSITION` is the aggregated room-level world model learned from that trajectory.
- `ADJACENT` is static grid geometry, not learned — it's the ground truth the agent is trying to discover.
