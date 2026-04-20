# World Model — How It Works

## What the world model is

The world model is a learned representation of the environment's dynamics: given the agent is in room S and takes action A, what room S' does it end up in, and what reward does it receive? Formally, it learns two functions from experience:

- **T(s'|s,a)** — transition probability: "if I move north from Kitchen, I go to... Hallway? Pantry? How sure am I?"
- **R(s,a)** — expected reward: "moving north from Kitchen costs about -0.1 per step"

These are not hardcoded. The agent discovers them by acting in the real environment and recording what happens. The graph is the memory that stores this experience. The world model reads from it.

---

## Graph schema — what gets stored

The Neo4j database holds two layers of data: a detailed per-tick history and an aggregate transition model.

### Per-tick history (existing before the world model)

Every simulation step writes five node/edge types:

```
(:State {id, tick, type, room_id, agent_x, agent_y, inventory, done})
(:Action {id, tick, type, direction, target_id, cost})
(:Observation {id, tick, sensor, visible_entities, distances, goal_achieved})
(:Entity {id, name, type, x, y, pickable, locked})

(:State)-[:TRIGGERS]->(:Action)
(:Action)-[:PRODUCES]->(:Observation)
(:Observation)-[:CONCERNS]->(:Entity)
(:State)-[:HAS]->(:Entity)
(:State)-[:LEADS_TO {via_action, reward, probability}]->(:State)
```

This is a complete log: every tick, every action, every observation. It's useful for replay and Cypher queries over episode history, but it grows linearly with time and has no aggregation.

### Aggregate transition model (the world model layer)

The world model reads from a separate, aggregated layer:

```
(:Room {id: "Kitchen", x: 0, y: 0})
(:Room)-[:TRANSITION {action, visit_count, r_sum, r_mean}]->(:Room)
```

**Room nodes** — 25 total, one per grid cell. Created once at setup (`graph_store.create_rooms()`). They persist across steps and episodes. Their `id` is the room name (e.g., "Kitchen"), and `x`/`y` are the grid coordinates.

**TRANSITION edges** — one per unique `(from_room, action, to_room)` triple. The `action` field is direction-qualified: `"move_north"`, `"move_south"`, `"move_east"`, `"move_west"`, or non-movement actions like `"observe"`, `"pick_up"`. Properties:

| Property | Type | Meaning |
|----------|------|---------|
| `action` | string | The direction-qualified action key (part of the MERGE key) |
| `visit_count` | int | How many times this exact transition was observed |
| `r_sum` | float | Sum of all rewards received on this transition |
| `r_mean` | float | `r_sum / visit_count` — average reward per transition |

These edges are created and updated via `MERGE`: the first time the agent moves north from Kitchen to Pantry, a new edge is created with `visit_count=1`. Every subsequent time, `visit_count` is incremented and `r_mean` is recalculated. This is the learning signal — the more the agent explores, the more accurate these statistics become.

---

## When data is written

### On episode reset

`SimulationState.reset_episode()` triggers context-specific cleanup:

**Spatial / scene contexts:**

1. **`graph.clear_context()`** — `MATCH (n {context: $ctx}) DETACH DELETE n` — wipes all nodes and edges in the active context only (other contexts are untouched)
2. **`graph.create_rooms()`** — recreates the 25 Room nodes via `MERGE`
3. **`graph.create_state(initial_state)`** — writes the tick-0 State node
4. **`graph.link_state_room(initial_state)`** — writes an IN_ROOM edge from the State to its Room
5. **`graph.create_entity(e)`** for each entity — writes key, lamp, door Entity nodes
6. **`graph.link_state_entities()`** — writes HAS edges from initial State to visible entities

After reset, the graph has 25 Room nodes, 1 State node, 3 Entity nodes, and some HAS/IN_ROOM edges. Zero TRANSITION edges — the world model has no data yet.

**Generalised context:**

Reset only deletes per-tick scaffolding (State, Action, Observation, WorkingState nodes). AgentState signature Rooms and learned TRANSITION/OF edges are **preserved** across resets — without them every fresh episode would start the rollout panel at zero data. This means the world model accumulates evidence across episodes in the generalised context.

### On every simulation step

`do_step()` calls `graph.write_step()`, which executes seven operations in sequence:

1. **`create_state(new_state)`** — writes a new State node for this tick
2. **`link_state_room(new_state)`** — writes an IN_ROOM edge from the State to its Room
3. **`link_state_entities(new_state, visible_ids)`** — HAS edges to entities in the current room
4. **`record_action(prev_state, action)`** — writes an Action node and a TRIGGERS edge from the previous State
5. **`attach_observation(action_id, obs)`** — writes an Observation node, a PRODUCES edge from the Action, and CONCERNS edges to visible entities
6. **`add_transition(prev_state, new_state, action, reward)`** — writes a LEADS_TO edge between the two State nodes (per-tick history)
7. **`record_transition(from_room, to_room, action_key, reward)`** — the world model write

Step 7 is the critical one for the world model. It runs this Cypher:

```cypher
MATCH (a:Room {id: $fr}), (b:Room {id: $tr})
MERGE (a)-[t:TRANSITION {action: $act}]->(b)
ON CREATE SET t.visit_count = 1,
              t.r_sum       = $reward,
              t.r_mean      = $reward
ON MATCH  SET t.r_sum       = t.r_sum + $reward,
              t.visit_count = t.visit_count + 1,
              t.r_mean      = t.r_sum / t.visit_count
```

The `action_key` is direction-qualified. For a move action:

```python
if action.type == ActionType.MOVE and action.direction:
    action_key = f"move_{action.direction.value}"  # e.g., "move_north"
else:
    action_key = action.type.value                  # e.g., "observe"
```

This means the model learns separate transition statistics for each direction. Moving north from Kitchen produces a different TRANSITION edge than moving east from Kitchen.

**Writes happen on every step regardless of whether the model is ON or OFF.** The graph always accumulates experience. Model OFF means the experience is collected but not used for planning. Model ON means the agent both reads from and writes to the transition statistics.

---

## When data is read

### By the world model (during planning)

When the model is ON, the `ModelBasedPolicy` calls `WorldModel.best_action()` once per step, before the action is executed. This triggers a cascade of graph reads:

**1. `best_action(room_id, candidate_actions)`**

For each candidate action (e.g., `["move_north", "move_south", "move_east", "move_west"]`), calls `evaluate_action()`.

**2. `evaluate_action(room_id, action)`**

First, queries the transition for this (room, action) pair to check if the model has any data:

```cypher
MATCH (r:Room {id: $rid})-[t:TRANSITION {action: $act}]->(next:Room)
WHERE t.visit_count >= $min
RETURN next.id AS next_room, t.visit_count AS visit_count, t.r_mean AS r_mean
ORDER BY t.visit_count DESC
```

If nothing comes back (the agent has never taken this action from this room, or hasn't taken it enough times), the action is scored as `-inf` — the model says "I don't know what happens."

If transitions are found, probabilities are computed from visit counts:

```python
total = sum(visit_count for each row)
probability = row.visit_count / total
```

In a deterministic environment, moving north from Kitchen always goes to the same room, so there's one row with probability 1.0. In a stochastic environment, there could be multiple outcomes with different probabilities.

Then runs `n_rollouts` (default 6) imagined rollouts.

**3. `imagined_rollout(start_room, first_action, horizon=5)`**

This is the core of the world model — the agent literally imagines what would happen:

- **Step 0**: Sample from `T(s'|start_room, first_action)`. The agent asks "if I go north from Kitchen, where do I end up?" and samples a next room weighted by the learned probabilities. Record the reward.

- **Steps 1-4**: At each subsequent room, query `known_actions(room)` to find what actions the model has data for:

  ```cypher
  MATCH (r:Room {id: $rid})-[t:TRANSITION]->(next:Room)
  WHERE t.visit_count >= $min AND next.id <> r.id
  RETURN DISTINCT t.action AS action
  ```

  Then greedily pick the action with the best expected immediate reward (computed as the weighted sum of `probability * r_mean` across all transition outcomes). Sample the next room. Apply discount factor `gamma^t` to the reward.

- **Termination**: The rollout stops when it hits a room where the model has no known actions — the boundary of explored territory. The agent's imagination literally runs out at that point.

Each rollout produces a `RolloutResult` with the imagined path (sequence of rooms), expected return (discounted cumulative reward), and number of steps completed.

**4. Back in `best_action()`**

The Q-value for each action is the average expected return across its rollouts:

```python
q_value = sum(rollout.expected_return for rollout in rollouts) / n_rollouts
```

The action with the highest Q-value wins. If no action has any data (`q_value == -inf` for all), the model reports `used_model=False` and the policy falls back to a greedy heuristic.

### Total graph reads per step (model ON)

For a room with 4 valid move directions, one planning step involves:

- 4 calls to `query_transitions()` (one per candidate action) — 4 Cypher queries
- Up to 24 rollouts (4 actions x 6 rollouts each), each making up to 5 steps
- Each rollout step: 1 `known_actions()` query + 1 `query_transitions()` per known action to evaluate immediate rewards + 1 `query_transitions()` for the chosen action

In the worst case (fully explored graph), this is roughly **4 + 24*(5*(1 + 4 + 1))** = ~724 Cypher queries per step. In practice, rollouts terminate early at the boundary of explored territory, and many rooms have fewer than 4 known actions, so the actual number is much lower.

### By the graph visualization (UI refresh)

When the frontend requests graph data (every ~5 ticks or on manual refresh), the server runs two queries:

```cypher
-- All nodes (State, Action, Observation, Entity, Room)
MATCH (n)
RETURN n.id AS id, labels(n)[0] AS label, ... AS props

-- All edges (LEADS_TO, TRIGGERS, PRODUCES, HAS, CONCERNS, TRANSITION)
MATCH (a)-[r]->(b)
RETURN a.id AS source, b.id AS target, type(r) AS rel_type, ...
```

For TRANSITION edges, the response includes `visit_count`, `r_mean`, and `action` — these drive the confidence visualization in the graph view (thicker/more opaque edges = more visits).

---

## How the model evaluates actions (not just the best edge)

A common misconception: the model picks the outgoing TRANSITION edge with the highest `r_mean`. It does not. The `r_mean` on a single edge is the average *immediate* reward for that one transition. The model cares about *cumulative* return over multiple steps.

Here's the distinction with a concrete example. The agent is in Kitchen and can go east or south:

```
Kitchen -[move_east,  r_mean=-0.1]-> Pantry
Kitchen -[move_south, r_mean=-0.1]-> Hallway
```

Both immediate edges have the same `r_mean`. If the model only looked at the best edge, it would pick arbitrarily. But the model runs rollouts — it imagines what happens *after* the first step:

**Rollout for move_east:**
```
Kitchen →(move_east)→ Pantry         r=-0.1
Pantry  →(move_east)→ Dining Room    r=-0.1   (× 0.95)
Dining Room →(move_east)→ Lounge     r=-0.1   (× 0.90)
Lounge  →(move_south)→ Music Room    r=-0.1   (× 0.86)
Music Room → ??? (unknown territory, rollout ends)
                                     Q ≈ -0.37
```

**Rollout for move_south:**
```
Kitchen →(move_south)→ Hallway       r=-0.1
Hallway →(move_south)→ Basement      r=-0.1   (× 0.95)
Basement →(move_east)→ Storage       r=-0.1   (× 0.90)
Storage →(move_east)→ Workshop       r=-0.1   (× 0.86)
Workshop: agent picks up key          r=+1.0   (× 0.81)
                                     Q ≈ +0.45
```

The model picks **move_south** because the 5-step imagined trajectory leads to the key (reward +1.0), even though the immediate edge reward is identical. The Q-value reflects the entire rollout, not just the first hop.

This is the core difference between a world model and simple graph traversal. The model simulates forward through learned dynamics, sampling from transition probabilities and accumulating discounted reward at each step. The rollout's internal decisions (which action to take at each imagined step) are greedy on immediate `r_mean` — but the top-level action selection uses the full multi-step return.

To be precise about the two levels:

- **Top-level action selection** (which direction does the agent actually move): picks the action with the highest average Q-value across 6 rollouts, where Q-value = discounted cumulative reward over a 5-step imagined trajectory.
- **Rollout continuation policy** (which action does the imagination take at each step after the first): greedy on expected immediate reward, computed as `sum(probability × r_mean)` across all transition outcomes for each known action. This is a heuristic — the rollout doesn't do recursive planning, it just picks the locally best-looking action to push the simulation forward.

The `r_mean` stored on TRANSITION edges matters, but as an input to rollout simulation, not as the direct selection criterion.

---

## The planning loop in detail

Here's what happens on a single tick when the model is ON:

```
1.  ModelBasedPolicy.act(world, state) called
2.  Check: any pickable items here? → pick up if yes
3.  Determine target: KEY_ROOM (Workshop) or GOAL_ROOM (Garden)
4.  Get valid move actions: [move_north, move_south, move_east, move_west]
5.  Call WorldModel.best_action("Kitchen", ["move_north", ...])
    │
    ├── evaluate_action("Kitchen", "move_north")
    │   ├── query: T(s'|Kitchen, move_north) → [{Pantry, p=1.0, r=-0.1}]  ← READ
    │   └── imagined_rollout("Kitchen", "move_north") × 6
    │       ├── Sample: Kitchen →(move_north)→ Pantry, r=-0.1
    │       ├── known_actions("Pantry") → ["move_east", "move_south"]     ← READ
    │       ├── Best immediate: move_east (r=-0.1) vs move_south (r=-0.1)
    │       ├── Sample: Pantry →(move_east)→ Dining Room, r=-0.1
    │       ├── known_actions("Dining Room") → [...]                       ← READ
    │       └── ... continues for horizon steps
    │
    ├── evaluate_action("Kitchen", "move_south")
    │   └── ... same process
    │
    ├── evaluate_action("Kitchen", "move_east")
    │   └── ... same process
    │
    └── Pick action with highest average Q-value
        
6.  Convert winning action_key ("move_east") → Direction.EAST
7.  Return Action(type=MOVE, direction=EAST)
8.  World executes the action → new state, reward
9.  graph.write_step() records everything including TRANSITION update        ← WRITE
10. Tick message sent to UI with action_evaluations table
```

---

## What the UI shows

### Trajectory vs rollouts

These are two distinct concepts in the UI:

- **Trajectory**: the actual sequence of states and actions the agent takes through the environment. Shown in the StatusPanel (current state) and GridCanvas (agent position, visited cells, planned path overlay).
- **Rollouts**: the model's *simulated* predictions from a given room — "if I go north, what do I think happens?" Shown in the Rollout panel as a branching tree. Each branch is an imagined future, sampled from learned transition probabilities.

### Tick message fields

When the model is ON, each tick message includes:

- **`using_model`** (bool): whether the model had enough data to choose an action, or fell back to greedy
- **`planned_path`** (list of room names): the imagined trajectory from the best rollout — drawn as a gold dashed line on the grid canvas
- **`action_evaluations`** (list): one entry per candidate action, showing:
  - `action`: the direction key (e.g., "move_north")
  - `q_value`: average expected return from rollouts (null if the model has no data)
  - `immediate_next`: the most likely next room
  - `immediate_reward`: expected immediate reward
  - `chosen`: whether this action was selected

Every tick message (regardless of model toggle) also includes:

- **`rollout_data`** (dict): rollout predictions computed independently for the Rollout panel, containing:
  - `current_room`: the room rollouts were computed from (the pre-move room — see timing note below)
  - `evaluations`: one entry per direction (N/S/E/W), each with:
    - `rollouts`: list of simulated traces, each with `path` (room sequence), `expected_return`, and `steps`
    - `chosen`: best action by the visualization model
    - `chosen_by_model`: the action the model-based policy actually chose (when model is ON)

### Rollout computation and timing

Rollouts are computed by `_compute_rollouts()` in the backend, which runs on every tick. It uses a separate `WorldModel(min_visits=1)` so rollout data appears as soon as any transitions have been observed — even a single visit is enough.

**Critical timing detail**: transitions are recorded as `FROM room_A → TO room_B`. After a step, the agent is at `room_B`, but outgoing transitions from `room_B` may not exist yet (first visit). The function therefore computes rollouts from the **previous room** (`prev_room` — the room the agent was in before this step), where outgoing transitions are guaranteed to exist. When the model-based policy is active, the model's own chosen action is flagged so the Rollout panel can highlight it.

### StatusPanel

Renders the `action_evaluations` as a table. When `using_model` is true, the heading shows "Model Planning" and the chosen action row is highlighted. When the model is active but fell back to greedy (not enough transitions above the policy's `min_visits` threshold), it shows "Model Exploring" with all evaluated actions including those with unknown Q-values (shown as "—").

### Rollout panel

Renders `rollout_data` as a D3 SVG tree:
- **Root node**: the room rollouts were computed from (pre-move room)
- **Action nodes** (depth 1): one pill per direction (N/S/E/W). Shows the direction and Q-value. Directions with no learned transitions appear as dashed outlines with "?".
- **Rollout paths** (depth 2+): room sequences from imagined rollouts. Multiple rollouts sharing the same prefix are merged into a trie to reduce visual clutter. Branch nodes show a count badge (e.g., "x4") when multiple rollouts pass through.
- **Color**: absolute scale anchored at zero — positive returns are green, negative are red, near-zero is neutral grey. This avoids the confusion of relative scaling where all-good predictions would show some as red.
- **Model highlight**: when the model policy chose an action, that branch gets a green border and "MODEL" badge.

### ControlBar

Shows a badge:
- **PLANNING** (green): the model chose the action via rollouts
- **GREEDY** (yellow): the model had no data; fell back to coordinate-based heuristic

---

## The learning curve

The world model starts knowing nothing. Here's how it improves:

**Tick 0**: Zero TRANSITION edges. Model has no data. Falls back to greedy.

**Ticks 1-50 (random exploration)**: TRANSITION edges accumulate. Each new move creates or increments an edge. With `min_visits=2`, an edge becomes usable after the agent takes the same action from the same room twice.

**Ticks 50-100**: The explored territory grows. The model can plan through rooms the agent has visited multiple times. Rollouts reach further before hitting the unknown boundary. The agent starts making model-driven decisions for parts of the grid it knows well, while falling back to greedy in unexplored areas.

**Ticks 100+**: Most of the grid is covered. The model can imagine complete trajectories from the current room to the goal. Planning consistently outperforms random exploration.

The threshold slider controls where this boundary sits. `threshold=1` means the model trusts a transition after seeing it once — fast learning but potentially unreliable. `threshold=5` means it needs 5 observations — slower but more confident. The demo's instructive failure: set threshold=1, watch the agent confidently plan down a dead end because it saw one transition; set threshold=5, watch it plan reliably because every edge it trusts has been confirmed multiple times.

---

## How this differs from BFS

The BFS policy (`BFSPolicy`) also uses the graph to navigate, but it does something fundamentally different:

| | BFS Policy | World Model Policy |
|-|------------|-------------------|
| **What it reads** | Per-tick State nodes and LEADS_TO edges | Aggregate Room nodes and TRANSITION edges |
| **Planning method** | Cypher `shortestPath()` query | Imagined rollouts sampled from learned T(s'\|s,a) |
| **Considers reward** | No — finds shortest path by hop count | Yes — evaluates expected discounted return |
| **Considers uncertainty** | No — treats all edges equally | Yes — only trusts edges above visit threshold |
| **Learns from experience** | Indirectly (more State nodes = more paths) | Directly (visit counts and reward means update) |
| **Shows reasoning** | None — just follows the path | Action evaluation table with Q-values per direction |

BFS answers "what's the shortest path?" The world model answers "what action gives the highest expected return, given what I've learned about the environment?" The second question is what model-based reinforcement learning is about.
