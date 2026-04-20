"""World model — learns transition dynamics T(s'|s,a) and R(s,a) from the graph.

The model reads aggregate TRANSITION edges between Room nodes to answer:
  - "If I do action A in room S, where do I end up and what reward do I get?"
  - "Which action from room S gives the best expected return over H steps?"

This is the core of the Dyna-Q architecture: the agent imagines rollouts
through the learned model and picks the action with the highest value.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore


@dataclass
class RolloutResult:
    """One imagined rollout — what the model predicted would happen."""
    action: str
    path: list[str]          # rooms visited in imagination
    expected_return: float   # discounted cumulative reward
    steps: int               # how far the rollout got before hitting unknown territory
    success: bool = False    # True if the rollout saw a goal-sized reward (>= 5.0)


@dataclass
class ActionEvaluation:
    """Result of evaluating one candidate action via imagined rollouts."""
    action: str
    q_value: float                        # average expected return across rollouts
    rollouts: list[RolloutResult]         # individual rollout traces
    immediate_transition: str | None      # predicted next room (most likely)
    immediate_reward: float               # predicted immediate reward


@dataclass
class PlanningResult:
    """Full result of one planning step — what the model considered."""
    current_room: str
    chosen_action: str | None
    evaluations: list[ActionEvaluation]   # one per candidate action
    used_model: bool                       # False if fell back to greedy


class WorldModel:
    """Reads aggregate TRANSITION edges to model the environment dynamics.

    Each TRANSITION edge stores:
      - action: direction-qualified key like "move_north"
      - visit_count: how many times this (room, action) -> next_room was observed
      - r_mean: average reward for this transition

    The model uses these to simulate rollouts in imagination and evaluate
    candidate actions by their expected return.
    """

    def __init__(self, graph: GraphStore, min_visits: int = 2) -> None:
        self.graph = graph
        self.min_visits = min_visits
        self._rng = random.Random()
        self.last_plan: PlanningResult | None = None

    def transition(self, room_id: str, action: str) -> list[tuple[str, float, float]]:
        """Query T(s'|s,a): return [(next_room, probability, r_mean)].

        Probability is derived from visit counts. Only transitions observed
        at least min_visits times are returned — low-evidence edges are
        treated as unknown (the model says "I don't know what happens").
        """
        rows = self.graph.query_transitions(room_id, action, self.min_visits)
        if not rows:
            return []
        total = sum(r.get("visit_count", 0) for r in rows)
        results: list[tuple[str, float, float]] = []
        for row in rows:
            count = row.get("visit_count", 0)
            prob = count / total if total > 0 else 0.0
            results.append((row["next_room"], prob, row.get("r_mean", 0.0)))
        return results

    def known_actions(self, room_id: str) -> list[str]:
        """Return actions the model has learned transitions for from this room."""
        rows = self.graph.run_write(
            """
            MATCH (r:Room {id: $rid, context: $context})
                  -[t:TRANSITION {context: $context}]->(next:Room {context: $context})
            WHERE t.visit_count >= $min AND next.id <> r.id
            RETURN DISTINCT t.action AS action
            """,
            rid=room_id,
            min=self.min_visits,
        )
        return [row["action"] for row in rows]

    def _sample(self, candidates: list[tuple[str, float, float]]) -> tuple[str, float]:
        """Sample next state from T(s'|s,a) weighted by probability."""
        r = self._rng.random()
        cumulative = 0.0
        for next_room, prob, r_mean in candidates:
            cumulative += prob
            if r <= cumulative:
                return next_room, r_mean
        return candidates[-1][0], candidates[-1][2]

    def imagined_rollout(
        self,
        start_room: str,
        first_action: str,
        horizon: int = 6,
        gamma: float = 0.95,
    ) -> RolloutResult:
        """Run one imagined rollout: take first_action, then pick the best
        known action at each subsequent step for the remaining horizon.

        This is the core of the world model — the agent literally imagines
        "what would happen if I went north? then what? then what?" by
        sampling from learned transition probabilities.
        """
        path = [start_room]
        G = 0.0
        s = start_room
        steps = 0
        success = False

        # First step: take the specified action
        candidates = self.transition(s, first_action)
        if not candidates:
            return RolloutResult(
                action=first_action, path=path,
                expected_return=float("-inf"), steps=0,
            )
        s, r = self._sample(candidates)
        G += r
        if r >= 5.0:
            success = True
        path.append(s)
        steps = 1

        # Remaining steps: greedily pick the known action with best
        # immediate reward (simple heuristic for the rollout policy)
        for t in range(1, horizon):
            available = self.known_actions(s)
            if not available:
                break  # hit unknown territory — model can't predict further

            # Pick the action with the best immediate expected reward
            best_a, best_r = None, float("-inf")
            for a in available:
                cands = self.transition(s, a)
                if cands:
                    # Expected immediate reward for this action
                    exp_r = sum(p * rm for _, p, rm in cands)
                    if exp_r > best_r:
                        best_r = exp_r
                        best_a = a
            if best_a is None:
                break

            cands = self.transition(s, best_a)
            if not cands:
                break
            s, r = self._sample(cands)
            G += (gamma ** t) * r
            if r >= 5.0:
                success = True
            path.append(s)
            steps += 1

        return RolloutResult(
            action=first_action, path=path,
            expected_return=G, steps=steps, success=success,
        )

    def evaluate_action(
        self,
        room_id: str,
        action: str,
        n_rollouts: int = 8,
        horizon: int = 6,
    ) -> ActionEvaluation:
        """Evaluate one action by running multiple imagined rollouts."""
        candidates = self.transition(room_id, action)
        if not candidates:
            return ActionEvaluation(
                action=action, q_value=float("-inf"),
                rollouts=[], immediate_transition=None, immediate_reward=0.0,
            )

        # Most likely next room and expected reward
        best_next = max(candidates, key=lambda c: c[1])
        exp_reward = sum(p * rm for _, p, rm in candidates)

        rollouts = [
            self.imagined_rollout(room_id, action, horizon=horizon)
            for _ in range(n_rollouts)
        ]
        q = sum(ro.expected_return for ro in rollouts) / len(rollouts)

        return ActionEvaluation(
            action=action, q_value=q, rollouts=rollouts,
            immediate_transition=best_next[0], immediate_reward=exp_reward,
        )

    def best_action(
        self,
        room_id: str,
        candidate_actions: list[str],
        n_rollouts: int = 8,
        horizon: int = 6,
    ) -> PlanningResult:
        """Evaluate all candidate actions and return the full planning result.

        This is called every step when the model is ON. The agent imagines
        the outcome of each possible action, runs forward rollouts, and
        picks the action with the highest expected return.
        """
        evaluations: list[ActionEvaluation] = []
        for a in candidate_actions:
            ev = self.evaluate_action(room_id, a, n_rollouts=n_rollouts, horizon=horizon)
            evaluations.append(ev)

        # Filter to actions the model actually knows about
        viable = [ev for ev in evaluations if ev.q_value > float("-inf")]

        if viable:
            best = max(viable, key=lambda ev: ev.q_value)
            result = PlanningResult(
                current_room=room_id,
                chosen_action=best.action,
                evaluations=evaluations,
                used_model=True,
            )
        else:
            result = PlanningResult(
                current_room=room_id,
                chosen_action=None,
                evaluations=evaluations,
                used_model=False,
            )

        self.last_plan = result
        return result
