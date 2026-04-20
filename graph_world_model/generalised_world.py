"""Generalised demo — non-spatial graph world with per-agent toolsets.

The world is a directed call graph of three node types:

  * **Infrastructure** (``human``, ``client``, ``server``) — deterministic
    passthrough. The user's message arrives once at the human node and
    relays up to the orchestrator; the final response relays back.

  * **Agents** (``orchestrator``, ``search_agent``, ``memory_agent``) —
    the only *decision points*. Each agent carries its own local state
    (a small dataclass) and has its own toolshelf. At each tick an agent
    chooses which tool to call next; the tool's effect on the agent's
    local state depends on that state — so the *order* of calls matters
    (e.g. reasoning before grounding produces a hallucinated answer).

  * **Tools** (``llm``, ``db``, ``regulatory_db``, ``pubmed``) —
    deterministic responders. They return scripted payloads and mutate
    the *calling* agent's local state when they come back.

Control flow uses a call stack: an agent or relay that forwards a message
pushes itself onto the stack and hops to the target; when a tool or agent
``return``\\s, the stack pops and the caller's state gets the result.

``TRANSITION`` edges in Neo4j are keyed by *agent state signatures* (e.g.
``search:endpoint:grounded``) rather than raw actor ids, so the shared
``WorldModel`` can learn — and rollouts can display — the causal effects
of tool-call sequences per agent. Rollouts are only meaningful at agent
decision points; the UI gates the Rollout panel on that.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field, replace
from typing import Optional

from .graph_store import GraphStore
from .policies import Policy
from .world import Entity, Observation, StateSnapshot


# ---------------------------------------------------------------------------
# Actor topology
# ---------------------------------------------------------------------------


NodeKind = str  # "infra" | "agent" | "tool"


@dataclass(frozen=True)
class ActorSpec:
    id: str
    kind: NodeKind
    role: str   # display role: "Human"|"Client"|"Server"|"Agent"|"LLM"|"Tool"
    label: str
    pos: tuple[float, float]   # normalised (x, y) canvas hint in [0, 1]


ACTORS: dict[str, ActorSpec] = {
    # infrastructure
    "human":          ActorSpec("human",          "infra", "Agent (Human)", "Researcher",    (0.05, 0.48)),
    "client":         ActorSpec("client",         "infra", "Client",        "Browser/UI",    (0.16, 0.48)),
    "server":         ActorSpec("server",         "infra", "Server",        "API server",    (0.27, 0.48)),
    # agents — all LLM-backed (they call the shared ``llm`` tool as part of their logic)
    "orchestrator":   ActorSpec("orchestrator",   "agent", "Agent (LLM)",   "Orchestrator",  (0.42, 0.48)),
    "search_agent":   ActorSpec("search_agent",   "agent", "Agent (LLM)",   "Search Agent",  (0.60, 0.24)),
    "memory_agent":   ActorSpec("memory_agent",   "agent", "Agent (LLM)",   "Memory Agent",  (0.60, 0.72)),
    # tools
    "llm":            ActorSpec("llm",            "tool",  "LLM",           "Reasoning LLM", (0.45, 0.08)),
    "regulatory_db":  ActorSpec("regulatory_db",  "tool",  "Tool",   "Regulatory DB",   (0.80, 0.08)),
    "pubmed":         ActorSpec("pubmed",         "tool",  "Tool",   "PubMed",          (0.86, 0.30)),
    "db":             ActorSpec("db",             "tool",  "Tool",   "WorkingState DB", (0.86, 0.60)),
}

AGENT_IDS: tuple[str, ...] = tuple(a.id for a in ACTORS.values() if a.kind == "agent")
TOOL_IDS: tuple[str, ...] = tuple(a.id for a in ACTORS.values() if a.kind == "tool")
INFRA_IDS: tuple[str, ...] = tuple(a.id for a in ACTORS.values() if a.kind == "infra")


# Static CALLS edges — directed, labelled with the call type carried along them.
# Every agent has an edge to ``llm`` (per the demo brief).
TOPOLOGY: list[tuple[str, str, str]] = [
    # infrastructure spine — one-shot request forward, terminal response back
    ("human",         "client",        "request"),
    ("client",        "human",         "respond"),
    ("client",        "server",        "forward"),
    ("server",        "client",        "respond"),
    ("server",        "orchestrator",  "dispatch"),
    ("orchestrator",  "server",        "respond"),
    # orchestrator toolshelf
    ("orchestrator",  "llm",           "invoke_llm"),
    ("llm",           "orchestrator",  "return"),
    ("orchestrator",  "db",            "invoke_db"),
    ("db",            "orchestrator",  "return"),
    ("orchestrator",  "search_agent",  "delegate"),
    ("search_agent",  "orchestrator",  "return"),
    ("orchestrator",  "memory_agent", "delegate"),
    ("memory_agent", "orchestrator",  "return"),
    # search_agent toolshelf — every agent has access to llm
    ("search_agent",  "llm",           "invoke_llm"),
    ("llm",           "search_agent",  "return"),
    ("search_agent",  "regulatory_db", "invoke_tool"),
    ("regulatory_db", "search_agent",  "return"),
    ("search_agent",  "pubmed",        "invoke_tool"),
    ("pubmed",        "search_agent",  "return"),
    # memory_agent toolshelf — llm + db (read)
    ("memory_agent", "llm",           "invoke_llm"),
    ("llm",           "memory_agent", "return"),
    ("memory_agent", "db",            "invoke_db"),
    ("db",            "memory_agent", "return"),
]


def outgoing(node: str) -> list[tuple[str, str]]:
    """Outgoing CALLS edges from ``node``: list of (target, call_action)."""
    return [(t, a) for f, t, a in TOPOLOGY if f == node]


# ---------------------------------------------------------------------------
# Task + shared working state
# ---------------------------------------------------------------------------


CANNED_TASK = (
    "Design a Phase II trial for Compound-X in rheumatoid arthritis "
    "(2 doses vs placebo). What sample size do we need?"
)


@dataclass
class WorkingState:
    """Shared document agents read from and (via the db tool) write to."""
    task: str = ""
    primary_endpoint: Optional[str] = None
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    status: str = "drafting"  # "drafting" | "done"

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "primary_endpoint": self.primary_endpoint,
            "effect_size": self.effect_size,
            "sample_size": self.sample_size,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Final artefact — the textual goal of the demo.
# ---------------------------------------------------------------------------
#
# The demo's "Garden" is a 5-section clinical-trial-design recommendation. Each
# section is filled in by a specific tool-call sequence; if the right tools
# weren't called, the section is missing or hallucinated. The artefact's
# *completeness + grounding* is the thing the model-based policy is trying to
# maximise — it's what makes "toggle model on" visibly worthwhile.


FRAGMENT_GROUNDED = "grounded"
FRAGMENT_HALLUCINATED = "hallucinated"
FRAGMENT_MISSING = "missing"


@dataclass
class ArtefactFragment:
    status: str          # "grounded" | "hallucinated" | "missing"
    text: str            # synthetic text snippet (the "generated" content)
    source: Optional[str] = None  # tool that produced this (e.g. "regulatory_db")

    def to_dict(self) -> dict:
        return {"status": self.status, "text": self.text, "source": self.source}


@dataclass
class FinalArtefact:
    """The textual artefact the system is trying to assemble for the human.

    Five sections; the synthesizer (`invoke_llm_synth`) derives the final
    confidence statement from whatever's grounded in the other four.
    """
    indication:    ArtefactFragment
    endpoint:      ArtefactFragment
    effect_size:   ArtefactFragment
    sample_size:   ArtefactFragment
    confidence:    ArtefactFragment

    @classmethod
    def empty(cls, task: str) -> "FinalArtefact":
        return cls(
            indication=ArtefactFragment(
                status=FRAGMENT_GROUNDED,
                text="Compound-X for moderate-to-severe rheumatoid arthritis (Phase II).",
                source="user",
            ),
            endpoint=ArtefactFragment(
                status=FRAGMENT_MISSING,
                text="[primary endpoint unknown — regulatory search not yet performed]",
            ),
            effect_size=ArtefactFragment(
                status=FRAGMENT_MISSING,
                text="[effect size unknown — literature search not yet performed]",
            ),
            sample_size=ArtefactFragment(
                status=FRAGMENT_MISSING,
                text="[sample size unknown — compute pipeline not yet run]",
            ),
            confidence=ArtefactFragment(
                status=FRAGMENT_MISSING,
                text="[confidence not yet assessed — synthesise step not run]",
            ),
        )

    def sections(self) -> list[tuple[str, ArtefactFragment]]:
        return [
            ("indication",  self.indication),
            ("endpoint",    self.endpoint),
            ("effect_size", self.effect_size),
            ("sample_size", self.sample_size),
            ("confidence",  self.confidence),
        ]

    def grounded_count(self) -> int:
        return sum(1 for _, f in self.sections() if f.status == FRAGMENT_GROUNDED)

    def hallucinated_count(self) -> int:
        return sum(1 for _, f in self.sections() if f.status == FRAGMENT_HALLUCINATED)

    def to_dict(self) -> dict:
        sects = {k: f.to_dict() for k, f in self.sections()}
        return {
            "sections": sects,
            "grounded": self.grounded_count(),
            "hallucinated": self.hallucinated_count(),
            "total": len(self.sections()),
        }

    def render_text(self) -> str:
        """Concatenate the fragments into the final response the human sees."""
        lines = []
        for label, frag in self.sections():
            prefix = {"indication": "Indication", "endpoint": "Primary endpoint",
                      "effect_size": "Effect size", "sample_size": "Sample size",
                      "confidence": "Confidence"}[label]
            lines.append(f"{prefix}: {frag.text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-agent local state — each has a ``signature()`` that serves as the Room
# id for the shared WorldModel's TRANSITION edges.
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorState:
    planned:   bool = False
    endpoint:  bool = False
    effect:    bool = False
    sample:    bool = False
    synth:     bool = False
    responded: bool = False
    # Whether each sub-result has been persisted to the WorkingState DB.
    # Persistence is its own state transition so db writes change the
    # signature — without this, the model loops on writes (same state,
    # positive reward, repeat forever).
    endpoint_persisted: bool = False
    effect_persisted:   bool = False
    sample_persisted:   bool = False
    pending:   Optional[str] = None

    def signature(self) -> str:
        if self.responded:
            return "orch:done"
        # Lower-case = obtained but not persisted; UPPER-case = persisted.
        flags = (
            ("p" if self.planned else "-")
            + ("E" if self.endpoint_persisted else ("e" if self.endpoint else "-"))
            + ("F" if self.effect_persisted   else ("f" if self.effect   else "-"))
            + ("N" if self.sample_persisted   else ("n" if self.sample   else "-"))
            + ("s" if self.synth else "-")
        )
        return f"orch:{flags}"


@dataclass
class SearchAgentState:
    topic:         Optional[str] = None    # "endpoint" | "effect" | None
    grounded:      bool = False
    refined:       bool = False
    hallucinated:  bool = False
    result:        Optional[str] = None

    def signature(self) -> str:
        if self.topic is None:
            return "search:idle"
        quality = (
            "hallucinated" if self.hallucinated and not self.refined else
            "refined"      if self.refined else
            "grounded"     if self.grounded else
            "fresh"
        )
        return f"search:{self.topic}:{quality}"


@dataclass
class MemoryAgentState:
    active:        bool = False
    have_params:   bool = False
    have_raw_n:    bool = False
    have_adj_n:    bool = False
    hallucinated:  bool = False
    result:        Optional[int] = None

    def signature(self) -> str:
        if not self.active:
            return "memory:idle"
        flags = (
            ("p" if self.have_params else "-")
            + ("r" if self.have_raw_n else "-")
            + ("a" if self.have_adj_n else "-")
        )
        if self.hallucinated:
            flags += "!"
        return f"memory:{flags}"


@dataclass
class HumanState:
    """The human is also an agent. They emit the initial query, wait for the
    response, then *evaluate* the artefact and decide accept / reject.

    The acceptance test is deterministic on the artefact: accept iff every
    section is grounded and nothing is hallucinated. This makes the human a
    *quality gate* — a premature ``respond`` from the orchestrator gets
    rejected with a large negative reward, which is exactly the signal the
    world model needs to learn "don't respond until the artefact is ready"."""
    phase: str = "drafting"  # "drafting" | "queried" | "awaiting" | "accepted" | "rejected"

    def signature(self) -> str:
        return f"human:{self.phase}"


# Enumerate all signatures so the subgraph builder can pre-create Room nodes
# (WorldModel's TRANSITION record query MATCHes existing Rooms). A couple of
# extra signatures covering the realistic off-path visits by RandomPolicy.
def all_state_signatures() -> list[str]:
    sigs: set[str] = set()
    # Orchestrator: planned × endpoint(none/found/persisted) × effect(none/found/
    # persisted) × sample(none/found/persisted) × synth × responded.
    # endpoint_persisted ⇒ endpoint, etc.
    for planned in (False, True):
        for ep_state in ("none", "found", "persisted"):
            for ef_state in ("none", "found", "persisted"):
                for sm_state in ("none", "found", "persisted"):
                    for synth in (False, True):
                        s = OrchestratorState(
                            planned=planned,
                            endpoint=(ep_state != "none"),
                            endpoint_persisted=(ep_state == "persisted"),
                            effect=(ef_state != "none"),
                            effect_persisted=(ef_state == "persisted"),
                            sample=(sm_state != "none"),
                            sample_persisted=(sm_state == "persisted"),
                            synth=synth,
                        )
                        sigs.add(s.signature())
    sigs.add("orch:done")
    # Search agent
    sigs.add("search:idle")
    for topic in ("endpoint", "effect"):
        for quality in ("fresh", "grounded", "refined", "hallucinated"):
            sigs.add(f"search:{topic}:{quality}")
    # Compute agent
    sigs.add("memory:idle")
    for p in (False, True):
        for r in (False, True):
            for a in (False, True):
                for h in (False, True):
                    cs = MemoryAgentState(
                        active=True, have_params=p, have_raw_n=r, have_adj_n=a,
                        hallucinated=h,
                    )
                    sigs.add(cs.signature())
    # Human
    for phase in ("drafting", "queried", "awaiting", "accepted", "rejected"):
        sigs.add(f"human:{phase}")
    return sorted(sigs)


# ---------------------------------------------------------------------------
# Action / step result shapes
# ---------------------------------------------------------------------------


@dataclass
class GeneralisedAction:
    """One hop from src → dst along a CALLS edge, with a payload text."""
    src: str
    dst: str
    call_type: str   # the CALLS edge label (e.g. "invoke_llm", "delegate")
    text: str = ""


@dataclass
class StepOutcome:
    """Result of one tick.

    ``actor_transition`` is set only when the hop moved an agent between
    two *different* state signatures — that's what gets recorded as a
    TRANSITION edge in Neo4j (keyed by the agent whose state changed).
    """
    state: StateSnapshot
    observation: Observation
    reward: float
    done: bool
    call: GeneralisedAction
    working_state: WorkingState
    # If this hop changed some agent's state, the transition (keyed by state signatures):
    actor_transition: Optional[tuple[str, str, str, str]] = None
    # (agent_id, from_signature, to_signature, tool_call) — action is the tool that caused the change


# ---------------------------------------------------------------------------
# Tool effects — pure functions (agent_state, tool, payload) -> (new_state, reward, response_text)
# ---------------------------------------------------------------------------


def _orch_apply_tool_return(
    orch: OrchestratorState,
    tool: str,
    payload: str,
) -> tuple[OrchestratorState, float, str]:
    """Apply the return of a tool call back to the orchestrator's state.

    Each branch returns ``(new_state, reward, response_text)``. Redundant
    or premature calls (no state change) get a small penalty so the
    world model learns not to repeat them — without this, db writes loop
    forever (same state, +0.15 reward, repeat).
    """
    new = replace(orch)
    if tool == "invoke_llm_plan":
        if orch.planned:
            return new, -0.10, "[redundant: plan already drafted]"
        new.planned = True
        return new, 0.15, "Plan: endpoint → effect → compute → persist → synthesise."
    if tool == "invoke_llm_synth":
        if orch.synth:
            return new, -0.10, "[redundant: synthesis already done]"
        if orch.endpoint and orch.effect and orch.sample:
            new.synth = True
            return new, 0.30, (
                "Recommend ACR20 at week 12 with n=170 (incl. 15% dropout). "
                "Δ≈0.28; Dunnett correction."
            )
        return new, -0.20, "[synthesis failed: missing endpoint/effect/sample_size]"
    if tool == "invoke_db_write_endpoint":
        if orch.endpoint_persisted:
            return new, -0.10, "[redundant: endpoint already persisted]"
        if orch.endpoint:
            new.endpoint_persisted = True
            return new, 0.15, "ack: primary_endpoint persisted."
        return new, -0.10, "[db write failed: no endpoint to persist]"
    if tool == "invoke_db_write_effect":
        if orch.effect_persisted:
            return new, -0.10, "[redundant: effect already persisted]"
        if orch.effect:
            new.effect_persisted = True
            return new, 0.15, "ack: effect_size persisted."
        return new, -0.10, "[db write failed: no effect to persist]"
    if tool == "invoke_db_write_sample":
        if orch.sample_persisted:
            return new, -0.10, "[redundant: sample_size already persisted]"
        if orch.sample:
            new.sample_persisted = True
            return new, 0.20, "ack: sample_size + status persisted."
        return new, -0.10, "[db write failed: no sample_size to persist]"
    if tool == "delegate_search_endpoint":
        if orch.endpoint:
            return new, -0.20, "[redundant: endpoint already obtained]"
        new.endpoint = True
        return new, 0.40, payload or "search returned endpoint"
    if tool == "delegate_search_effect":
        if orch.effect:
            return new, -0.20, "[redundant: effect already obtained]"
        new.effect = True
        return new, 0.40, payload or "search returned effect"
    if tool == "delegate_memory":
        if orch.sample:
            return new, -0.20, "[redundant: sample size already obtained]"
        new.sample = True
        return new, 0.50, payload or "compute returned sample size"
    if tool == "respond":
        # respond's reward is computed by the world from the artefact —
        # this branch is invoked only by the legacy code path; the new
        # _step_orchestrator_respond bypasses it.
        new.responded = True
        return new, 0.0, "[response dispatched]"
    return new, 0.0, "[no-op]"


def _search_apply_tool_return(
    search: SearchAgentState,
    tool: str,
) -> tuple[SearchAgentState, float, str]:
    new = replace(search)
    topic = search.topic
    if tool == "invoke_regulatory_db":
        if topic == "endpoint":
            new.grounded = True
            new.result = "ACR20 at week 12"
            return new, 0.40, "FDA: ACR20 @ week 12 is the conventional Phase II RA endpoint."
        # wrong tool for this topic — small penalty but no state change
        return new, -0.15, "[regulatory_db returned nothing relevant]"
    if tool == "invoke_pubmed":
        if topic == "effect":
            new.grounded = True
            new.result = "0.28"
            return new, 0.40, "Upadacitinib: 62% vs 36% placebo → Δ ≈ 0.28 (ACR20)."
        return new, -0.15, "[pubmed returned nothing relevant]"
    if tool == "invoke_llm":
        if search.grounded:
            new.refined = True
            return new, 0.30, (
                "Refined: confident the endpoint is ACR20@12wk."
                if topic == "endpoint" else
                "Refined: effect size Δ≈0.28 (matching class precedent)."
            )
        # LLM before grounding → hallucinates
        new.hallucinated = True
        return new, -0.30, (
            "Hallucinated endpoint (no grounding data)."
            if topic == "endpoint" else
            "Hallucinated effect estimate (no grounding data)."
        )
    # unknown
    return new, 0.0, "[no-op]"


def _search_apply_return_to_caller(
    search: SearchAgentState,
) -> tuple[float, str, ArtefactFragment]:
    """Reward + caller payload + the artefact fragment the search produced.

    The fragment is what gets deposited into the FinalArtefact's
    ``endpoint`` or ``effect_size`` slot once the orchestrator receives it.
    """
    topic = search.topic or "?"
    if search.hallucinated and not search.refined:
        text = (
            "Hallucinated: ACR50 at week 16 (no source)."
            if topic == "endpoint" else
            "Hallucinated: Δ ≈ 0.45 (no literature support)."
        )
        return -0.40, f"[{topic}] hallucinated — low confidence.", ArtefactFragment(
            status=FRAGMENT_HALLUCINATED, text=text, source="llm-without-grounding",
        )
    if search.refined:
        text = (
            "ACR20 response at week 12 (FDA-aligned, refined)."
            if topic == "endpoint" else
            "Δ ≈ 0.28 (62% drug vs 36% placebo, JAK1 class precedent, refined)."
        )
        src = "regulatory_db+llm" if topic == "endpoint" else "pubmed+llm"
        return 0.80, f"[{topic}] {search.result} (refined).", ArtefactFragment(
            status=FRAGMENT_GROUNDED, text=text, source=src,
        )
    if search.grounded:
        text = (
            "ACR20 response at week 12 (FDA, raw lookup)."
            if topic == "endpoint" else
            "Δ ≈ 0.28 (raw literature value, not refined)."
        )
        src = "regulatory_db" if topic == "endpoint" else "pubmed"
        return 0.30, f"[{topic}] {search.result} (not refined).", ArtefactFragment(
            status=FRAGMENT_GROUNDED, text=text, source=src,
        )
    return -0.60, f"[{topic}] nothing found.", ArtefactFragment(
        status=FRAGMENT_MISSING,
        text=f"[{topic} unknown — search returned no result]",
    )


def _memory_apply_tool_return(
    memory: MemoryAgentState,
    tool: str,
) -> tuple[MemoryAgentState, float, str]:
    new = replace(memory)
    if tool == "invoke_db":
        new.have_params = True
        return new, 0.30, "Params recalled from DB: ratio=1:1:1, dropout=0.15, multiplicity=Dunnett."
    if tool == "invoke_sample_size_calc":
        if memory.have_params:
            new.have_raw_n = True
            return new, 0.40, "Per-arm n=48; total n=144 (pre-dropout) — recalled formula."
        new.hallucinated = True
        return new, -0.30, "Computed without recalled params — result unreliable."
    if tool == "invoke_dropout_adjust":
        if memory.have_raw_n:
            new.have_adj_n = True
            new.result = 170
            return new, 0.40, "Dropout-adjusted: n=170 (policy recalled from memory)."
        new.hallucinated = True
        return new, -0.30, "Dropout adjustment without raw n — result unreliable."
    if tool == "invoke_llm":
        # LLM doesn't directly add value for this agent; mild negative to
        # discourage reflexive LLM calls.
        return new, -0.05, "LLM commentary on the recalled plan."
    return new, 0.0, "[no-op]"


def _evaluate_artefact_for_response(
    artefact: "FinalArtefact",
) -> tuple[float, str, str]:
    """Decide what the human will do when the response arrives.

    Returns ``(reward, decision, justification)``. The reward is large
    positive on accept, large negative on reject — that's the signal the
    orchestrator's world-model uses to learn that ``respond`` is only
    worth choosing once the artefact is actually ready.
    """
    grounded = artefact.grounded_count()
    hallucinated = artefact.hallucinated_count()
    if grounded == 5 and hallucinated == 0:
        return (5.0, "accepted",
                "All five sections grounded; confident submission.")
    if hallucinated > 0:
        return (-3.0, "rejected",
                f"Contains {hallucinated} hallucinated section(s); not safe to submit.")
    if grounded >= 4:
        return (-1.0, "rejected",
                f"Almost there ({grounded}/5 grounded) but missing detail; please complete.")
    return (-2.0, "rejected",
            f"Only {grounded}/5 sections grounded; insufficient evidence.")


def _derive_confidence(artefact: "FinalArtefact") -> ArtefactFragment:
    """Derive the confidence section from the other four sections' status.

    Called when the orchestrator runs ``invoke_llm_synth``. This is the
    moment "synthesis" happens — the artefact is now considered finalised
    with whatever evidence has been collected.
    """
    grounded = sum(
        1 for label, frag in artefact.sections()
        if label != "confidence" and frag.status == FRAGMENT_GROUNDED
    )
    hallucinated = sum(
        1 for label, frag in artefact.sections()
        if label != "confidence" and frag.status == FRAGMENT_HALLUCINATED
    )
    missing = sum(
        1 for label, frag in artefact.sections()
        if label != "confidence" and frag.status == FRAGMENT_MISSING
    )
    if grounded == 4 and hallucinated == 0 and missing == 0:
        return ArtefactFragment(
            status=FRAGMENT_GROUNDED,
            text="High — every section grounded in a tool result.",
            source="synthesis",
        )
    if hallucinated == 0:
        return ArtefactFragment(
            status=FRAGMENT_GROUNDED,
            text=(f"Mixed — {grounded}/4 sections grounded, "
                  f"{missing} missing. Recommend completing the missing "
                  "lookups before submission."),
            source="synthesis",
        )
    return ArtefactFragment(
        status=FRAGMENT_HALLUCINATED,
        text=(f"Low — {hallucinated} section(s) contain hallucinated "
              f"content (no tool grounding). Do NOT submit without "
              "re-running the corresponding searches."),
        source="synthesis",
    )


def _memory_apply_return_to_caller(
    memory: MemoryAgentState,
) -> tuple[float, str, ArtefactFragment]:
    if memory.have_adj_n and not memory.hallucinated:
        text = (f"n = {memory.result} (3 arms, 1:1:1, Dunnett correction, "
                "15% dropout-adjusted; recalled from memory + precedent + "
                "policy pipeline).")
        return 0.80, text, ArtefactFragment(
            status=FRAGMENT_GROUNDED, text=text,
            source="db+sample_size_calc+dropout_adjust",
        )
    if memory.have_raw_n and not memory.hallucinated:
        text = "n = 144 (raw — dropout adjustment not applied)."
        return 0.30, text, ArtefactFragment(
            status=FRAGMENT_GROUNDED, text=text,
            source="db+sample_size_calc",
        )
    if memory.hallucinated:
        return -0.40, "Memory returned an unreliable n.", ArtefactFragment(
            status=FRAGMENT_HALLUCINATED,
            text="n = ?? (recalled without grounded params — unreliable).",
            source="sample_size_calc-without-params",
        )
    return -0.60, "Memory returned no n.", ArtefactFragment(
        status=FRAGMENT_MISSING,
        text="[sample size unknown — memory pipeline incomplete]",
    )


# ---------------------------------------------------------------------------
# Agent toolshelves — what a policy may pick at each agent decision point.
# Each shelf entry is (option_key, description).
# ---------------------------------------------------------------------------


ORCHESTRATOR_SHELF: list[str] = [
    "invoke_llm_plan",
    "invoke_llm_synth",
    "invoke_db_write_endpoint",
    "invoke_db_write_effect",
    "invoke_db_write_sample",
    "delegate_search_endpoint",
    "delegate_search_effect",
    "delegate_memory",
    "respond",
]

SEARCH_AGENT_SHELF: list[str] = [
    "invoke_regulatory_db",
    "invoke_pubmed",
    "invoke_llm",
    "return",
]

MEMORY_AGENT_SHELF: list[str] = [
    "invoke_db",
    "invoke_sample_size_calc",
    "invoke_dropout_adjust",
    "invoke_llm",
    "return",
]


def agent_shelf(agent_id: str) -> list[str]:
    if agent_id == "orchestrator":
        return ORCHESTRATOR_SHELF
    if agent_id == "search_agent":
        return SEARCH_AGENT_SHELF
    if agent_id == "memory_agent":
        return MEMORY_AGENT_SHELF
    return []


# Maps a tool-shelf key to (physical_target_node, call_type) so the world
# can record the visible hop. Some keys share a physical node (e.g.
# ``invoke_llm_plan`` and ``invoke_llm_synth`` both hop to the ``llm`` node)
# — the *call* distinguishes them for world-model purposes.
def shelf_key_to_hop(agent_id: str, key: str) -> tuple[str, str]:
    if agent_id == "orchestrator":
        if key == "invoke_llm_plan" or key == "invoke_llm_synth":
            return ("llm", "invoke_llm")
        if key.startswith("invoke_db_write"):
            return ("db", "invoke_db")
        if key == "delegate_search_endpoint" or key == "delegate_search_effect":
            return ("search_agent", "delegate")
        if key == "delegate_memory":
            return ("memory_agent", "delegate")
        if key == "respond":
            return ("server", "respond")
    if agent_id == "search_agent":
        if key == "invoke_regulatory_db":
            return ("regulatory_db", "invoke_tool")
        if key == "invoke_pubmed":
            return ("pubmed", "invoke_tool")
        if key == "invoke_llm":
            return ("llm", "invoke_llm")
        if key == "return":
            return ("orchestrator", "return")
    if agent_id == "memory_agent":
        if key == "invoke_db":
            return ("db", "invoke_db")
        if key == "invoke_sample_size_calc" or key == "invoke_dropout_adjust":
            # These are "calc" sub-skills of the agent itself — round-trip via
            # the LLM node as a stand-in for internal computation, so every
            # tick is still a hop on the topology.
            return ("llm", "invoke_llm")
        if key == "invoke_llm":
            return ("llm", "invoke_llm")
        if key == "return":
            return ("orchestrator", "return")
    raise ValueError(f"Unknown shelf key {key!r} for agent {agent_id!r}")


# ---------------------------------------------------------------------------
# Policy — consulted only at agent decision points.
# ---------------------------------------------------------------------------


class GeneralisedPolicy(Policy):
    """Orchestrator + worker-agent policy rolled into one.

    ``mode`` switches the strategy at *every* agent (orchestrator, search,
    compute):

    * ``scripted`` — oracle baseline that always picks the next useful tool
      given the current state. Produces the perfect artefact in ~37 ticks.
    * ``random`` — uniform over the agent's toolshelf. Premature LLM calls
      hallucinate, wrong tools waste hops; artefacts come out partial /
      hallucinated. This is the regime that fills the world model with
      causal evidence (good and bad) for the model-based policy to use.
    * ``model`` — at each agent decision, queries ``WorldModel.best_action``
      over the agent's tool shelf at the agent's current state signature.
      Falls back to ``scripted`` when the model has no evidence for the
      current signature. With enough random-mode evidence first, this
      converges on near-scripted artefact quality without being told
      the rules.

    At infra and tool nodes the world determines the next hop deterministically
    — this policy isn't consulted there. ``act()`` always delegates to
    ``world.auto_next_action(self)``.
    """

    def __init__(
        self,
        seed: int | None = None,
        mode: str = "scripted",
        world_model=None,
    ) -> None:
        self.rng = random.Random(seed)
        self.mode = mode
        # Only used when mode == "model"; safe to leave as None otherwise.
        self.world_model = world_model

    def reset(self) -> None:
        pass

    def act(self, world, _state):  # type: ignore[override]
        return world.auto_next_action(self)

    def pick_shelf(self, world, agent_id: str) -> str:
        shelf = agent_shelf(agent_id)
        if self.mode == "random":
            return self.rng.choice(shelf)
        if self.mode == "model" and self.world_model is not None:
            sig = world._agent_state(agent_id).signature()
            plan = self.world_model.best_action(
                room_id=sig,
                candidate_actions=shelf,
                n_rollouts=4,
                horizon=3,
            )
            if plan.used_model and plan.chosen_action:
                return plan.chosen_action
            # Model has no evidence for this signature yet — defer to oracle
            # so the agent still makes forward progress (and the next time
            # this signature is seen the model will have at least one entry).
            return world.scripted_pick(agent_id)
        # scripted (default)
        return world.scripted_pick(agent_id)


# ---------------------------------------------------------------------------
# Call stack
# ---------------------------------------------------------------------------


@dataclass
class StackFrame:
    caller: str                # the node that pushed itself
    # For agent → sub-agent delegations, remembers what the caller is waiting
    # to observe (e.g. orchestrator delegating for "endpoint" or "effect").
    delegation_context: Optional[str] = None
    # For agents: signature at the moment of pushing, so we can record
    # a TRANSITION once the stack pops and the caller's state has advanced.
    entry_signature: Optional[str] = None
    # Tool-shelf key that caused this push (e.g. "delegate_search_endpoint").
    shelf_key: Optional[str] = None


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


class GeneralisedWorld:
    """Scripted / policy-driven call-graph environment.

    Public surface mirrors :class:`graph_world_model.world.GridWorld` enough
    to drop into the existing ``SimulationState`` container.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.tick = 0
        # Physical packet location.
        self.current_node: str = "human"
        # Spatial-parity (kept at 0; canvas ignores them in this context).
        self.agent_x = 0
        self.agent_y = 0
        self.inventory: list[str] = []
        self.entities: dict[str, Entity] = {}
        self.done = False

        self.working_state = WorkingState(task=CANNED_TASK)
        # The artefact the human will see — populated piecewise by tool returns.
        self.artefact = FinalArtefact.empty(CANNED_TASK)
        # Per-agent local state
        self.orch_state = OrchestratorState()
        self.search_state = SearchAgentState()
        self.memory_state = MemoryAgentState()
        self.human_state = HumanState(phase="drafting")
        # Decision computed when orch responds; consumed when human terminates.
        self._pending_human_decision: Optional[str] = None
        self._pending_human_justification: Optional[str] = None
        # Call stack (empty means human is the outer caller)
        self.stack: list[StackFrame] = []
        # Whether the initial request has been emitted by human
        self._emitted_request = False
        # "Deferred delivery": when an agent returns, the caller's state has
        # to be updated on the *return* hop, not during the intermediate ones.
        # When a tool call lands on an agent node, we need to know "who made
        # this call" to apply the right effect on return. We encode that via
        # the top stack frame's shelf_key.
        #
        # State-signature tracking for TRANSITION recording:
        # When an agent makes a decision, we remember its signature-at-decision
        # alongside the shelf_key. When that call returns and the agent's
        # state has changed, we record the transition.
        self._pending_transitions: list[tuple[str, str, str, str]] = []
        # ^ list of (agent_id, from_sig, to_sig, shelf_key) — consumed by
        #   _do_generalised_step in the outer loop.

    # -- API parity with GridWorld ------------------------------------------

    def get_state(self) -> StateSnapshot:
        stype = "initial" if self.tick == 0 else "step"
        return self._make_snapshot(stype)

    def get_entities(self) -> list[Entity]:
        return []

    def _entities_here(self) -> list[Entity]:
        return []

    def valid_actions(self) -> list[GeneralisedAction]:
        # Used by the rollout panel only. The scripted simulation loop calls
        # ``auto_next_action`` directly.
        return [
            GeneralisedAction(src=self.current_node, dst=t, call_type=a)
            for t, a in outgoing(self.current_node)
        ]

    def reset(self, seed: int | None = None) -> StateSnapshot:
        self.__init__(seed=seed)  # type: ignore[misc]
        return self.get_state()

    # -- rollout support ----------------------------------------------------

    def current_agent_signature(self) -> Optional[str]:
        """Signature of the agent currently in control, or None if the
        packet is at an infra/tool node."""
        spec = ACTORS.get(self.current_node)
        if spec is None or spec.kind != "agent":
            return None
        return self._agent_state(self.current_node).signature()

    def current_shelf_keys(self) -> list[str]:
        """Tool-shelf keys valid for the current node (empty at infra/tool)."""
        spec = ACTORS.get(self.current_node)
        if spec is None or spec.kind != "agent":
            return []
        return list(agent_shelf(self.current_node))

    # -- core step ----------------------------------------------------------

    def auto_next_action(self, policy: GeneralisedPolicy) -> GeneralisedAction:
        """Compute the next hop without advancing the world.

        Called by the sim loop to get an Action object that describes the
        upcoming tick (mirrors the GridWorld pattern of ``policy.act → world.step``).
        """
        node = self.current_node
        spec = ACTORS[node]
        if spec.kind == "infra":
            return self._infra_next()
        if spec.kind == "tool":
            return self._tool_return()
        # agent decision point
        shelf_key = policy.pick_shelf(self, node)
        dst, call_type = shelf_key_to_hop(node, shelf_key)
        self._pending_shelf_key = shelf_key  # consumed in step()
        return GeneralisedAction(src=node, dst=dst, call_type=call_type,
                                 text=f"[{shelf_key}]")

    def step(self, action: GeneralisedAction) -> StepOutcome:
        """Advance one tick. ``action`` must be what ``auto_next_action``
        just produced (the pairing keeps the world + UI in sync).
        """
        if self.done:
            raise RuntimeError("Episode finished; call reset().")
        spec = ACTORS[self.current_node]
        if spec.kind == "infra":
            return self._step_infra(action)
        if spec.kind == "tool":
            return self._step_tool_return(action)
        return self._step_agent_decision(action)

    # -- infra --------------------------------------------------------------

    def _infra_next(self) -> GeneralisedAction:
        """Infrastructure nodes are deterministic relays. Direction is fixed
        by whether the orchestrator has issued its final response yet:
        before respond → forward to orchestrator; after → back to human."""
        node = self.current_node
        if self.orch_state.responded:
            if node == "server":
                return GeneralisedAction(src="server", dst="client", call_type="respond",
                                         text=self._last_response_text)
            if node == "client":
                return GeneralisedAction(src="client", dst="human", call_type="respond",
                                         text=self._last_response_text)
            # node == "human" — response has arrived; signal end of episode.
            return GeneralisedAction(src="human", dst="human", call_type="terminate",
                                     text="[episode complete]")
        # Forward journey.
        if node == "human":
            return GeneralisedAction(src="human", dst="client", call_type="request",
                                     text=CANNED_TASK)
        if node == "client":
            return GeneralisedAction(src="client", dst="server", call_type="forward",
                                     text=CANNED_TASK)
        # node == "server"
        return GeneralisedAction(src="server", dst="orchestrator", call_type="dispatch",
                                 text=CANNED_TASK)

    def _step_infra(self, action: GeneralisedAction) -> StepOutcome:
        if action.call_type == "terminate":
            # Human evaluates the artefact: accept iff every section is
            # grounded and nothing hallucinated. The reward was already
            # granted on the orchestrator's ``respond`` hop (so the world
            # model learns to associate respond-from-X with accept-or-reject);
            # this step just makes the human's *decision* visible and
            # records its own per-agent TRANSITION.
            prev_human = self.human_state.signature()
            decision = self._pending_human_decision or "rejected"
            self.human_state.phase = decision  # "accepted" | "rejected"
            self.done = True
            new_human = self.human_state.signature()
            actor_transition = ("human", prev_human, new_human, decision)
            justification = self._pending_human_justification or ""
            text = (
                f"ACCEPTED. {justification}"
                if decision == "accepted" else
                f"REJECTED. {justification}"
            )
            state = self._make_snapshot("terminal")
            obs = self._make_observation(text)
            return StepOutcome(
                state=state, observation=obs, reward=0.0, done=True,
                call=action, working_state=self.working_state,
                actor_transition=actor_transition,
            )
        if self.current_node == "human":
            self._emitted_request = True
            # Human's first action — emit the query.
            prev_human = self.human_state.signature()
            self.human_state.phase = "queried"
            new_human = self.human_state.signature()
            self.current_node = action.dst
            self.tick += 1
            state = self._make_snapshot("step")
            obs = self._make_observation(action.text)
            return StepOutcome(
                state=state, observation=obs, reward=0.0, done=False,
                call=action, working_state=self.working_state,
                actor_transition=("human", prev_human, new_human, "query"),
            )
        self.current_node = action.dst
        self.tick += 1
        state = self._make_snapshot("step")
        obs = self._make_observation(action.text)
        return StepOutcome(
            state=state, observation=obs, reward=0.0, done=False,
            call=action, working_state=self.working_state,
        )

    # -- agent decision -----------------------------------------------------

    def _step_agent_decision(self, action: GeneralisedAction) -> StepOutcome:
        node = self.current_node
        shelf_key = getattr(self, "_pending_shelf_key", None)
        if shelf_key is None:
            # Shouldn't happen — fall back to random pick.
            shelf_key = self.rng.choice(agent_shelf(node))
        self._pending_shelf_key = None
        entry_sig = self._agent_state(node).signature()

        # Special cases: "return" and "respond" don't push a new frame — they
        # pop the existing one (caller resumes).
        if shelf_key == "return":
            return self._step_agent_return(action, entry_sig)
        if shelf_key == "respond":
            return self._step_orchestrator_respond(action, entry_sig)

        # Regular delegation: push a frame and hop out.
        frame = StackFrame(
            caller=node,
            delegation_context=self._delegation_context(node, shelf_key),
            entry_signature=entry_sig,
            shelf_key=shelf_key,
        )
        self.stack.append(frame)

        # If delegating to another agent, initialise the delegate's state.
        if action.dst == "search_agent":
            topic = "endpoint" if shelf_key == "delegate_search_endpoint" else "effect"
            self.search_state = SearchAgentState(topic=topic)
        elif action.dst == "memory_agent":
            self.memory_state = MemoryAgentState(active=True)

        self.current_node = action.dst
        self.tick += 1
        state = self._make_snapshot("step")
        obs = self._make_observation(action.text)
        return StepOutcome(
            state=state, observation=obs, reward=0.0, done=False,
            call=action, working_state=self.working_state,
        )

    def _delegation_context(self, agent_id: str, shelf_key: str) -> Optional[str]:
        if agent_id == "orchestrator":
            return {
                "delegate_search_endpoint": "endpoint",
                "delegate_search_effect":   "effect",
                "delegate_memory":         "compute",
                "invoke_llm_plan":          "plan",
                "invoke_llm_synth":         "synth",
                "invoke_db_write_endpoint": "endpoint",
                "invoke_db_write_effect":   "effect",
                "invoke_db_write_sample":   "sample",
            }.get(shelf_key)
        if agent_id == "search_agent":
            return shelf_key
        if agent_id == "memory_agent":
            return shelf_key
        return None

    def _step_agent_return(
        self, action: GeneralisedAction, entry_sig: str,
    ) -> StepOutcome:
        """Worker agent (search/compute) returning to its caller. The
        agent's accumulated knowledge is *deposited as a fragment* into
        the FinalArtefact at the end of its sub-task — that's the moment
        the system's understanding becomes "official" output."""
        node = self.current_node
        # Compute reward + payload + fragment to deposit.
        if node == "search_agent":
            reward, payload, fragment = _search_apply_return_to_caller(self.search_state)
            slot = "endpoint" if self.search_state.topic == "endpoint" else "effect_size"
            setattr(self.artefact, slot, fragment)
            self._last_response_text = payload
        elif node == "memory_agent":
            reward, payload, fragment = _memory_apply_return_to_caller(self.memory_state)
            self.artefact.sample_size = fragment
            self._last_response_text = payload
        else:
            reward, payload = 0.0, ""
            self._last_response_text = ""

        # The frame at the top of the stack is the one that pushed this
        # agent's delegation (the orchestrator, typically). Pop it and hop.
        caller_frame = self.stack.pop() if self.stack else None
        caller_node = caller_frame.caller if caller_frame else "orchestrator"

        # Apply the sub-result on the orchestrator's state.
        actor_transition = None
        if caller_node == "orchestrator" and caller_frame and caller_frame.shelf_key:
            orch_before_sig = caller_frame.entry_signature or self.orch_state.signature()
            new_orch, delta_r, response_text = _orch_apply_tool_return(
                self.orch_state, caller_frame.shelf_key, payload,
            )
            self.orch_state = new_orch
            reward += delta_r
            self._last_response_text = response_text
            orch_after_sig = self.orch_state.signature()
            actor_transition = (
                "orchestrator", orch_before_sig, orch_after_sig,
                caller_frame.shelf_key,
            )

        # Clear the worker's local state now that it's handed off.
        if node == "search_agent":
            self.search_state = SearchAgentState()
        elif node == "memory_agent":
            self.memory_state = MemoryAgentState()

        self.current_node = caller_node
        self.tick += 1
        state = self._make_snapshot("step")
        obs = self._make_observation(payload)
        return StepOutcome(
            state=state, observation=obs, reward=reward, done=False,
            call=action, working_state=self.working_state,
            actor_transition=actor_transition,
        )

    def _step_orchestrator_respond(
        self, action: GeneralisedAction, entry_sig: str,
    ) -> StepOutcome:
        """Orchestrator sends the final response. Sets ``orch.responded`` —
        which flips the infra nodes into return-journey mode. No stack push:
        the call stack is only for tool/agent delegation, not for the
        infra chain (which runs deterministically by dead reckoning).

        The reward for ``respond`` is the human's accept/reject reward,
        computed from the *current artefact*. Choosing to respond before
        the artefact is ready costs heavily; choosing to respond when
        everything is grounded pays handsomely. This is what makes the
        model toggle visibly worthwhile — the model learns the contract."""
        # Stash the decision for the eventual human-terminate step to reflect
        # in human's local state + payload text.
        reward, decision, justification = _evaluate_artefact_for_response(self.artefact)
        self._pending_human_decision = decision
        self._pending_human_justification = justification

        self.orch_state.responded = True
        self._last_response_text = self.artefact.render_text()

        actor_transition = (
            "orchestrator", entry_sig, self.orch_state.signature(), "respond",
        )
        self.current_node = action.dst  # "server"
        self.tick += 1
        state = self._make_snapshot("step")
        obs = self._make_observation(self._last_response_text)
        return StepOutcome(
            state=state, observation=obs, reward=reward, done=False,
            call=action, working_state=self.working_state,
            actor_transition=actor_transition,
        )

    # -- tool return --------------------------------------------------------

    def _tool_return(self) -> GeneralisedAction:
        """The tool has finished processing; next hop is the return to caller."""
        if not self.stack:
            raise RuntimeError(f"Tool {self.current_node!r} with empty stack")
        caller = self.stack[-1].caller
        return GeneralisedAction(src=self.current_node, dst=caller,
                                 call_type="return", text="[tool response]")

    def _step_tool_return(self, action: GeneralisedAction) -> StepOutcome:
        """Tool returns to its caller. Apply the tool's effect on the caller's
        local state (if caller is an agent) and record a TRANSITION edge on
        that agent."""
        caller_frame = self.stack.pop() if self.stack else None
        if caller_frame is None:
            raise RuntimeError("Tool return with empty call stack")
        caller = caller_frame.caller
        shelf_key = caller_frame.shelf_key
        entry_sig = caller_frame.entry_signature

        reward = 0.0
        text = "[tool responded]"
        actor_transition: Optional[tuple[str, str, str, str]] = None

        # Apply the tool effect to the calling agent's state.
        if caller == "orchestrator" and shelf_key is not None:
            # Update working_state if this was a db write whose precondition held.
            if shelf_key == "invoke_db_write_endpoint" and self.orch_state.endpoint:
                self.working_state.primary_endpoint = "ACR20 at week 12"
            elif shelf_key == "invoke_db_write_effect" and self.orch_state.effect:
                self.working_state.effect_size = 0.28
            elif shelf_key == "invoke_db_write_sample" and self.orch_state.sample:
                self.working_state.sample_size = 170
                self.working_state.status = "done"

            new, delta_r, response = _orch_apply_tool_return(
                self.orch_state, shelf_key, "",
            )
            self.orch_state = new
            reward = delta_r
            text = response

            # Synthesis derives the artefact's confidence section from
            # the grounded/hallucinated/missing status of the other four.
            if shelf_key == "invoke_llm_synth":
                self.artefact.confidence = _derive_confidence(self.artefact)
            actor_transition = (
                "orchestrator",
                entry_sig or self.orch_state.signature(),
                self.orch_state.signature(),
                shelf_key,
            )
        elif caller == "search_agent" and shelf_key is not None:
            new, delta_r, response = _search_apply_tool_return(
                self.search_state, shelf_key,
            )
            self.search_state = new
            reward = delta_r
            text = response
            actor_transition = (
                "search_agent",
                entry_sig or self.search_state.signature(),
                self.search_state.signature(),
                shelf_key,
            )
        elif caller == "memory_agent" and shelf_key is not None:
            new, delta_r, response = _memory_apply_tool_return(
                self.memory_state, shelf_key,
            )
            self.memory_state = new
            reward = delta_r
            text = response
            actor_transition = (
                "memory_agent",
                entry_sig or self.memory_state.signature(),
                self.memory_state.signature(),
                shelf_key,
            )

        self._last_response_text = text
        self.current_node = caller
        self.tick += 1
        state = self._make_snapshot("step")
        obs = self._make_observation(text)
        return StepOutcome(
            state=state, observation=obs, reward=reward, done=False,
            call=action, working_state=self.working_state,
            actor_transition=actor_transition,
        )

    # -- scripted policy helpers --------------------------------------------

    def scripted_pick(self, agent_id: str) -> str:
        """Oracle next move for the given agent. Follows the canonical
        clinical-protocol plan. Used when ``GeneralisedPolicy.mode ==
        'scripted'`` (the default)."""
        if agent_id == "orchestrator":
            o = self.orch_state
            if not o.planned:
                return "invoke_llm_plan"
            if not o.endpoint:
                return "delegate_search_endpoint"
            if not o.endpoint_persisted:
                return "invoke_db_write_endpoint"
            if not o.effect:
                return "delegate_search_effect"
            if not o.effect_persisted:
                return "invoke_db_write_effect"
            if not o.sample:
                return "delegate_memory"
            if not o.sample_persisted:
                return "invoke_db_write_sample"
            if not o.synth:
                return "invoke_llm_synth"
            return "respond"
        if agent_id == "search_agent":
            s = self.search_state
            if s.result is None:
                return "invoke_regulatory_db" if s.topic == "endpoint" else "invoke_pubmed"
            if not s.refined:
                return "invoke_llm"
            return "return"
        if agent_id == "memory_agent":
            c = self.memory_state
            if not c.have_params:
                return "invoke_db"
            if not c.have_raw_n:
                return "invoke_sample_size_calc"
            if not c.have_adj_n:
                return "invoke_dropout_adjust"
            return "return"
        return agent_shelf(agent_id)[0]

    # -- internal helpers ---------------------------------------------------

    def _agent_state(self, agent_id: str):
        return {
            "orchestrator":  self.orch_state,
            "search_agent":  self.search_state,
            "memory_agent": self.memory_state,
            "human":         self.human_state,
        }[agent_id]

    @property
    def _last_response_text(self) -> str:
        return getattr(self, "__last_response_text", "")

    @_last_response_text.setter
    def _last_response_text(self, v: str) -> None:
        self.__last_response_text = v

    def _make_snapshot(self, stype: str = "step") -> StateSnapshot:
        # room_id = the *state-signature* the world model should learn over.
        # Agents (and the human, which is also an agent) have their own
        # signatures; tool nodes borrow the caller's signature; client /
        # server fall back to the actor id (no real state worth keying on).
        node = self.current_node
        spec = ACTORS[node]
        if node in ("orchestrator", "search_agent", "memory_agent", "human"):
            room = self._agent_state(node).signature()
        elif spec.kind == "tool" and self.stack:
            caller = self.stack[-1].caller
            if caller in ("orchestrator", "search_agent", "memory_agent", "human"):
                room = self._agent_state(caller).signature()
            else:
                room = node
        else:
            room = node
        return StateSnapshot(
            id=str(uuid.uuid4()),
            tick=self.tick,
            room_id=room,
            agent_x=0,
            agent_y=0,
            inventory=[],
            done=self.done,
            type=stype,
        )

    def _make_observation(self, _text: str) -> Observation:
        return Observation(
            id=str(uuid.uuid4()),
            tick=self.tick,
            sensor="text",
            visible_entities=[],
            distances=[],
            goal_achieved=self.done,
        )


# ---------------------------------------------------------------------------
# Subgraph builder
# ---------------------------------------------------------------------------


def build_generalised_subgraph(graph: GraphStore) -> WorkingState:
    """Pre-seed only the static actor topology — Actor nodes + :CALLS edges.

    AgentState signature Rooms and the WorkingState document are created
    lazily on demand (see :func:`ensure_agent_state` and the WorkingState
    upsert in ``_do_generalised_step``). This keeps a freshly reset graph
    view limited to the static topology, mirroring the spatial/scene demos.
    """
    for actor in ACTORS.values():
        graph.run_write(
            """
            MERGE (a:Actor {id: $id, context: $context})
            SET a.kind  = $kind,
                a.role  = $role,
                a.name  = $label,
                a.x     = $x,
                a.y     = $y
            """,
            id=actor.id, kind=actor.kind, role=actor.role, label=actor.label,
            x=int(actor.pos[0] * 100), y=int(actor.pos[1] * 100),
        )

    # CALLS topology — semantic edges exposed in the Cypher console.
    for src, dst, action in TOPOLOGY:
        graph.run_write(
            """
            MATCH (a:Actor {id: $src, context: $context}),
                  (b:Actor {id: $dst, context: $context})
            MERGE (a)-[rel:CALLS {action: $action}]->(b)
            SET rel.context = $context
            """,
            src=src, dst=dst, action=action,
        )

    return WorkingState(task=CANNED_TASK)


def ensure_agent_state(graph: GraphStore, signature: str) -> None:
    """Lazily MERGE an :AgentState:Room node for the given signature, plus
    an :OF edge to the owning Actor so the per-tick subgraph stays joined
    to the static actor topology.
    """
    agent_prefix = signature.split(":", 1)[0]
    graph.run_write(
        """
        MERGE (r:AgentState:Room {id: $id, context: $context})
        SET r.agent = $agent, r.signature = $id
        WITH r
        MATCH (a:Actor {id: $agent, context: $context})
        MERGE (r)-[rel:OF {context: $context}]->(a)
        """,
        id=signature,
        agent=agent_prefix,
    )
