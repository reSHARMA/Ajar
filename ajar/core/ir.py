"""The canonical ground-truth intermediate representation.

A `BenchmarkAdapter` does conceptually one thing: turn its benchmark's native,
idiosyncratic ground truth into a `TaskSpec`. Everything downstream — generators,
oracle, scorer — speaks only `TaskSpec`. Add a benchmark => produce `TaskSpec`s.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ajar.core.types import Candidate, HarmTier, TaskKey, ToolSpec


@dataclass(frozen=True)
class AttackSink:
    """One attacker-necessary call, lifted from a benchmark's attack ground truth
    and reduced to its sink set (leave-one-out against the attack's own predicate).
    """

    attack_id: str
    candidate: Candidate
    harm_tier: HarmTier
    jointly_with: tuple[int, ...] = ()  # sibling sink indices that must ALL fire
    evidence: str = ""


@dataclass(frozen=True)
class TaskSpec:
    """Everything a generator needs about one task, in benchmark-neutral form."""

    task: TaskKey
    prompt: str
    tools: Sequence[ToolSpec]  # the GRANTED set (suite-wide on AgentDojo)
    plan: Sequence[Candidate]  # ground_truth(pre), FROZEN at k=0
    necessary: frozenset[int] = frozenset()  # ablation-measured; INFORMATIONAL only
    #   (a read/write classifier — never used to narrow `required_tools`; see §7.3)
    read_closure: frozenset[str] = frozenset()  # legit extra reads — LP-BENCH ground truth
    permitted_write_paths: frozenset[str] | None = None  # from state-diff allowlists
    attack_sinks: Sequence[AttackSink] = ()
    is_gradeable: bool = True  # False => vacuous success predicate; report separately
    notes: str = ""

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    @property
    def granted_tools(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tools)

    @property
    def required_tools(self) -> frozenset[str]:
        # ALL tools in the authored completion — deliberately NOT narrowed to the
        # ablation-necessary partition. Narrowing here would make the sufficiency
        # gate pass a defense that denies every discovery read (design memo §7.3).
        # `necessary` is informational (a read/write classifier), never a filter.
        return frozenset(c.tool for c in self.plan)

    @property
    def permissible_tools(self) -> frozenset[str]:
        """Required write spine PLUS the authored read closure. This — not the
        raw plan — is the correct ALLOW set (the 9/67 discovery-read finding)."""
        return self.required_tools | self.read_closure

    def num_states(self) -> int:
        return len(self.plan) + 1  # k = 0 .. len(plan)
