"""The thin per-benchmark connector — the ONLY code a benchmark author writes.

A benchmark does NOT hand lp-bench an environment or a forkable state. It points at
three things (plus optional credentials):

  * the task            — the prompt / goal
  * the read-only tools — CALLABLE, so lp-bench can discover state by reading
  * the ground truth    — the authorized calls, and attack targets if the benchmark has them

lp-bench's Explorer then figures out the minimal task-relevant state by *calling* the
read-only tools. There is no `materialize`, no env object, no write access. This is
the entire extensibility surface: implement `Connector`, `@register_connector`, done.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ajar.core.types import Candidate, HarmTier


@runtime_checkable
class ReadTool(Protocol):
    """A read-only tool the Explorer may actually invoke. `call` executes the real
    read API with credentials already bound by the connector. MUST be side-effect
    free — the connector is responsible for only exposing genuine reads here."""

    name: str
    params_schema: Mapping[str, Any]
    resource_domain: str | None

    def call(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TaskDescriptor:
    task_id: str
    prompt: str
    extra: Mapping[str, Any] = field(default_factory=dict)  # benchmark-specific hints


@dataclass(frozen=True)
class AttackSpec:
    """An attacker objective the benchmark ships, if any. `target_calls` are the
    write calls the attack tries to induce — the H4 anchor for attacker-seeded probes."""

    attack_id: str
    target_calls: Sequence[Candidate]
    harm_hint: HarmTier = HarmTier.H4
    description: str = ""


@dataclass(frozen=True)
class ArgSpec:
    """One authorized argument, annotated by what the SUCCESS PREDICATE does with it.

    This annotation is the whole point of the representation. AgentDojo's raw
    `ground_truth()` gives `date="2022-01-01"` and `recipient="UK..890"` as flat,
    equal-looking args — but `utility()` checks only the recipient. Labeling them
    identically would score a filler-arg mutation as a real over-grant. So each arg
    carries how the predicate treats it.
    """

    value: Any
    load_bearing: bool | None = None  # predicate-checked? None = no predicate to triage against
    matcher: str = "exact"  # exact | case_insensitive | membership | range | ignored
    provenance: str = "unknown"  # constant | read_derived | unknown
    source_read: str | None = None  # if read_derived: which read tool surfaced the value


@dataclass(frozen=True)
class AuthorizedCall:
    tool: str
    args: Mapping[str, ArgSpec]

    @property
    def candidate(self) -> Candidate:
        return Candidate(self.tool, {a: s.value for a, s in self.args.items()})

    def load_bearing_args(self) -> dict[str, Any]:
        return {a: s.value for a, s in self.args.items() if s.load_bearing}


@dataclass(frozen=True)
class GroundTruth:
    """What the benchmark authored, in the form our tool wants it.

    `authorized` is the legitimate solution as an ARGUMENT-ANNOTATED plan, not a flat
    call list — each arg says whether the predicate checks it and how. `attacks` are
    optional (present => attacker probes are ground-truth-derived; absent => the LLM
    proposes them, weaker). `success_predicate` is an opaque, connector-resolvable
    handle to the benchmark's own oracle, kept for EXECUTION confirmation at run time
    (the read-only generation phase can't run it, but the running phase can)."""

    authorized: Sequence[AuthorizedCall]
    attacks: Sequence[AttackSpec] = ()
    success_predicate: Any | None = None

    def plan(self) -> list[Candidate]:
        return [c.candidate for c in self.authorized]


@runtime_checkable
class Connector(Protocol):
    name: str

    def list_tasks(self) -> Sequence[str]: ...

    def task(self, task_id: str) -> TaskDescriptor: ...

    def read_tools(self, task_id: str) -> Sequence[ReadTool]: ...

    def ground_truth(self, task_id: str) -> GroundTruth: ...


_CONNECTORS: dict[str, Connector] = {}


def register_connector(c: Connector) -> Connector:
    _CONNECTORS[c.name] = c
    return c


def get_connector(name: str) -> Connector:
    if name not in _CONNECTORS:
        raise KeyError(f"unknown connector {name!r}; registered: {sorted(_CONNECTORS)}")
    return _CONNECTORS[name]
