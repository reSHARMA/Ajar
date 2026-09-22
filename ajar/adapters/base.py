"""The benchmark extension seam.

To add a benchmark: implement `BenchmarkAdapter` (produce `TaskSpec`s, materialize
states, provide an oracle + harm model) and `@register_benchmark` it. Nothing else
in lp-bench changes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ajar.core.ir import TaskSpec
from ajar.core.oracle import Oracle
from ajar.core.state import State
from ajar.core.types import Candidate, ContextRef, HarmTier, TaskKey


@runtime_checkable
class HarmModel(Protocol):
    """harm(tool, args, state) -> tier. AUTHORED, not measured (env-diff size
    anti-correlates with harm: rho = -0.645). Recipient/target args select the
    KIND (tier); magnitude modulates within a tier and only promotes at its own
    qualitative boundary. H4 is anchored to the benchmark's own attack sinks.
    """

    def tier(self, candidate: Candidate, state: State | None = None) -> HarmTier: ...


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str

    def list_tasks(self) -> Sequence[TaskKey]: ...

    def task_spec(self, key: TaskKey) -> TaskSpec: ...

    def materialize(self, context: ContextRef) -> State: ...

    def oracle(self, key: TaskKey) -> Oracle: ...

    def harm_model(self, key: TaskKey) -> HarmModel: ...

    def value_pool(self, context: ContextRef, tool: str, arg: str) -> Sequence[Any]:
        """Env-derived, plausible-but-not-authorized values for one argument
        (real object ids, other accounts, other file paths). Powers wrong-resource
        and partition generation without inventing values. May return [].

        CONTRACT: must NOT include the argument's authorized (ground-truth) value —
        returning it would mint a candidate byte-identical to a required-ALLOW probe.
        (build_suite resolves such collisions in favor of ALLOW, but keep the pool
        clean so off-path/wrong-resource probes stay genuinely excess.)"""
        ...


_BENCHMARKS: dict[str, BenchmarkAdapter] = {}


def register_benchmark(adapter: BenchmarkAdapter) -> BenchmarkAdapter:
    _BENCHMARKS[adapter.name] = adapter
    return adapter


def get_benchmark(name: str) -> BenchmarkAdapter:
    if name not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; registered: {sorted(_BENCHMARKS)}")
    return _BENCHMARKS[name]


def list_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS)
