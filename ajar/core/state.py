"""The materialized, executable state a probe is evaluated at."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ajar.core.types import Candidate


@dataclass
class StateOutcome:
    """What the benchmark's success predicates say about the current state.

    `utility` is the user-task predicate; `security` is the bound injection
    task's predicate if one is bound (else None). `trace` is the executed call
    list — carried because some predicates grade from traces, not final state.
    """

    utility: bool | None
    security: bool | None
    trace: Sequence[Candidate]


@runtime_checkable
class State(Protocol):
    """A materialized benchmark state at some plan prefix.

    Obtained from `BenchmarkAdapter.materialize(context)`. Generators and the
    oracle `fork()` before executing so the shared state is never mutated.
    `raw()` exposes the benchmark-native object for that benchmark's own oracle;
    benchmark-agnostic code must not touch it.
    """

    def fork(self) -> "State": ...

    def execute(self, candidate: Candidate) -> tuple[Any, str | None]:
        """Run one call. Returns (result, error). Never raises on tool error —
        the error string is the signal (a raised call is an ontological DENY)."""

    def outcome(self, against: str | None = None) -> StateOutcome:
        """Grade the current state. `against` optionally names an injection task
        whose `security()` should be evaluated (for attacker-seeded probes)."""

    def fingerprint(self) -> str:
        """Stable digest of observable state — for determinism / replay checks."""

    def raw(self) -> Any:
        """The benchmark-native state object. Adapter-internal use only."""
