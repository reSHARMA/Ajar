"""The defense (system-under-test) extension seam.

To add a defense: implement `Defense` and `@register_defense` it. A defense turns
a task into a `Policy` (possibly by calling the injected LLM), then decides each
probe. Stateless tool-filters, arg-level policies, and stateful external systems
(Progent) all fit — a stateful defense snapshots/restores inside `decide`.

Note: two of AgentDojo's four shipped defenses are prompt-injection *detectors*,
not authorization policies — they have no `decide()`. Represent those with
`privilege_surface = ()` so the scorer reports them as LPD-undefined instead of
scoring a prompt-hardener as maximally over-privileged.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ajar.core.ir import TaskSpec
from ajar.core.types import Candidate, ContextRef, Verdict
from ajar.llm.base import LLMClient


class PolicyUnavailable(Exception):
    """Raised by `prepare` when a defense cannot produce a policy for a task
    (e.g. an offline-replay SUT has no logged run). The runner skips that
    (defense, task) pair and records it as uncovered, never as a score."""


@runtime_checkable
class Defense(Protocol):
    name: str
    # Which axes this defense can even constrain. A tool-name filter is ("TOOL",);
    # the scorer only charges it on those axes and reports the rest as N/A.
    privilege_surface: tuple[str, ...]

    def prepare(self, spec: TaskSpec, llm: LLMClient | None = None) -> Any:
        """Produce an opaque policy for this task. May call the LLM. Called once."""
        ...

    def decide(self, policy: Any, context: ContextRef, candidate: Candidate) -> Verdict:
        """Authorize (or not) one candidate call at one state."""
        ...


_DEFENSES: dict[str, Defense] = {}


def register_defense(defense: Defense) -> Defense:
    _DEFENSES[defense.name] = defense
    return defense


def get_defense(name: str) -> Defense:
    if name not in _DEFENSES:
        raise KeyError(f"unknown defense {name!r}; registered: {sorted(_DEFENSES)}")
    return _DEFENSES[name]


def list_defenses() -> list[str]:
    return sorted(_DEFENSES)
