"""Reference systems under test — the baselines every benchmark run needs.

allow_all   the benchmark default (suite-wide tools) — the over-privilege ceiling
deny_all    the degenerate case the sufficiency gate must catch
tool_allowlist  the "perfect" least-privilege policy at TOOL granularity
arg_policy  a hand-written argument-level policy (the only thing that survives)

These four make the headline finding legible: tool_allowlist passes the gate at
S=1.0 and admits the attacks; only arg_policy denies them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ajar.core.ir import TaskSpec
from ajar.core.types import Candidate, ContextRef, Verdict
from ajar.defenses.base import register_defense


@dataclass
class _Simple:
    name: str
    privilege_surface: tuple[str, ...]
    _decide: Callable[[Any, ContextRef, Candidate], Verdict]
    _prepare: Callable[[TaskSpec], Any] = lambda spec: spec

    def prepare(self, spec: TaskSpec, llm=None) -> Any:
        return self._prepare(spec)

    def decide(self, policy: Any, context: ContextRef, candidate: Candidate) -> Verdict:
        return self._decide(policy, context, candidate)


register_defense(_Simple("allow_all", ("TOOL", "PARAMETER", "SEQUENCE", "PROVENANCE"),
                         lambda pol, ctx, c: Verdict.ALLOW))

register_defense(_Simple("deny_all", ("TOOL", "PARAMETER", "SEQUENCE", "PROVENANCE"),
                         lambda pol, ctx, c: Verdict.DENY))


def _tool_allowlist_prepare(spec: TaskSpec) -> frozenset[str]:
    # The "minimal" policy: exactly the tools the plan requires. Deliberately does
    # NOT include the read closure — that is the whole point (S=1.0 yet breaks runs).
    return spec.required_tools


register_defense(_Simple(
    "tool_allowlist", ("TOOL",),
    _decide=lambda allow, ctx, c: Verdict.ALLOW if c.tool in allow else Verdict.DENY,
    _prepare=_tool_allowlist_prepare,
))


@dataclass
class ArgPolicy:
    """A hand-written argument-level policy: for each required call, pin the
    load-bearing args to their authorized values. Everything else denied."""

    name = "arg_policy"
    privilege_surface = ("TOOL", "PARAMETER")

    def prepare(self, spec: TaskSpec, llm=None) -> dict[str, dict]:
        rules: dict[str, dict] = {}
        for c in spec.plan:
            # v0 heuristic: pin every arg present in the plan call. A real policy
            # would pin only load-bearing args (mutation-triaged); pinning all is
            # a strict upper bound and still admits the discovery reads via tool.
            rules.setdefault(c.tool, dict(c.args))
        for tool in spec.read_closure:
            rules.setdefault(tool, {})  # reads allowed at tool granularity
        return rules

    def decide(self, policy: dict, context: ContextRef, candidate: Candidate) -> Verdict:
        if candidate.tool not in policy:
            return Verdict.DENY
        required = policy[candidate.tool]
        for arg, val in required.items():
            got = candidate.args.get(arg)
            if isinstance(val, str) and isinstance(got, str):
                if got.lower() != val.lower():
                    return Verdict.DENY
            elif got != val:
                return Verdict.DENY
        return Verdict.ALLOW


register_defense(ArgPolicy())
