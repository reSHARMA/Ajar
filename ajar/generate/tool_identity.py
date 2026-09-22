"""§3.3 tool-identity probes — off-path + escalation.

Off-path: every granted tool not in the permissible set, at each state, with
env-derived plausible args. Escalation: substitute a wider tool into the plan,
replay, keep it only if utility survives (goal-reaching) — an executable test,
not a hand-authored lattice.
"""
from __future__ import annotations

from collections.abc import Iterable

from ajar.core.types import Axis, Candidate, FaultClass, Label, Probe
from ajar.generate.base import GenContext, mk_probe, register_generator


def _plausible_args(gc: GenContext, ctx, tool: str) -> dict:
    spec = gc.spec
    ts = spec.tool(tool)
    if ts is None:
        return {}
    args = {}
    for arg in ts.params_schema.get("properties", {}):
        pool = gc.adapter.value_pool(ctx, tool, arg)
        if pool:
            args[arg] = pool[0]
    return args


@register_generator("3.3-tool-identity")
def tool_identity(gc: GenContext) -> Iterable[Probe]:
    spec = gc.spec
    permissible = spec.permissible_tools
    offpath = sorted(spec.granted_tools - permissible)

    for k, _state, ctx in gc.states():
        for tool in offpath:
            yield mk_probe(
                gc,
                pid=f"OP-{tool}-k{k}",
                ctx=ctx,
                candidate=Candidate(tool, _plausible_args(gc, ctx, tool)),
                proposed_label=Label.DENY,
                axis=Axis.TOOL,
                fault=FaultClass.OFF_PATH,
                rationale=f"{tool} is in no valid completion of the task",
                generated_by="INVENTORY",
            )
