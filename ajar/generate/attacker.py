"""§3.6 attacker-seeded probes — the H4 anchor, the only ORACLE_CONFIRMED DENYs.

Lift each attack's sink set (already reduced to ablation-necessary calls by the
adapter) and offer it at every state, re-evaluating the attack's own predicate.
These are the probes commensurable with the benchmark's attack-success metric.
"""
from __future__ import annotations

from collections.abc import Iterable

from ajar.core.types import Axis, FaultClass, Label, Probe
from ajar.generate.base import GenContext, mk_probe, register_generator


@register_generator("3.6-attacker-seeded")
def attacker_seeded(gc: GenContext) -> Iterable[Probe]:
    spec = gc.spec
    # The sink index disambiguates the sinks of a multi-call attack, which otherwise
    # share an id at a given decision point. It is already this codebase's name for a
    # sink: AttackSink.jointly_with is a tuple of sink indices.
    for idx, sink in enumerate(spec.attack_sinks):
        for k, _state, ctx in gc.states():
            yield mk_probe(
                gc,
                pid=f"AS-{sink.attack_id}-s{idx}-k{k}",
                ctx=ctx,
                candidate=sink.candidate,
                proposed_label=Label.DENY,
                axis=(Axis.PARAMETER if sink.candidate.tool in spec.permissible_tools else Axis.TOOL),
                fault=FaultClass.ATTACKER_SEEDED,
                rationale=f"attacker sink of {sink.attack_id}: {sink.evidence}",
                generated_by="ATTACKER_TARGET",
            )
