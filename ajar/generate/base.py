"""Probe-generation framework: generators PROPOSE, the oracle DISPOSES.

A generator is a callable `(GenContext) -> Iterable[Probe]` that emits probes with
a *proposed* label. `dispose()` then overwrites the label from the oracle (upgrading
to ORACLE_CONFIRMED, or leaving it ASSERTED / flipping it to UNKNOWN) and stamps the
harm tier from the benchmark's harm model. `build_suite()` runs every registered
generator and disposes the union.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable

from ajar.adapters.base import BenchmarkAdapter, HarmModel
from ajar.core.ir import TaskSpec
from ajar.core.oracle import Oracle
from ajar.core.state import State
from ajar.core.types import (
    Candidate,
    ContextRef,
    FaultClass,
    HarmTier,
    Label,
    LabelSource,
    Probe,
)
from ajar.llm.base import LLMClient


@dataclass
class GenContext:
    spec: TaskSpec
    adapter: BenchmarkAdapter
    oracle: Oracle
    harm: HarmModel
    llm: LLMClient | None = None

    def state_at(self, prefix: Sequence[int], injections=None, label: str = "") -> tuple[State, ContextRef]:
        ctx = ContextRef(
            task=self.spec.task,
            injections=dict(injections or {}),
            prefix=tuple(prefix),
            label=label or f"k={len(tuple(prefix))}",
        )
        return self.adapter.materialize(ctx), ctx

    def states(self) -> Iterable[tuple[int, State, ContextRef]]:
        """Yield (k, state, ctx) for every plan prefix k = 0 .. len(plan)."""
        for k in range(self.spec.num_states()):
            state, ctx = self.state_at(range(k), label=f"k={k}")
            yield k, state, ctx


Generator = Callable[[GenContext], Iterable[Probe]]

_GENERATORS: list[tuple[str, Generator]] = []


def register_generator(family: str) -> Callable[[Generator], Generator]:
    def deco(fn: Generator) -> Generator:
        _GENERATORS.append((family, fn))
        return fn

    return deco


def registered_generators() -> list[tuple[str, Generator]]:
    return list(_GENERATORS)


def dispose(probe: Probe, gc: GenContext) -> Probe:
    """Replace the proposed label with the oracle's verdict and author the harm."""
    state, _ = gc.state_at(probe.context.prefix, probe.context.injections)
    label, source, evidence = gc.oracle.label(state, probe.candidate)

    if label is Label.UNKNOWN:
        # Oracle cannot witness. A required-ALLOW keeps ALLOW (ASSERTED); an
        # over-grant keeps its proposed DENY (ASSERTED); a provenance probe with
        # no carrier stays UNKNOWN (excluded from OPL, counted as a coverage hole).
        if probe.fault_class in (FaultClass.PROVENANCE,):
            probe.expected_label = Label.UNKNOWN
        probe.label_source = LabelSource.ASSERTED
        probe.oracle_evidence = evidence or "no benchmark predicate witnesses this (s, c)"
    else:
        probe.expected_label = label
        probe.label_source = LabelSource.ORACLE_CONFIRMED
        probe.oracle_evidence = evidence

    # Harm is authored by the harm model — the single source of truth, never the
    # generator. UNDER_GRANT ALLOW probes carry no tier.
    if probe.fault_class is FaultClass.UNDER_GRANT:
        probe.harm_tier = HarmTier.H0
    else:
        probe.harm_tier = gc.harm.tier(probe.candidate, state)

    # Prune H0 over-grants (semantically equivalent to correct — no widening).
    # H0 carries weight zero, so a DENY probe left in it could never penalize an
    # over-grant. That must hold by construction, not because no required call in the
    # current suite happens to coincide with an attack sink: such a call is labelled
    # DENY by the oracle, keeps its UNDER_GRANT class, and is forced to H0 above,
    # which is the one way a zero-weight DENY can reach scoring.
    zero_weight_deny = probe.harm_tier is HarmTier.H0 and probe.expected_label is Label.DENY
    if zero_weight_deny or (probe.harm_tier is HarmTier.H0
                            and probe.fault_class is not FaultClass.UNDER_GRANT):
        probe.pruned = True
    return probe


def build_suite(gc: GenContext, generators=None) -> list[Probe]:
    generators = generators if generators is not None else registered_generators()
    proposed: list[Probe] = []
    for _family, gen in generators:
        proposed.extend(gen(gc))
    disposed = [dispose(p, gc) for p in proposed]
    # Deduplicate by (state prefix, candidate) — families overlap heavily. On a
    # collision the more authoritative label wins, so generator import order never
    # decides a probe: ORACLE_CONFIRMED beats ASSERTED, and a required-ALLOW beats
    # an asserted over-grant DENY on the byte-identical call (it IS authorized).
    def _rank(p: Probe) -> tuple[int, int]:
        confirmed = 1 if p.label_source is LabelSource.ORACLE_CONFIRMED else 0
        required_allow = 1 if (p.expected_label is Label.ALLOW and p.required) else 0
        return (confirmed, required_allow)

    seen: dict[tuple, Probe] = {}
    for p in disposed:
        k = (p.context.prefix, p.candidate.key())
        if k not in seen or _rank(p) > _rank(seen[k]):
            seen[k] = p
    suite = list(seen.values())

    # Attach a natural-language prompt to each probe (mechanical floor; fluent when
    # gc.llm is set). Done here so every path that builds a suite gets it.
    from ajar.generate.prompting import render_prompts

    render_prompts(suite, gc)

    # Attach each probe's success predicate ('ground truth') — how a run's post-state
    # decides whether the probe's action landed. Generated here, with the probe.
    from ajar.generate.groundtruth import attach_ground_truth

    attach_ground_truth(suite, gc)
    return suite


def mk_probe(
    gc: GenContext,
    *,
    pid: str,
    ctx: ContextRef,
    candidate: Candidate,
    proposed_label: Label,
    axis,
    fault: FaultClass,
    rationale: str,
    generated_by: str,
    provenance=None,
    required: bool = True,
) -> Probe:
    return Probe(
        probe_id=pid,
        task=gc.spec.task,
        context=ctx,
        candidate=candidate,
        expected_label=proposed_label,
        label_source=LabelSource.ASSERTED,
        axis=axis,
        fault_class=fault,
        harm_tier=HarmTier.H0,  # set by dispose()
        rationale=rationale,
        provenance=provenance,
        generated_by=generated_by,
        required=required,
    )
