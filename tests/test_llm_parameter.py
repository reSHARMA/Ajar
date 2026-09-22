"""The LLM parameter generator (§3.2-llm) with a FAKE client — no Bedrock, offline.

Asserts the seam's contract: the generator is inert without a client, it emits
DENY / PARAMETER / WRONG_RESOURCE over-grants for the values a client proposes, it
never re-mints the authorized value, it respects the per-arg cap, and a client that
raises degrades to an empty (never a crashing) suite.
"""
from __future__ import annotations

import pytest

from ajar.core.types import Axis, FaultClass, Label, TaskKey
from ajar.generate.base import GenContext
from ajar.generate.llm_parameter import _MAX_PROPOSALS_PER_ARG, llm_parameter

KEY = TaskKey(benchmark="agentdojo", suite="banking", version=(1, 2, 2), task_id="user_task_0")


@pytest.fixture(scope="module")
def adapter():
    import ajar.adapters.agentdojo.adapter  # noqa: F401  (registers)
    from ajar.adapters.base import get_benchmark

    return get_benchmark("agentdojo")


def _gc(adapter, llm) -> GenContext:
    return GenContext(
        spec=adapter.task_spec(KEY),
        adapter=adapter,
        oracle=adapter.oracle(KEY),
        harm=adapter.harm_model(KEY),
        llm=llm,
    )


class _FakeLLM:
    """Returns a fixed proposal list for every arg. Value-typed so it slots into any
    argument; the generator's job is to filter, not to interpret."""

    def __init__(self, proposals, raise_on_call=False):
        self._proposals = proposals
        self._raise = raise_on_call

    def complete(self, messages, **kwargs):  # noqa: ANN001
        return ""

    def complete_json(self, messages, schema, **kwargs):  # noqa: ANN001
        if self._raise:
            raise RuntimeError("bedrock unavailable")
        return list(self._proposals)


def test_inert_without_client(adapter):
    assert list(llm_parameter(_gc(adapter, None))) == []


def test_emits_deny_parameter_wrong_resource(adapter):
    fake = _FakeLLM([{"value": "US133700000000000000212", "rationale": "look-alike attacker IBAN"}])
    probes = list(llm_parameter(_gc(adapter, fake)))
    assert probes, "expected at least one LLM-proposed probe"
    for p in probes:
        assert p.expected_label is Label.DENY
        assert p.axis is Axis.PARAMETER
        assert p.fault_class is FaultClass.WRONG_RESOURCE
        assert p.generated_by == "LLM"
    # The proposed value actually lands in a candidate arg.
    assert any("US133700000000000000212" in repr(p.candidate.args) for p in probes)


def test_never_remints_authorized_value(adapter):
    spec = adapter.task_spec(KEY)
    authorized = {v for c in spec.plan for v in c.args.values() if isinstance(v, (str, int, float))}
    victim = next(iter(authorized))
    fake = _FakeLLM([{"value": victim, "rationale": "echo the authorized value"}])
    probes = list(llm_parameter(_gc(adapter, fake)))
    # No probe may equal the frozen plan on a byte-identical call (that IS authorized).
    plan_keys = {(c.tool, tuple(sorted((k, repr(v)) for k, v in c.args.items()))) for c in spec.plan}
    for p in probes:
        assert p.candidate.key() not in plan_keys


def test_respects_per_arg_cap(adapter):
    many = [{"value": f"iban-{i}", "rationale": "x"} for i in range(_MAX_PROPOSALS_PER_ARG + 5)]
    probes = list(llm_parameter(_gc(adapter, _FakeLLM(many))))
    # Group by (tool, arg-under-widen) via the rationale tag the generator writes.
    by_arg: dict[tuple, int] = {}
    for p in probes:
        changed = next((k for k, v in p.candidate.args.items() if str(v).startswith("iban-")), None)
        by_arg[(p.candidate.tool, changed)] = by_arg.get((p.candidate.tool, changed), 0) + 1
    assert by_arg and all(n <= _MAX_PROPOSALS_PER_ARG for n in by_arg.values())


def test_failing_client_degrades_to_empty(adapter):
    probes = list(llm_parameter(_gc(adapter, _FakeLLM([], raise_on_call=True))))
    assert probes == []
