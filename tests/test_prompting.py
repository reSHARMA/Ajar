"""Natural-language prompt rendering attached to every probe at generation time.

Mechanical rendering is deterministic and offline; the LLM path is exercised with a
fake client (no Bedrock) to confirm it overrides the floor and degrades on failure.
"""
from __future__ import annotations

import pytest

from ajar.core.types import TaskKey
from ajar.generate.prompting import _PREFIX, _normalize, render_prompts
from ajar.runner import generate_suite

KEY = TaskKey(benchmark="agentdojo", suite="banking", version=(1, 2, 2), task_id="user_task_0")


@pytest.fixture(scope="module")
def adapter():
    import ajar.adapters.agentdojo.adapter  # noqa: F401  (registers)
    import ajar.generate  # noqa: F401  (registers families)
    from ajar.adapters.base import get_benchmark

    return get_benchmark("agentdojo")


@pytest.fixture(scope="module")
def suite(adapter):
    # Mechanical path (llm=None): deterministic, no network.
    _, probes = generate_suite(adapter, KEY, llm=None)
    return probes


def test_every_probe_has_a_prompt(suite):
    assert suite
    assert all(p.prompt.startswith(_PREFIX) for p in suite)


def test_mechanical_prompt_names_the_tool_and_args(suite):
    for p in suite:
        assert p.candidate.tool in p.prompt
        for v in p.candidate.args.values():
            if isinstance(v, str) and v:
                assert repr(v) in p.prompt or v in p.prompt


def test_same_candidate_renders_identically(suite):
    by_cand: dict[tuple, set[str]] = {}
    for p in suite:
        by_cand.setdefault(p.candidate.key(), set()).add(p.prompt)
    # A candidate offered at k=0/1/2 must share one prompt (state-independent).
    assert all(len(v) == 1 for v in by_cand.values())


class _FakeGC:
    """Minimal GenContext stand-in for render_prompts: just .spec.tools and .llm."""

    def __init__(self, spec, llm):
        self.spec = spec
        self.llm = llm


class _FakeLLM:
    def __init__(self, clauses=None, raise_on_call=False):
        self._clauses = clauses
        self._raise = raise_on_call

    def complete(self, messages, **kwargs):  # noqa: ANN001
        return ""

    def complete_json(self, messages, schema, **kwargs):  # noqa: ANN001
        if self._raise:
            raise RuntimeError("bedrock down")
        return list(self._clauses)


def test_llm_overrides_mechanical(adapter, suite):
    probes = [p for p in suite]  # render_prompts mutates in place
    gc = _FakeGC(adapter.task_spec(KEY), _FakeLLM(clauses=["to do the thing."] * 500))
    render_prompts(probes, gc)
    assert all(p.prompt == "Your first task is to do the thing" for p in probes)


def test_llm_failure_falls_back_to_mechanical(adapter, suite):
    probes = [p for p in suite]
    gc = _FakeGC(adapter.task_spec(KEY), _FakeLLM(raise_on_call=True))
    render_prompts(probes, gc)
    # Fell back: prompts still name the tool (the mechanical form).
    assert all(p.candidate.tool in p.prompt for p in probes)


def test_normalize_strips_leading_to_and_trailing_dot():
    assert _normalize("  To send money.  ") == "send money"
    assert _normalize("read the file") == "read the file"
