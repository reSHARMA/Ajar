"""The exported dataset must score exactly like a freshly generated one.

If it does not, the published benchmark and the code that produced it disagree, and
every number in the paper is measured against a file nobody can reproduce. The
failure mode this pins down is not a crash -- it is a missing field falling back to
its dataclass default (`required=True`, `pruned=False`), which changes the score
without raising anything.

Scoped to banking because generating all 97 tasks takes ~9 minutes, which does not
belong in a unit test run. banking is the right choice for the sample: it is the suite
carrying the headline (7 of 9 attacks admitted by a tool-name filter), so a
serialization bug that erased the PARAMETER axis would show here first. The all-suite
equivalence is checked at export time by `scripts/export_probes.py` + the sweep
recorded in the commit message, not here.
"""
from __future__ import annotations

import pytest

from ajar import io
from ajar.core.types import Probe

pytest.importorskip("agentdojo")

DEFENSES = ["allow_all", "deny_all", "tool_allowlist", "arg_policy", "tool_filter"]


@pytest.fixture(scope="module")
def defenses():
    import ajar.adapters.agentdojo.tool_filter  # noqa: F401  (registers)
    import ajar.defenses.baselines  # noqa: F401  (registers the 4 baselines)
    from ajar.defenses.base import get_defense

    return [get_defense(n) for n in DEFENSES]


@pytest.fixture(scope="module")
def banking_suites():
    """(spec, live probes) for every banking task. Module-scoped: generation is the
    expensive half of this benchmark, so it runs once for all tests here."""
    import ajar.generate  # noqa: F401  (registers every probe family)
    from ajar.adapters.agentdojo.adapter import AgentDojoAdapter
    from ajar.runner import generate_suite

    adapter = AgentDojoAdapter()
    out = []
    for key in adapter.list_tasks():
        if key.suite != "banking":
            continue
        gc, probes = generate_suite(adapter, key)
        out.append((gc.spec, probes))
    assert out, "no banking tasks generated"
    return out


def _norm(score):
    """Every published field, rounded. `adequacy` is included on purpose: it counts
    pruned/unknown probes, so it catches a `pruned` flag lost in transit that the
    headline numbers would hide."""
    return (
        round(score.sufficiency, 9),
        score.gate_passed,
        round(score.opl, 9),
        {k: round(v, 9) for k, v in sorted(score.opl_by_axis.items())},
        score.attacks_admitted,
        score.attacks_total,
        round(score.over_restriction, 9),
        score.adequacy,
    )


def test_probe_fields_are_all_serialized():
    """The import-time guard in ajar.io, asserted explicitly so the reason is
    visible: a Probe field absent from _PROBE_FIELDS round-trips to its default."""
    declared = {name for name, _, _ in io._PROBE_FIELDS}
    assert declared == set(Probe.__dataclass_fields__)


def test_derived_fields_are_not_persisted(banking_suites):
    """`scored` and `penalty_weight` are computed from pruned/label/fault/tier/required.
    Persisting them creates a second copy that can contradict its own inputs."""
    _spec, probes = banking_suites[0]
    d = io.probe_to_dict(probes[0])
    assert "scored" not in d
    assert "penalty_weight" not in d


def test_probe_roundtrip_preserves_every_field(banking_suites):
    for _spec, probes in banking_suites:
        for p in probes:
            back = io.probe_from_dict(io.probe_to_dict(p))
            for field in Probe.__dataclass_fields__:
                assert getattr(back, field) == getattr(p, field), f"{p.probe_id}.{field}"
            # Derived properties must agree too -- they are what the scorer reads.
            assert back.scored == p.scored
            assert back.penalty_weight == p.penalty_weight


def test_context_prefix_survives_as_a_tuple(banking_suites):
    """build_suite dedupes on `(context.prefix, candidate.key())`. A prefix that
    comes back as a list is unhashable, so dedup raises TypeError instead of
    deduping -- and JSON has no tuples, so this is the default failure."""
    _spec, probes = banking_suites[0]
    for p in probes[:50]:
        back = io.probe_from_dict(io.probe_to_dict(p))
        assert isinstance(back.context.prefix, tuple)
        hash((back.context.prefix, back.candidate.key()))


def test_spec_roundtrip_preserves_what_defenses_read(banking_suites):
    """`prepare()` reads required_tools, plan, read_closure and tool names. Losing
    read_closure would silently make arg_policy stricter -- i.e. look better."""
    for spec, _probes in banking_suites:
        back = io.spec_from_dict(io.spec_to_dict(spec))
        assert back.task == spec.task
        assert back.required_tools == spec.required_tools
        assert back.read_closure == spec.read_closure
        assert back.permissible_tools == spec.permissible_tools
        assert list(back.plan) == list(spec.plan)
        assert back.granted_tools == spec.granted_tools
        assert back.is_gradeable == spec.is_gradeable
        assert [s.candidate for s in back.attack_sinks] == [s.candidate for s in spec.attack_sinks]
        assert [s.harm_tier for s in back.attack_sinks] == [s.harm_tier for s in spec.attack_sinks]


def test_scores_are_identical_live_vs_roundtripped(banking_suites, defenses):
    """The one that matters: same probes through JSON must produce the same Score."""
    from ajar.runner import score_probes

    for spec, probes in banking_suites:
        spec_rt = io.spec_from_dict(io.spec_to_dict(spec))
        probes_rt = [io.probe_from_dict(io.probe_to_dict(p)) for p in probes]

        live = score_probes(spec, probes, defenses)
        rt = score_probes(spec_rt, probes_rt, defenses)

        assert set(live.scores) == set(rt.scores), spec.task.task_id
        for name in live.scores:
            assert _norm(live.scores[name]) == _norm(rt.scores[name]), (
                f"{spec.task.task_id}/{name}"
            )


def test_export_is_byte_stable(tmp_path, banking_suites):
    """Two exports of the same probes must be byte-identical, or every regeneration
    produces a diff that says nothing about whether the probes changed."""
    rows = [io.probe_to_dict(p) for _s, ps in banking_suites for p in ps]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    io.write_jsonl(a, rows)
    io.write_jsonl(b, rows)
    assert a.read_bytes() == b.read_bytes()


def test_load_probes_groups_by_task(tmp_path, banking_suites):
    rows = [io.probe_to_dict(p) for _s, ps in banking_suites for p in ps]
    path = tmp_path / "probes.jsonl"
    io.write_jsonl(path, rows)
    loaded = io.load_probes(path)
    assert len(loaded) == len(banking_suites)
    assert sum(len(v) for v in loaded.values()) == len(rows)
    for spec, probes in banking_suites:
        assert len(loaded[spec.task]) == len(probes)


def test_export_path_names_no_benchmark():
    """The export/score path must stay benchmark-neutral: it resolves adapters through
    the registry and derives suites from the TaskKeys an adapter lists. A hardcoded
    `AgentDojoAdapter` or suite tuple here means adding a second benchmark requires
    editing shared code, which is exactly what the adapter seam exists to avoid.

    Docstrings and the `--benchmark` default are allowed to name agentdojo; code that
    imports it is not.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("scripts/export_probes.py", "ajar/io.py", "ajar/runner.py"):
        tree = ast.parse((root / rel).read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
        offenders = [m for m in imported if "agentdojo" in m]
        assert not offenders, f"{rel} imports benchmark-specific modules: {offenders}"


def test_runs_dir_is_resolved_in_exactly_one_place():
    """paths.py owns the runs-corpus location. A second inlined copy is a
    silent-divergence hazard: bootstrap would verify one path and the adapter glob
    another, and read_closure would degrade to empty without any error."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", '"runs"', "ajar"],
        cwd=root, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    non_paths = [h for h in hits if not h.startswith("ajar/paths.py")]
    assert not non_paths, f"runs path computed outside paths.py: {non_paths}"
