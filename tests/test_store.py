"""The harness reads one task out of the exported dataset with full fidelity (offline).

`store` used to own a second on-disk layout (one JSON per task) with its own copy of
the probe encoding. Now it reads `benchmarks/<benchmark>/data/probes.jsonl` -- the same
file the offline scorer reads -- so these tests write a JSONL export directly rather
than going through a save function that no longer exists.
"""
from __future__ import annotations

import pytest

from ajar import io
from ajar.core.types import Candidate, RenderedInjection, TaskKey
from ajar.runner import generate_suite
from ajar.store import has_injections, has_suite, load_injections, load_suite

KEY = TaskKey(benchmark="agentdojo", suite="banking", version=(1, 2, 2), task_id="user_task_0")
OTHER = TaskKey(benchmark="agentdojo", suite="banking", version=(1, 2, 2), task_id="user_task_1")


@pytest.fixture(scope="module")
def adapter():
    import ajar.adapters.agentdojo.adapter  # noqa: F401
    import ajar.generate  # noqa: F401
    from ajar.adapters.base import get_benchmark

    return get_benchmark("agentdojo")


@pytest.fixture(scope="module")
def probes(adapter):
    _, ps = generate_suite(adapter, KEY, llm=None)
    return ps


def _export(root, probes, key=KEY):
    """Write probes to a temp dataset root, the way export_probes.py would.

    Layout is `<root>/<benchmark>/data/probes.jsonl` -- a benchmark's exported suite
    lives under its own `data/` dir (beside its `defenses/`), which `dataset_dir`
    composes for both the real tree and this temp root."""
    path = root / key.benchmark / "data" / "probes.jsonl"
    io.write_jsonl(path, [io.probe_to_dict(p) for p in probes])
    return path


def test_dict_round_trip_is_identical(probes):
    for p in probes:
        again = io.probe_from_dict(io.probe_to_dict(p))
        assert io.probe_to_dict(again) == io.probe_to_dict(p)


def test_export_load_preserves_everything(probes, tmp_path):
    _export(tmp_path, probes)
    loaded = load_suite(KEY, root=tmp_path)
    assert len(loaded) == len(probes)
    for a, b in zip(probes, loaded):
        assert io.probe_to_dict(a) == io.probe_to_dict(b)
    # spot-check the fields most likely to drift on (de)serialization
    for a, b in zip(probes, loaded):
        assert a.expected_label is b.expected_label       # enum identity
        assert a.harm_tier is b.harm_tier
        assert a.candidate.key() == b.candidate.key()     # args types (int/float/str) survive
        assert a.context.prefix == b.context.prefix        # tuple, not list
        assert a.task.version == b.task.version            # tuple, not list
        assert a.prompt == b.prompt


def test_loads_only_the_requested_task(probes, tmp_path):
    """The dataset holds every task in one file, so the filter is load-bearing: without
    it the harness would run another task's probes against this task's environment."""
    import dataclasses

    foreign = [dataclasses.replace(probes[0], task=OTHER, probe_id="foreign-1")]
    _export(tmp_path, list(probes) + foreign)

    loaded = load_suite(KEY, root=tmp_path)
    assert len(loaded) == len(probes)
    assert all(p.task == KEY for p in loaded)
    assert "foreign-1" not in {p.probe_id for p in loaded}

    assert has_suite(KEY, root=tmp_path)
    assert has_suite(OTHER, root=tmp_path)


def test_missing_dataset_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_probes"):
        load_suite(KEY, root=tmp_path)


def test_task_absent_from_export_raises(probes, tmp_path):
    """Asking for a named task that is not in the file is an error, not an empty list:
    [] would score as a task with nothing to test."""
    _export(tmp_path, probes)
    with pytest.raises(FileNotFoundError, match="user_task_1"):
        load_suite(OTHER, root=tmp_path)
    assert not has_suite(OTHER, root=tmp_path)


def test_injections_round_trip_and_filter_by_suite(tmp_path):
    recs = [
        RenderedInjection(suite="banking", injection_id="i0", goal="g0", prompt="p0",
                          calls=[Candidate(tool="send_money", args={"amount": 1.5})]),
        RenderedInjection(suite="slack", injection_id="i1", goal="g1", prompt="p1", calls=[]),
    ]
    io.write_jsonl(tmp_path / "agentdojo" / "data" / "injections.jsonl",
                   [io.injection_to_dict(r) for r in recs])

    banking = load_injections("banking", root=tmp_path)
    assert [r.injection_id for r in banking] == ["i0"]
    assert banking[0].calls[0].args["amount"] == 1.5   # float survives, not "1.5"
    assert load_injections("slack", root=tmp_path)[0].calls == []

    assert has_injections("banking", root=tmp_path)
    assert not has_injections("travel", root=tmp_path)


def test_missing_injections_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_injections"):
        load_injections("banking", root=tmp_path)
