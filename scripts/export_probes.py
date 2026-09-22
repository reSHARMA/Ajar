#!/usr/bin/env python
"""Generate a benchmark's probe suite once and write it to data/, so consumers never
regenerate it.

    python scripts/export_probes.py                          # every suite
    python scripts/export_probes.py banking                   # one suite
    python scripts/export_probes.py banking --task user_task_0  # one task
    python scripts/export_probes.py --benchmark agentdojo      # pick the benchmark

Writes, per benchmark:

    benchmarks/<benchmark>/data/tasks.jsonl     one TaskSpec per task -- the ground truth a
                                    defense sees
    benchmarks/<benchmark>/data/probes.jsonl   one Probe per line, the labelled decision points
    benchmarks/<benchmark>/data/manifest.json  counts + what failed, so a truncated export is
                                    visible rather than looking like a small suite

A PARTIAL export (a suite or a single task) MERGES: the rebuilt tasks' rows replace
their old ones and every other task is carried through untouched. Otherwise exporting
one task would silently truncate the dataset to that task. Rows are written sorted by
task, so partial and full exports converge on the same bytes -- a diff means the probes
changed, not the order they were built in.

Generation is the expensive half of this benchmark (~9 min for AgentDojo's 97 tasks):
every probe's label comes from executing its candidate against a materialized
environment. That work is defense-independent, so paying it per defense is waste.

The generate path (`runner.run_task`) is NOT retired by this. These files are its
output, so a change to a generator has to be re-exported to be visible -- evaluating
only from files would make generator bugs unobservable.

Nothing here knows what AgentDojo is. The adapter is resolved through the registry
(`ajar.adapters.base`), suites are derived from the TaskKeys the adapter lists, and
anything benchmark-specific in the manifest comes from an optional
`adapter.export_metadata()`. Adding a benchmark means writing an adapter, not editing
this script.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections import defaultdict
from pathlib import Path

from ajar import io, paths
from ajar.adapters.base import get_benchmark
import ajar.generate  # noqa: F401  (registers every probe family)
from ajar.runner import generate_suite

DEFAULT_BENCHMARK = "agentdojo"
_ADAPTERS_DIR = Path(__file__).resolve().parents[1] / "ajar" / "adapters"


def _available_adapters() -> list[str]:
    """From the filesystem, not the registry: adapters register at import, so an
    un-imported one is invisible to `list_benchmarks()` -- and reporting
    "registered: []" when the user typos a name is useless."""
    return sorted(
        p.name
        for p in _ADAPTERS_DIR.iterdir()
        if p.is_dir() and (p / "adapter.py").is_file() and not p.name.startswith("_")
    )


def _load_adapter(name: str):
    """Adapters self-register at import (`register_benchmark(...)` at module scope),
    so the module has to be imported before the registry knows the name."""
    try:
        importlib.import_module(f"ajar.adapters.{name}.adapter")
    except ModuleNotFoundError as e:
        # Only swallow a missing adapter module; a missing dependency INSIDE the
        # adapter (e.g. agentdojo itself) must not be reported as "unknown benchmark".
        if getattr(e, "name", "") not in (f"ajar.adapters.{name}", f"ajar.adapters.{name}.adapter"):
            raise
    return get_benchmark(name)


def _task_num(task_id: str) -> int:
    """Trailing number of a task id, for numeric ordering. 0 when there is none --
    the full id stays in the sort key, so ties still order deterministically."""
    tail = task_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _row_key(task: dict) -> tuple:
    """Sort/identity key for a task row. Suite first, then task number NUMERICALLY --
    a lexical sort puts user_task_10 before user_task_2, which makes a re-export look
    like it reordered the dataset."""
    tid = task["task_id"]
    return (task["benchmark"], task["suite"], _task_num(tid), tid)


def _carry_over(path: Path, rebuilt: set[tuple]) -> list[dict]:
    """Rows for tasks this run did NOT build, read back from the existing export."""
    if not path.is_file():
        return []
    return [d for d in io.read_jsonl(path) if _row_key(d["task"]) not in rebuilt]


def _manifest(adapter, spec_rows: list[dict], probe_rows: list[dict], failed: list[dict]) -> dict:
    """Recomputed from the rows on their way to disk, not accumulated during the build.

    A partial export carries other tasks' rows through, so counters that only saw this
    run would under-report the file they are describing -- and a manifest that
    disagrees with its own dataset is worse than no manifest.
    """
    per_suite: dict[str, dict] = {}
    versions: dict[str, set] = defaultdict(set)
    for r in spec_rows:
        t = r["task"]
        per_suite.setdefault(t["suite"], {"tasks": 0, "probes": 0, "scored": 0})["tasks"] += 1
        versions[t["suite"]].add(".".join(map(str, t["version"])))
    for r in probe_rows:
        st = per_suite.setdefault(r["task"]["suite"], {"tasks": 0, "probes": 0, "scored": 0})
        st["probes"] += 1
        # Mirrors Probe.scored (types.py): a property, so it is not in the row.
        if not r["pruned"] and r["expected_label"] != "UNKNOWN":
            st["scored"] += 1

    manifest = {
        "schema_version": io.SCHEMA_VERSION,
        "benchmark": adapter.name,
        "suites": sorted(per_suite),
        # Per suite, not one field: one release label does not imply one version
        # tuple (AgentDojo's "v1.2.2" reports (1,2,2) for two suites and (1,2,0) for
        # the other two), and flattening that has already cost us 41 tasks once.
        "suite_versions": {s: sorted(v) for s, v in sorted(versions.items())},
        "n_tasks": len(spec_rows),
        "n_probes": len(probe_rows),
        "per_suite": {s: per_suite[s] for s in sorted(per_suite)},
        # Only this run's failures: a task that failed here but exists in the dataset
        # from an earlier full export is still present, and listing it as failed would
        # misreport the file.
        "failed_tasks": failed,
    }
    # Anything only the adapter can know (upstream release label, excluded tasks and
    # why). Optional: a benchmark that does not implement it exports fine.
    extra = getattr(adapter, "export_metadata", None)
    if callable(extra):
        manifest["benchmark_metadata"] = extra()
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="*", help="suite names to export (default: all)")
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help=f"default: {DEFAULT_BENCHMARK}")
    ap.add_argument("--task", default=None,
                    help="export a single task id (requires exactly one suite); merges into the dataset")
    ap.add_argument("--llm", action="store_true",
                    help="use Bedrock for LLM-generated probes + fluent prompts")
    args = ap.parse_args(argv)

    if args.task and len(args.suites) != 1:
        print("--task needs exactly one suite, e.g. banking --task user_task_0", file=sys.stderr)
        return 2

    try:
        adapter = _load_adapter(args.benchmark)
    except KeyError:
        print(
            f"unknown benchmark {args.benchmark!r}; available: {_available_adapters()}",
            file=sys.stderr,
        )
        return 2

    keys = list(adapter.list_tasks())
    available = sorted({k.suite for k in keys})
    only = args.suites or available
    bad = [s for s in only if s not in available]
    if bad:
        print(f"unknown suite(s) {bad} for {adapter.name}; available: {available}", file=sys.stderr)
        return 2
    keys = [k for k in keys if k.suite in only]

    # Refuse to export a silently-wrong AgentDojo suite. read_closure is MEASURED from a
    # run-log corpus (bench/agentdojo/runs/**/none.json) that is GITIGNORED and therefore
    # worktree-LOCAL. If the corpus is absent, read_closure comes back EMPTY for every
    # task, which re-labels every legitimate discovery read as an off-path DENY -- an
    # export that looks fine but disagrees with the committed suite. Fail loud instead.
    if adapter.name == "agentdojo":
        import glob as _glob
        if not _glob.glob(str(paths.runs_dir() / "*" / "*" / "*" / "none" / "none.json")):
            print(
                f"ERROR: AgentDojo run-log corpus not found under {paths.runs_dir()} "
                "(0 none.json). read_closure is measured from it and would come back EMPTY, "
                "turning every legitimate discovery read into an off-path DENY -- producing a "
                "suite that silently disagrees with the committed one. This corpus is "
                "gitignored (worktree-local). Set AJAR_AGENTDOJO_ROOT to a worktree that "
                "has bench/agentdojo/runs, or symlink it in, then re-run.",
                file=sys.stderr,
            )
            return 2

    # Only a whole-benchmark export may replace the dataset; anything narrower merges.
    full = not args.suites and not args.task

    if args.task:
        keys = [k for k in keys if k.task_id == args.task]
        if not keys:
            print(f"unknown task {args.task!r} in suite {only[0]}", file=sys.stderr)
            return 2

    llm = None
    if args.llm:
        from ajar.llm.bedrock import BedrockLLM

        llm = BedrockLLM()

    out = paths.benchmark_data_dir(adapter.name)
    spec_rows: list[dict] = []
    probe_rows: list[dict] = []
    # A task that fails to build is recorded, never dropped silently: a short
    # probes.jsonl and a fully-exported one look identical from the outside.
    failed: list[dict] = []
    t0 = time.monotonic()

    for i, key in enumerate(keys, 1):
        t = time.monotonic()
        try:
            gc, probes = generate_suite(adapter, key, llm=llm)
        except Exception as e:  # noqa: BLE001
            failed.append({"task": str(key), "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(keys)}] {key.suite}/{key.task_id}  FAILED {type(e).__name__}: {e}")
            continue
        spec_rows.append(io.spec_to_dict(gc.spec))
        probe_rows.extend(io.probe_to_dict(p) for p in probes)
        n_scored = sum(1 for p in probes if p.scored)
        print(
            f"[{i}/{len(keys)}] {key.suite}/{key.task_id:14s} "
            f"{len(probes):5d} probes ({n_scored} scored)  {time.monotonic() - t:5.1f}s"
            + ("" if gc.spec.is_gradeable else "  [non-gradeable]")
        )

    # A partial export merges rather than replaces. Keyed on (benchmark, suite, task),
    # not task_id alone: ids repeat across suites -- `user_task_0` exists in all four.
    rebuilt = {(k.benchmark, k.suite, _task_num(k.task_id), k.task_id) for k in keys}
    if not full:
        spec_rows = _carry_over(out / "tasks.jsonl", rebuilt) + spec_rows
        probe_rows = _carry_over(out / "probes.jsonl", rebuilt) + probe_rows

    # Sort by task so a partial and a full export of the same probes agree byte for
    # byte; within a task, generation order is meaningful (k0 before k1) so it stands.
    spec_rows.sort(key=lambda r: _row_key(r["task"]))
    probe_rows.sort(key=lambda r: _row_key(r["task"]))

    manifest = _manifest(adapter, spec_rows, probe_rows, failed)

    n_specs = io.write_jsonl(out / "tasks.jsonl", spec_rows)
    n_probes = io.write_jsonl(out / "probes.jsonl", probe_rows)
    io.write_json(out / "manifest.json", manifest)

    print(f"\nwrote {out}/tasks.jsonl   {n_specs} tasks")
    print(f"wrote {out}/probes.jsonl  {n_probes} probes")
    print(f"wrote {out}/manifest.json")
    if failed:
        print(f"\n{len(failed)} task(s) FAILED to build -- see manifest.failed_tasks")
    print(f"total {time.monotonic() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
