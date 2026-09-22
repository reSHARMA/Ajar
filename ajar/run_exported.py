"""Score every defense against an exported dataset -- no benchmark, no generation.

    python -m ajar.run_exported                        # every suite
    python -m ajar.run_exported banking                 # one suite
    python -m ajar.run_exported --benchmark agentdojo    # pick the benchmark

This is the path a consumer of the benchmark takes. It reads
`benchmarks/<benchmark>/data/probes.jsonl` and `tasks.jsonl` and scores from those alone, so it
needs neither the benchmark's own checkout nor (for AgentDojo) the ~36k-file runs
corpus, and it finishes in a fraction of a second instead of the ~9 minutes
generation takes.

`ajar.run_benchmark` remains the path that GENERATES probes. Use it after touching
a generator, an oracle, or the harm model -- then re-export. Scoring only from files
would make a generator change invisible.

Defenses that cannot produce a policy are reported as uncovered, never as a score:
AgentDojo's `tool_filter` is an offline replay that reads its logged replies out of
the runs corpus, so its row is absent when that corpus is. Do not read a missing row
as a perfect defense.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections import defaultdict

import ajar.defenses.baselines  # noqa: F401  (registers the 4 baselines)
from ajar.defenses.base import get_defense
from ajar.report import leaderboard
from ajar.runner import run_from_files
from ajar.score.aggregate import aggregate

# The benchmark-neutral reference systems. Anything a specific benchmark adds (an
# offline replay of one of its own shipped defenses) is discovered below.
DEFENSES = ["allow_all", "deny_all", "tool_allowlist", "arg_policy"]

# Optional per-benchmark defense modules: importable => registered, and the name is
# appended. Absent or unimportable => simply not scored.
_BENCHMARK_DEFENSES: dict[str, list[tuple[str, str]]] = {
    "agentdojo": [
        ("ajar.adapters.agentdojo.tool_filter", "tool_filter"),
    ]
}

# A deployment that has wrappers of its own registers them from a `local_defenses`
# module on the path, as EXTRA_DEFENSES = {benchmark: [(module, name), ...]}. Keeping
# them there rather than here means this file names only what it can import: a wrapper
# needs its defense installed, and a library install has no reason to have one.
try:  # pragma: no cover - depends on what is installed alongside
    import local_defenses as _local

    for _bench, _entries in getattr(_local, "EXTRA_DEFENSES", {}).items():
        _BENCHMARK_DEFENSES.setdefault(_bench, []).extend(_entries)
except ImportError:
    pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", nargs="?", help="one suite name (default: all)")
    ap.add_argument("--benchmark", default="agentdojo")
    args = ap.parse_args(argv)

    names = list(DEFENSES)
    for module, name in _BENCHMARK_DEFENSES.get(args.benchmark, []):
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001  (its own deps may be absent -- fine)
            continue
        names.append(name)
    defenses = [get_defense(n) for n in names]

    try:
        runs = run_from_files(defenses, benchmark=args.benchmark, suite=args.suite)
    except FileNotFoundError as e:
        print(
            f"{e}\n\nGenerate the dataset first:\n"
            f"  python scripts/export_probes.py --benchmark {args.benchmark}",
            file=sys.stderr,
        )
        return 1
    if not runs:
        where = f"suite {args.suite!r}" if args.suite else args.benchmark
        print(f"no probes found for {where}", file=sys.stderr)
        return 1

    per_defense: dict[str, list] = defaultdict(list)
    by_suite: dict[str, int] = defaultdict(int)
    n_total = 0
    skipped: list[str] = []
    # Attack admission is an over-privilege quantity, so it is pooled over every
    # task, while the under-privilege quantities stay on the gradeable subset below.
    per_defense_all: dict[str, list] = defaultdict(list)
    for tr in runs:
        for name, sc in tr.scores.items():
            per_defense_all[name].append(sc)

    for tr in runs:
        # Matches run_benchmark: a vacuous success predicate cannot punish any
        # defense, so averaging those tasks in inflates every score equally
        # (docs/upstream-agentdojo-defects.md §2).
        if tr.spec is not None and not tr.spec.is_gradeable:
            skipped.append(f"{tr.task.suite}/{tr.task.task_id}")
            continue
        by_suite[tr.task.suite] += 1
        n_total += 1
        for name, sc in tr.scores.items():
            per_defense[name].append(sc)

    scope = args.suite or "all suites"
    print(f"=== lp-bench on {args.benchmark} from data/ ({scope}: {n_total} gradeable tasks) ===")
    print("  " + "  ".join(f"{s}={n}" for s, n in sorted(by_suite.items())))
    if skipped:
        print(f"  {len(skipped)} non-gradeable excluded: {', '.join(skipped)}")
    uncovered = [d.name for d in defenses if not per_defense.get(d.name)]
    if uncovered:
        print(f"  uncovered (no policy available): {', '.join(uncovered)}")
    print()

    aggs = [
        aggregate(d.name, per_defense[d.name], n_total, per_defense_all.get(d.name))
        for d in defenses
        if per_defense.get(d.name)
    ]
    print(leaderboard(aggs))
    # Cross-suite means dilute the headline: a tool-name allowlist blocks most
    # workspace/slack attacks only because those attacks use tools the benign plans
    # never need. Per-suite is the honest view (docs §5).
    if args.suite is None and len(by_suite) > 1:
        print("\nNote: aggregated across suites, which dilutes the finding. Pass a suite name")
        print("for the per-suite view -- that is the one to report (docs §5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
