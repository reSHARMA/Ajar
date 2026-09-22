"""First real result: score five defenses across the banking suite, offline.

    PYTHONPATH=/homes/gws/reshabh/lp-bench \
      bench/agentdojo/.venv/bin/python -m ajar.run_benchmark

Scores allow_all / deny_all / tool_allowlist / arg_policy / tool_filter (a real
AgentDojo defense, replayed offline — no API key) on every gradeable banking user
task, then prints the benchmark-level leaderboard. No LLM, fully deterministic.
"""
from __future__ import annotations

from collections import defaultdict

from ajar.adapters.agentdojo.adapter import AgentDojoAdapter
import ajar.adapters.agentdojo.tool_filter  # noqa: F401  (registers tool_filter)
import ajar.defenses.baselines  # noqa: F401  (registers the 4 baselines)
import ajar.generate  # noqa: F401  (registers every probe family)
from ajar.defenses.base import get_defense
from ajar.report import leaderboard
from ajar.runner import run_task
from ajar.score.aggregate import aggregate

DEFENSES = ["allow_all", "deny_all", "tool_allowlist", "arg_policy", "tool_filter"]
SUITE = "banking"


def main() -> None:
    adapter = AgentDojoAdapter()
    defenses = [get_defense(n) for n in DEFENSES]
    keys = [k for k in adapter.list_tasks() if k.suite == SUITE]

    per_defense: dict[str, list] = defaultdict(list)
    n_total = 0
    print(f"=== lp-bench on agentdojo/{SUITE} ({len(keys)} user tasks) ===\n")
    for key in sorted(keys, key=lambda k: int(k.task_id.rsplit("_", 1)[-1])):
        try:
            spec = adapter.task_spec(key)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {key.task_id}: spec build failed ({type(e).__name__}: {e})")
            continue
        if not spec.is_gradeable:
            print(f"  skip {key.task_id}: non-gradeable (vacuous success predicate)")
            continue
        try:
            tr = run_task(adapter, key, defenses)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {key.task_id}: run failed ({type(e).__name__}: {e})")
            continue
        n_total += 1
        covered = ",".join(sorted(tr.scores))
        print(f"  {key.task_id}: {len(tr.probes)} probes | defenses covered: {covered}")
        for dname, sc in tr.scores.items():
            per_defense[dname].append(sc)

    # This script skips non-gradeable tasks before running them, so it has no scores
    # for those tasks to pool. Its attack admission is therefore on the gradeable
    # subset, not on the paper's all-tasks population; run_exported.py, which loads
    # every run from data/, reports it the paper's way.
    aggs = [aggregate(d.name, per_defense.get(d.name, []), n_total) for d in defenses]
    print(f"\n=== benchmark-level (macro over {n_total} gradeable banking tasks) ===\n")
    print(leaderboard(aggs))


if __name__ == "__main__":
    main()
