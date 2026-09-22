"""Drive an external runner over every (user task, prompt) — probes AND injections.

For each user task, iterate its stored probes and then the suite's stored injection
tasks; fetch each item's text prompt; and invoke an external runner with
(suite, user_task, prompt). Probes and injections are BUILT once (ajar.build_probes
/ ajar.build_injections) and only LOADED here.

    # dry run (prints what it would call)
    bench/agentdojo/.venv/bin/python -m ajar.run_pairs --suite banking --task user_task_0

    # real run: point --runner at the script that takes  <suite> <user_task> <prompt>
    bench/agentdojo/.venv/bin/python -m ajar.run_pairs --suite banking --runner ./run_one.sh

The runner contract (assumed to exist): `runner <suite> <user_task> <prompt>`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter
from dataclasses import dataclass

from ajar.adapters.agentdojo.adapter import AgentDojoAdapter
from ajar.core.types import TaskKey
from ajar.store import has_injections, has_suite, load_injections, load_suite

_SUITES = ("banking", "slack", "travel", "workspace")


@dataclass
class Stats:
    tasks: int = 0
    probes: int = 0
    injections: int = 0
    skipped_tasks: int = 0


def run_item(runner: str | None, suite: str, task_id: str, prompt: str, kind: str, ident: str) -> None:
    """Hand one (suite, user_task, prompt) to the external runner (or dry-run print)."""
    if runner is None:
        print(f"[dry-run] {kind:9s} {ident:28s} {suite}/{task_id} | {prompt[:70]}")
        return
    subprocess.run([runner, suite, task_id, prompt], check=False)


def _task_num(task_id: str) -> int:
    try:
        return int(task_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run every stored probe and injection prompt per user task.")
    ap.add_argument("--suite", choices=_SUITES, action="append", help="restrict to suite(s); repeatable")
    ap.add_argument("--task", help="restrict to one user task id (e.g. user_task_0)")
    ap.add_argument("--runner", default=os.environ.get("AJAR_RUNNER"),
                    help="external script: `runner <suite> <user_task> <prompt>` (default: dry run)")
    ap.add_argument("--no-injections", action="store_true", help="probes only, skip injection prompts")
    args = ap.parse_args()

    suites = tuple(args.suite) if args.suite else _SUITES
    adapter = AgentDojoAdapter()
    keys = [k for k in adapter.list_tasks() if k.suite in suites]
    if args.task:
        keys = [k for k in keys if k.task_id == args.task]
    keys.sort(key=lambda k: (k.suite, _task_num(k.task_id)))

    stats = Stats()
    by_kind: Counter = Counter()

    # ---- outer loop: user tasks ------------------------------------------
    for key in keys:
        if not has_suite(key):
            stats.skipped_tasks += 1
            continue
        stats.tasks += 1

        # ---- probes for this user task -----------------------------------
        for probe in load_suite(key):
            run_item(args.runner, key.suite, key.task_id, probe.prompt, "probe", probe.probe_id)
            stats.probes += 1
            by_kind["probe"] += 1

        # ---- injection tasks that can run with this user task ------------
        if not args.no_injections:
            if has_injections(key.suite):
                for inj in load_injections(key.suite):
                    run_item(args.runner, key.suite, key.task_id, inj.prompt, "injection", inj.injection_id)
                    stats.injections += 1
                    by_kind["injection"] += 1
            else:
                print(f"  (no stored injections for {key.suite}; run: "
                      f"python -m ajar.build_injections --suite {key.suite})")

    if stats.tasks == 0:
        print("no stored probe suites found. Build one first:")
        print("  python -m ajar.build_probes --suite banking --task user_task_0")
        print("  python -m ajar.build_injections --suite banking")

    print(
        f"\n=== summary ===\n"
        f"runner          : {args.runner or '(dry run)'}\n"
        f"user tasks      : {stats.tasks}  (skipped, no probes: {stats.skipped_tasks})\n"
        f"probe prompts   : {stats.probes}\n"
        f"injection prompts: {stats.injections}\n"
        f"total invocations: {stats.probes + stats.injections}"
    )


if __name__ == "__main__":
    main()
