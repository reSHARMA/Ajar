"""Render a suite's injection tasks to text prompts and store them (once).

    bench/agentdojo/.venv/bin/python -m ajar.build_injections --suite banking
    bench/agentdojo/.venv/bin/python -m ajar.build_injections --suite banking --llm

Injection tasks are suite-scoped and un-paired: any injection can run against any user
task in the same suite. So we render each injection's attack (its `ground_truth`
calls) into one "Your first task is to ..." prompt -- the SAME renderer used for
probes -- and store one set per suite. run_pairs pairs them with every user task.

Fallback: an injection with no liftable call (e.g. a model-output substring attack
whose ground_truth() is empty) keeps its AgentDojo GOAL string as the prompt.

Output is `benchmarks/<benchmark>/data/injections.jsonl`, alongside the probe dataset, one row
per injection with its suite on the row. Rebuilding one suite preserves the others:
the file is read, the target suite's rows are replaced, and everything is written back
sorted -- so exporting suites one at a time and all at once give the same file.
"""
from __future__ import annotations

import argparse

from ajar import io, paths
from ajar.adapters.agentdojo.adapter import AgentDojoAdapter
from ajar.core.types import Candidate, RenderedInjection, TaskKey
from ajar.generate.prompting import as_prompt, render_clauses

_VERSIONS = {"banking": (1, 2, 2), "workspace": (1, 2, 2), "slack": (1, 2, 0), "travel": (1, 2, 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render + store a suite's injection-task prompts.")
    ap.add_argument("--suite", default="banking", choices=list(_VERSIONS))
    ap.add_argument("--llm", action="store_true", help="use Bedrock for fluent prompts")
    args = ap.parse_args()

    llm = None
    if args.llm:
        from ajar.llm.bedrock import BedrockLLM

        llm = BedrockLLM()

    adapter = AgentDojoAdapter()
    from agentdojo.task_suite.load_suites import get_suites

    suite = get_suites("v1.2.2")[args.suite]
    # Tool specs are suite-level; borrow them from any user task's spec.
    any_task = next(iter(suite.user_tasks))
    key0 = TaskKey(benchmark="agentdojo", suite=args.suite, version=_VERSIONS[args.suite], task_id=any_task)
    specs = {t.name: t for t in adapter.task_spec(key0).tools}

    base_env = suite.load_and_inject_default_environment({})

    # Collect every injection's ground-truth calls, render all distinct calls in one
    # batch, then assemble each injection's prompt (joined if it has several calls).
    per_injection: dict[str, list[Candidate]] = {}
    order: list[Candidate] = []
    index: dict[tuple, int] = {}
    for inj_id, inj in suite.injection_tasks.items():
        gt = inj.ground_truth(base_env.model_copy(deep=True))
        calls = [Candidate(tool=c.function, args=dict(c.args)) for c in gt]
        per_injection[inj_id] = calls
        for c in calls:
            if c.key() not in index:
                index[c.key()] = len(order)
                order.append(c)

    clauses = render_clauses(order, specs, llm) if order else []

    records: list[RenderedInjection] = []
    for inj_id, inj in suite.injection_tasks.items():
        calls = per_injection[inj_id]
        goal = getattr(inj, "GOAL", "")
        if calls:
            joined = ", and then ".join(clauses[index[c.key()]] for c in calls)
            prompt = as_prompt(joined)
        else:
            prompt = goal  # no liftable call: keep the authored attacker goal
        records.append(
            RenderedInjection(suite=args.suite, injection_id=inj_id, goal=goal, prompt=prompt, calls=calls)
        )

    # Merge, don't overwrite: this script builds ONE suite per invocation, and a
    # plain write would silently drop the other three suites' rows.
    path = paths.benchmark_data_dir(adapter.name) / "injections.jsonl"
    kept = [r for r in io.load_injections(path) if r.suite != args.suite]
    merged = sorted(kept + records, key=lambda r: (r.suite, r.injection_id))
    n = io.write_jsonl(path, [io.injection_to_dict(r) for r in merged])

    print(f"rendered {len(records)} injection prompts for {args.suite}  (llm={'on' if llm else 'off'})")
    print(f"wrote {path}  ({n} rows across {len({r.suite for r in merged})} suite(s))")


if __name__ == "__main__":
    main()
