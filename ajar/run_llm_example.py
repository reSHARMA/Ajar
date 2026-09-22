"""LLM-proposed probes on a real task, via Bedrock — the seam wired end to end.

    bench/agentdojo/.venv/bin/python -m ajar.run_llm_example
    bench/agentdojo/.venv/bin/python -m ajar.run_llm_example workspace user_task_0

Builds the probe suite for one task twice -- once mechanically (`llm=None`) and once
with a Bedrock-backed client -- and reports what the LLM added: the extra probes, how
many the oracle CONFIRMED (a proposal that matches an attack sink), and their harm
tiers. Everything the LLM proposes is still disposed by AgentDojo's own predicates and
tiered by the harm model; the LLM only widens coverage.

Auth is the instance IAM role (no API key). Default model: Claude Sonnet 4.
"""
from __future__ import annotations

import sys

from ajar.adapters.agentdojo.adapter import AgentDojoAdapter
import ajar.generate  # noqa: F401  (registers every family, incl. 3.2-parameter-llm)
from ajar.core.types import Label, LabelSource, TaskKey
from ajar.llm.bedrock import BedrockLLM
from ajar.runner import generate_suite

_VERSIONS = {"banking": (1, 2, 2), "workspace": (1, 2, 2), "slack": (1, 2, 0), "travel": (1, 2, 0)}


def main() -> None:
    suite = sys.argv[1] if len(sys.argv) > 1 else "banking"
    task_id = sys.argv[2] if len(sys.argv) > 2 else "user_task_0"
    model = sys.argv[3] if len(sys.argv) > 3 else BedrockLLM().model

    adapter = AgentDojoAdapter()
    key = TaskKey(benchmark="agentdojo", suite=suite, version=_VERSIONS[suite], task_id=task_id)

    _, base = generate_suite(adapter, key, llm=None)
    llm = BedrockLLM(model=model)
    _, full = generate_suite(adapter, key, llm=llm)

    print(f"=== {key} ===")
    print(f"model: {llm.model}  (backend={llm.backend})\n")
    print(f"mechanical probes : {len(base)}")
    print(f"with LLM          : {len(full)}")

    llm_probes = [p for p in full if p.generated_by == "LLM"]
    kept = [p for p in llm_probes if not p.pruned]
    confirmed = [p for p in kept if p.label_source is LabelSource.ORACLE_CONFIRMED]
    print(f"LLM-proposed kept : {len(kept)}  ({len(llm_probes) - len(kept)} pruned as H0/dup)")
    print(f"  oracle-CONFIRMED DENY (matched an attack sink): {len(confirmed)}")

    tiers: dict[str, int] = {}
    for p in kept:
        tiers[p.harm_tier.name] = tiers.get(p.harm_tier.name, 0) + 1
    print(f"  harm tiers        : {tiers}\n")

    print("--- sample LLM-proposed probes ---")
    for p in kept[:8]:
        arg = next((k for k in p.candidate.args if k in p.rationale), "?")
        val = p.candidate.args.get(arg)
        src = "CONFIRMED" if p.label_source is LabelSource.ORACLE_CONFIRMED else "asserted"
        print(f"  [{p.harm_tier.name} {src:9s}] {p.candidate.tool}.{arg} = {val!r}")

    if not kept:
        print("  (none — the LLM proposed only values that collide with mechanical probes)")


if __name__ == "__main__":
    main()
