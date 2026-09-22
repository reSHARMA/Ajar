"""The runner: benchmark + task + defense + generators -> Score.

    adapter -> TaskSpec -> generators (propose) -> oracle (dispose) -> Probe suite
                                                                          |
                                              defense.prepare -> decide  v
                                                                       Score

Two entry points, both ending at the same `score()`:

  `run_task`       generate probes now, score now. Needs agentdojo installed and a
                   ~9-minute pass over all 97 tasks, because every label comes from
                   executing the candidate against a materialized environment.
  `run_from_files` load probes exported earlier, score them. Needs neither agentdojo
                   nor the runs corpus.

`run_task` stays the primary path: the exported files are its output, so evaluating
only from files would make a generator change unobservable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ajar import io, paths
from ajar.adapters.base import BenchmarkAdapter
from ajar.core.ir import TaskSpec
from ajar.core.types import Probe, ProbeResult, TaskKey, Verdict
from ajar.defenses.base import Defense, PolicyUnavailable
from ajar.generate.base import GenContext, build_suite
from ajar.llm.base import LLMClient
from ajar.score.scorer import Score, classify, score


@dataclass
class TaskRun:
    task: TaskKey
    probes: list[Probe]
    scores: dict[str, Score]
    results: dict[str, list[ProbeResult]]
    # Carried so a caller can honour `spec.is_gradeable` without holding an adapter
    # (the file-based path has no adapter). A vacuous success predicate cannot punish
    # any defense, so averaging those tasks in inflates every score equally.
    spec: TaskSpec | None = None


def generate_suite(adapter: BenchmarkAdapter, key: TaskKey, llm: LLMClient | None = None) -> tuple[GenContext, list[Probe]]:
    spec = adapter.task_spec(key)
    gc = GenContext(
        spec=spec,
        adapter=adapter,
        oracle=adapter.oracle(key),
        harm=adapter.harm_model(key),
        llm=llm,
    )
    return gc, build_suite(gc)


def run_defense(defense: Defense, spec, probes: list[Probe], llm: LLMClient | None = None) -> list[ProbeResult]:
    policy = defense.prepare(spec, llm=llm)
    results: list[ProbeResult] = []
    for p in probes:
        if not p.scored:
            continue
        try:
            verdict = defense.decide(policy, p.context, p.candidate)
        except Exception:  # a defense that crashes is ERROR, not silent ALLOW
            verdict = Verdict.ERROR
        outcome, charge = classify(p, verdict if verdict is not Verdict.ERROR else Verdict.DENY)
        results.append(ProbeResult(probe=p, verdict=verdict, outcome=outcome, charge=charge))
    return results


def score_probes(spec: TaskSpec, probes: list[Probe], defenses: list[Defense], llm: LLMClient | None = None) -> TaskRun:
    """Score an already-built probe suite. Shared by both entry points, so a
    file-loaded suite cannot diverge from a freshly generated one."""
    scores, results = {}, {}
    for d in defenses:
        try:
            res = run_defense(d, spec, probes, llm=llm)
        except PolicyUnavailable:
            continue  # defense has no policy for this task; leave it uncovered
        results[d.name] = res
        scores[d.name] = score(d.name, probes, res)
    return TaskRun(task=spec.task, probes=probes, scores=scores, results=results, spec=spec)


def run_task(adapter: BenchmarkAdapter, key: TaskKey, defenses: list[Defense], llm: LLMClient | None = None) -> TaskRun:
    gc, probes = generate_suite(adapter, key, llm=llm)
    return score_probes(gc.spec, probes, defenses, llm=llm)


def run_from_files(
    defenses: list[Defense],
    benchmark: str = "agentdojo",
    data_dir: Path | None = None,
    llm: LLMClient | None = None,
    suite: str | None = None,
) -> list[TaskRun]:
    """Score exported probes without importing the benchmark that produced them.

    Reads `<data_dir>/<benchmark>/{tasks,probes}.jsonl`. The per-benchmark directory
    is not cosmetic: task ids are unique only within a suite, so two benchmarks
    exported to one path would overwrite rather than merge.

    Returns one TaskRun per task that has probes, in `tasks.jsonl` order. A task
    present in tasks.jsonl but absent from probes.jsonl is skipped rather than
    scored as empty -- an empty probe list scores a vacuous S=1.0, which would read
    as a perfect defense.
    """
    d = paths.benchmark_data_dir(benchmark, data_dir)
    specs = io.load_specs(d / "tasks.jsonl")
    by_task = io.load_probes(d / "probes.jsonl")
    runs: list[TaskRun] = []
    for key, spec in specs.items():
        if suite is not None and key.suite != suite:
            continue
        probes = by_task.get(key)
        if not probes:
            continue
        runs.append(score_probes(spec, probes, defenses, llm=llm))
    return runs
