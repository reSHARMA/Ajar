"""Read the exported probe dataset the way the live-run harness needs it.

Probe generation is expensive (~9 min for 97 tasks: every label comes from executing
its candidate against a materialized environment), so consumers load instead of
generating. `scripts/export_probes.py` writes `benchmarks/<benchmark>/data/`, and this module is
the reader for the harness, which works one task at a time.

There used to be a SECOND on-disk layout here -- one JSON per task under `probes/`,
with its own copy of the probe encoding -- written by `build_probes.py` while
`export_probes.py` wrote the JSONL dataset. Two writers, two formats, two ideas of
what a complete export is, and nothing saying which was authoritative. The dataset in
data/ is the answer; this reads it. Nothing in the repo writes `probes/` anymore.

Encoding lives in `ajar.io` and is not duplicated here: a second field list is how
the two copies silently disagree about what a Probe is, and only io's has the
import-time drift guard.
"""
from __future__ import annotations

from pathlib import Path

from ajar import io, paths
from ajar.core.types import Probe, RenderedInjection, TaskKey

__all__ = [
    "dataset_dir",
    "probes_path",
    "injections_path",
    "load_suite",
    "has_suite",
    "load_injections",
    "has_injections",
]

_DEFAULT_BENCHMARK = "agentdojo"


def dataset_dir(benchmark: str = _DEFAULT_BENCHMARK, root: Path | None = None) -> Path:
    return paths.benchmark_data_dir(benchmark, root)


def probes_path(benchmark: str = _DEFAULT_BENCHMARK, root: Path | None = None) -> Path:
    return dataset_dir(benchmark, root) / "probes.jsonl"


def injections_path(benchmark: str = _DEFAULT_BENCHMARK, root: Path | None = None) -> Path:
    return dataset_dir(benchmark, root) / "injections.jsonl"


def _missing(path: Path, what: str, how: str) -> FileNotFoundError:
    return FileNotFoundError(f"no exported {what} at {path}\nBuild them first:  {how}")


def load_suite(key: TaskKey, root: Path | None = None) -> list[Probe]:
    """One task's probes.

    Raises when the DATASET is absent (nothing has been exported -- a setup error) but
    also when the task has no rows, because the caller asked for a specific task by
    name: silently returning [] would read as "this task has no probes" and score as a
    task with nothing to test.
    """
    path = probes_path(key.benchmark, root)
    if not path.is_file():
        raise _missing(path, f"probes for {key.suite}/{key.task_id}", "python scripts/export_probes.py")
    probes = io.load_task_probes(path, key)
    if not probes:
        raise _missing(
            path,
            f"rows for {key.suite}/{key.task_id} in the export",
            f"python scripts/export_probes.py {key.suite} --task {key.task_id}",
        )
    return probes


def has_suite(key: TaskKey, root: Path | None = None) -> bool:
    """Whether the export covers this task. For callers that iterate over every task
    and skip the ones not exported (e.g. non-gradeable), where absence is expected."""
    path = probes_path(key.benchmark, root)
    return path.is_file() and key in io.load_task_keys(path)


def load_injections(
    suite: str, benchmark: str = _DEFAULT_BENCHMARK, root: Path | None = None
) -> list[RenderedInjection]:
    path = injections_path(benchmark, root)
    if not path.is_file():
        raise _missing(
            path,
            f"injections for suite {suite}",
            f"python -m ajar.build_injections --suite {suite}",
        )
    return io.load_injections(path, suite)


def has_injections(suite: str, benchmark: str = _DEFAULT_BENCHMARK, root: Path | None = None) -> bool:
    return bool(io.load_injections(injections_path(benchmark, root), suite))
