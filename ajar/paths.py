"""Where the vendored benchmark lives, and whether its run corpus is present.

One place resolves these paths. `scripts/bootstrap.sh` calls `describe()` and
`require_runs_dir()` to verify a fresh environment; the AgentDojo adapter calls
`runs_dir()` to measure the read closure. Those two used to compute the location
independently, which is a silent-divergence hazard: bootstrap could report a corpus
that the adapter then looks for somewhere else.

Nothing here imports agentdojo, so it stays importable in a bare interpreter.
"""
from __future__ import annotations

import os
from pathlib import Path

# ajar/paths.py -> ajar/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

_ENV_ROOT = "AJAR_AGENTDOJO_ROOT"  # honoured by bootstrap.sh too
_ENV_DATA = "AJAR_DATA_DIR"
_ENV_RUNNERS = "AJAR_RUNNERS_DIR"


def agentdojo_root() -> Path:
    """The vendored AgentDojo checkout. Gitignored (~940M with its own .git), so
    a fresh clone has to create it -- see the `git clone` line in .gitignore."""
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "bench" / "agentdojo"


def runners_dir() -> Path:
    """The shell scripts that launch an agent for a probe.

    Tracked in this repository, so the default is a repo-relative path rather than
    wherever a particular machine happens to keep a checkout. Override with
    AJAR_RUNNERS_DIR to point at a different set.
    """
    override = os.environ.get(_ENV_RUNNERS)
    if override:
        return Path(override).expanduser().resolve()
    try:  # pragma: no cover - depends on what is installed alongside
        import local_defenses as _local

        declared = getattr(_local, "RUNNERS_DIR", None)
        if declared:
            return Path(declared).expanduser().resolve()
    except ImportError:
        pass
    return REPO_ROOT / "runners"


def runner(name: str) -> str:
    """One runner by file name, as a string for subprocess.

    Raises when it is absent rather than letting subprocess fail on a path that says
    nothing about what to do: the runners belong to whatever harness drives the agent,
    which an install of this package alone does not provide.
    """
    path = runners_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"agent runner not found at {path}\n"
            f"Point {_ENV_RUNNERS} at a directory containing {name}."
        )
    return str(path)


def runs_dir() -> Path:
    """AgentDojo's own eval logs (~36k JSONs), shipped inside its repo."""
    return agentdojo_root() / "runs"


def require_runs_dir() -> Path:
    """Same, but fails loudly when absent.

    The corpus is load-bearing rather than nice-to-have: `read_closure` is MEASURED
    from it (9 of 67 successful runs call a legitimate read that ground truth omits,
    because the ground-truth author derives in Python what an LLM must discover).
    Without it the closure silently degrades to the empty set and those legitimate
    discovery reads get scored as over-privilege -- a wrong result that still looks
    like a result, which is why this raises instead of returning None.
    """
    d = runs_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"run corpus not found at {d}\n"
            f"It ships inside the AgentDojo repo, so a full clone provides it:\n"
            f"  git clone https://github.com/ethz-spylab/agentdojo.git {agentdojo_root()}\n"
            f"Or point {_ENV_ROOT} at an existing checkout."
        )
    return d


def data_dir() -> Path:
    """Root under which each benchmark keeps its exported probe suite, at
    `<benchmark>/data/{tasks,probes,injections}.jsonl` + `manifest.json`.

    This is the `benchmarks/` tree: everything belonging to one benchmark sits under
    `benchmarks/<benchmark>/` rather than in a separate top-level `data/`.

    Per-benchmark, not flat: task ids are unique only within a suite, so two
    benchmarks exported to one path would overwrite rather than merge.

    Tracked in git, unlike the benchmark checkouts -- consuming lp-bench should not
    require regenerating the probes (~9 min for AgentDojo) or cloning the benchmark.
    Override with AJAR_DATA_DIR (which then holds the `<benchmark>/` subdirs)."""
    override = os.environ.get(_ENV_DATA)
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "benchmarks"


def benchmark_data_dir(benchmark: str, root: Path | str | None = None) -> Path:
    """`<root>/<benchmark>/data` -- one benchmark's exported probe suite. The single
    place that composes the benchmark data path; everything reading probes/tasks/
    injections goes through here. `root` overrides the default `benchmarks/` tree
    (tests and callers pointing at an alternate export); otherwise `data_dir()`."""
    base = Path(root).expanduser() if root is not None else data_dir()
    return base / benchmark / "data"


def exported_benchmarks() -> list[str]:
    """Which benchmarks have a probes.jsonl on disk. Empty until something exports."""
    d = data_dir()
    if not d.is_dir():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and (p / "data" / "probes.jsonl").is_file()
    )


def describe() -> str:
    """One block of provenance, printed by bootstrap so a broken environment is
    obvious at setup time rather than at scoring time."""
    root = agentdojo_root()
    runs = runs_dir()
    data = data_dir()
    src = f" (from ${_ENV_ROOT})" if os.environ.get(_ENV_ROOT) else ""
    exported = exported_benchmarks()
    lines = [
        f"repo root    : {REPO_ROOT}",
        f"agentdojo    : {root}{src}" + ("" if root.is_dir() else "   [MISSING]"),
        f"runs dir     : {runs}" + ("" if runs.is_dir() else "   [MISSING -- read_closure will be empty]"),
        f"data dir     : {data}"
        + (
            f"   exported: {', '.join(exported)}"
            if exported
            else "   [nothing exported -- run scripts/export_probes.py]"
        ),
    ]
    return "\n".join(lines)
