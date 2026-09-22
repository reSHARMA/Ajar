"""Build probe suites and write them to the exported dataset (benchmarks/<benchmark>/data/).

    # one task (default: banking/user_task_0), mechanical prompts
    bench/agentdojo/.venv/bin/python -m ajar.build_probes

    # a specific task, with LLM probes + fluent prompts
    bench/agentdojo/.venv/bin/python -m ajar.build_probes --suite workspace --task user_task_13 --llm

    # every gradeable task in a suite / in all four
    bench/agentdojo/.venv/bin/python -m ajar.build_probes --all
    bench/agentdojo/.venv/bin/python -m ajar.build_probes --all --all-suites

This is a thin CLI over `scripts/export_probes.py`. It used to write a second on-disk
layout (one JSON per task under probes/), which meant two writers producing two copies
of the same probes, each with its own idea of what a complete export looks like -- and
nothing saying which was authoritative. The dataset in data/ is the answer, so this
translates its arguments and delegates.

Building a subset MERGES into the dataset: rebuilding one task replaces that task's
rows and leaves the other 96 alone. See export_probes for the merge and how the
manifest is recomputed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_VERSIONS = {"banking": (1, 2, 2), "workspace": (1, 2, 2), "slack": (1, 2, 0), "travel": (1, 2, 0)}

# scripts/ is not a package (no __init__.py), so it is not importable by name.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _export_main():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import export_probes

    return export_probes.main


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate probe suites into benchmarks/<benchmark>/data/.")
    ap.add_argument("--suite", default="banking", choices=list(_VERSIONS))
    ap.add_argument("--task", default="user_task_0", help="user task id, e.g. user_task_0")
    ap.add_argument("--all", action="store_true",
                    help="build EVERY gradeable task (just --suite, or all four with --all-suites)")
    ap.add_argument("--all-suites", action="store_true", help="with --all: span all four suites")
    ap.add_argument("--llm", action="store_true", help="use Bedrock for LLM probes + fluent prompts")
    args = ap.parse_args()

    argv: list[str] = []
    if args.all:
        if not args.all_suites:
            argv.append(args.suite)
    else:
        argv += [args.suite, "--task", args.task]
    if args.llm:
        argv.append("--llm")

    raise SystemExit(_export_main()(argv))


if __name__ == "__main__":
    main()
