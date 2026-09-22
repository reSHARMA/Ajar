#!/usr/bin/env bash
# Build the lp-bench environment from a bare checkout.
#
#   ./scripts/bootstrap.sh
#
# Idempotent: re-running skips work that is already done. What it does:
#   1. clone ethz-spylab/agentdojo into bench/agentdojo (gitignored, ~515M)
#   2. create bench/agentdojo/.venv and install agentdojo into it
#   3. install ajar (editable, with dev extras) into that SAME venv, so `-m ajar.*`
#      needs no PYTHONPATH
#   4. verify agentdojo imports, the four suites load, and the run corpus is present
#   5. run the test suite, which reproduces the walkthrough's numbers exactly
#
# Why one shared venv: lp-bench must import the benchmark it adapts, and agentdojo
# pins its own dependency tree. Installing ajar alongside it (rather than
# reaching in over PYTHONPATH) keeps a single resolved environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTDOJO_DIR="${AJAR_AGENTDOJO_ROOT:-$REPO_ROOT/bench/agentdojo}"
VENV="$AGENTDOJO_DIR/.venv"
PY="$VENV/bin/python"

# agentdojo requires >=3.10; the interpreter is pinned so the resolved tree does
# not drift with whatever `python3` happens to be on PATH.
PYTHON_VERSION="${AJAR_PYTHON_VERSION:-3.12}"
AGENTDOJO_REPO="${AJAR_AGENTDOJO_REPO:-https://github.com/ethz-spylab/agentdojo.git}"

say() { printf '\n=== %s\n' "$1"; }

command -v uv >/dev/null 2>&1 || {
  echo "error: uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

say "1/5  agentdojo checkout"
if [ -d "$AGENTDOJO_DIR/.git" ]; then
  echo "already present at $AGENTDOJO_DIR — skipping clone"
else
  # --depth 1: we pin a version and never need the history.
  git clone --depth 1 "$AGENTDOJO_REPO" "$AGENTDOJO_DIR"
fi

say "2/5  venv + agentdojo"
[ -x "$PY" ] || uv venv --python "$PYTHON_VERSION" "$VENV"
uv pip install --python "$PY" -e "$AGENTDOJO_DIR"

say "3/5  ajar (editable, same venv)"
uv pip install --python "$PY" -e "$REPO_ROOT[dev]"

say "4/5  verify"
"$PY" - <<'PYCODE'
import agentdojo  # noqa: F401
from agentdojo.task_suite.load_suites import get_suites

from ajar import paths

suites = get_suites("v1.2.2")
print("agentdojo    :", agentdojo.__file__)
for name, suite in sorted(suites.items()):
    version = ".".join(map(str, suite.benchmark_version))
    print(
        f"  {name:10s} v{version:6s} "
        f"{len(suite.user_tasks):3d} user tasks  {len(suite.injection_tasks):3d} injection tasks"
    )
print(paths.describe())

# The corpus is load-bearing, not optional: without it the read closure is empty
# and legitimate reads are scored as over-privilege.
runs = paths.require_runs_dir()
n = sum(1 for _ in runs.iterdir() if _.is_dir())
print(f"run corpus   : {n} pipeline dirs")
PYCODE

say "5/5  tests"
# The bootstrap is not "done" until the environment reproduces the walkthrough's
# numbers -- a venv that imports cleanly but scores differently is the failure
# this step exists to catch.
(cd "$REPO_ROOT" && "$PY" -m pytest)

cat <<EOF

Done. Run the reference example with:

  $PY -m ajar.run_example     # banking/user_task_0, 62 probes
  $PY -m ajar.run_benchmark   # the whole banking suite
  (cd $REPO_ROOT && $PY -m pytest)
EOF
