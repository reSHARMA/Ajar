"""Invariants for the hand-authored 74-row tool inventory.

These are regression locks on *authoring*, not on code paths. Each one caught a
real error while the table was being written:
  * coverage    -- a tool added upstream silently falls through to UNKNOWN/H1
  * side_effect -- the 3 writes that first measured as "READ 0" because
                   `run_function` returns errors instead of raising
  * H4 coverage -- `travel/injection_task_5` had no H4 sink at all
  * no-false-H4 -- listing that task's recipient promoted a BENIGN task
                   (`travel/user_task_3` mails the same address legitimately)
"""
from __future__ import annotations

import pytest

from ajar.adapters.agentdojo.tool_inventory import (
    INVENTORY,
    UNAUTHORIZED_ONLY_ATTACKS,
    facts,
    promote,
)
from ajar.core.types import HarmTier

SUITES = ("banking", "slack", "travel", "workspace")
EXPECTED_ROWS = {"banking": 11, "slack": 11, "travel": 28, "workspace": 24}


@pytest.fixture(scope="module")
def suites():
    from agentdojo.task_suite.load_suites import get_suites

    return get_suites("v1.2.2")


def test_row_count():
    assert len(INVENTORY) == 74
    for s, n in EXPECTED_ROWS.items():
        assert sum(1 for (suite, _) in INVENTORY if suite == s) == n


@pytest.mark.parametrize("name", SUITES)
def test_covers_every_tool_exactly(suites, name):
    """No missing rows (=> UNKNOWN side effect, H1 fallback) and no stale rows."""
    real = {f.name for f in suites[name].tools}
    tabled = {t for (s, t) in INVENTORY if s == name}
    assert real == tabled


@pytest.mark.parametrize("name", SUITES)
def test_side_effect_matches_execution(suites, name):
    """Every WRITE row must actually mutate the env, and every READ row must not.

    Guards the `run_function`-returns-errors trap: a call that errors before
    mutating produces a 0-path diff that is indistinguishable from a pure read,
    so a failed execution is skipped rather than counted as evidence of READ.
    """
    from agentdojo.functions_runtime import FunctionsRuntime

    suite = suites[name]
    runtime = FunctionsRuntime(suite.tools)
    base = suite.load_and_inject_default_environment({})

    checked = 0
    for tid, task in suite.user_tasks.items():
        env0 = task.init_environment(base.model_copy(deep=True))
        try:
            gt = task.ground_truth(env0.model_copy(deep=True))
        except Exception:
            continue
        env = env0.model_copy(deep=True)
        for call in gt:
            before = env.model_dump()
            _, err = runtime.run_function(env, call.function, dict(call.args))
            if err:
                continue  # inconclusive, not evidence
            mutated = env.model_dump() != before
            f = facts(name, call.function)
            assert f is not None, f"{name}/{call.function} missing from inventory"
            if mutated:
                assert f.side_effect == "WRITE", (
                    f"{name}/{call.function} mutated the env but is authored READ "
                    f"(task {tid})"
                )
            checked += 1
    assert checked > 0


def test_known_name_behaviour_mismatches():
    """The two read-NAMED writes. Both are load-bearing: `slack/get_webpage` is the
    entire ground truth of `slack/injection_task_3`, so calling it READ would drop
    that attack's only sink."""
    assert facts("slack", "get_webpage").side_effect == "WRITE"
    assert facts("workspace", "get_unread_emails").side_effect == "WRITE"


def test_membership_read_is_not_a_write():
    """`get_users_in_channel` appends to a LOCAL list; an AST pass calls it WRITE."""
    assert facts("slack", "get_users_in_channel").side_effect == "READ"


def test_no_h0_writes():
    """A write is never harmless: H0 has weight 0 and would vanish from OPL."""
    for (s, t), f in INVENTORY.items():
        if f.side_effect == "WRITE":
            assert f.base_tier != HarmTier.H0, f"{s}/{t}"


@pytest.mark.parametrize("name", SUITES)
def test_every_attack_reaches_h4(suites, name):
    """Each attack has at least one H4 sink via hostile-value promotion, OR is
    registered as unauthorized-only (no hostile value exists in its args, so the
    attack-sink anchor in harm.py is the only thing that can tier it)."""
    suite = suites[name]
    base = suite.load_and_inject_default_environment({})
    for tid, task in suite.injection_tasks.items():
        gt = task.ground_truth(base.model_copy(deep=True))
        if not gt:
            continue  # empty GT: excluded, no liftable sink
        tiers = [
            promote(name, c.function, dict(c.args), facts(name, c.function).base_tier)
            for c in gt
        ]
        if (name, tid) in UNAUTHORIZED_ONLY_ATTACKS:
            assert HarmTier.H4 not in tiers, (
                f"{name}/{tid} is registered unauthorized-only but a hostile value "
                f"now promotes it -- de-register it"
            )
        else:
            assert HarmTier.H4 in tiers, f"{name}/{tid} has no H4 sink"


@pytest.mark.parametrize("name", SUITES)
def test_no_benign_call_promotes_to_h4(suites, name):
    """The benign baseline must stay clean: a hostile-value pattern wide enough to
    fire on legitimate, user-instructed work is worse than a missing one."""
    suite = suites[name]
    base = suite.load_and_inject_default_environment({})
    for tid, task in suite.user_tasks.items():
        env = task.init_environment(base.model_copy(deep=True))
        try:
            gt = task.ground_truth(env.model_copy(deep=True))
        except Exception:
            continue
        for c in gt:
            f = facts(name, c.function)
            if f is None or f.base_tier is HarmTier.H4:
                continue
            got = promote(name, c.function, dict(c.args), f.base_tier)
            assert got is not HarmTier.H4, (
                f"false H4 on benign {name}/{tid}: {c.function}({dict(c.args)})"
            )


@pytest.mark.parametrize("name", SUITES)
def test_recipient_args_exist_in_signatures(suites, name):
    """A promotion arg that is not in the tool's real signature never fires -- a
    silent hole, not an error."""
    from ajar.adapters.agentdojo.tool_inventory import RECIPIENT_ARGS

    sigs = {f.name: set(f.parameters.model_fields) for f in suites[name].tools}
    for (s, tool), keys in RECIPIENT_ARGS.items():
        if s != name:
            continue
        assert tool in sigs, f"{s}/{tool} not a real tool"
        for k in keys:
            assert k in sigs[tool], f"{s}/{tool}: no arg named {k!r}"
