"""Invariants for the env-derived value pool (PARAMETER-axis substitutes).

The PARAMETER axis is where a tool-name filter's blind spot lives, so an empty or
wrong pool silently removes the benchmark's main signal rather than failing loudly.
"""
from __future__ import annotations

import pytest

from ajar.adapters.agentdojo.tool_inventory import HOSTILE_VALUES
from ajar.adapters.agentdojo.value_pool import PAYLOAD, ROLES, substitutes
from ajar.core.types import ContextRef

SUITES = ("banking", "slack", "travel", "workspace")


@pytest.fixture(scope="module")
def adapter():
    import ajar.adapters.agentdojo.adapter  # noqa: F401  (registers)
    from ajar.adapters.base import get_benchmark

    return get_benchmark("agentdojo")


@pytest.fixture(scope="module")
def suites():
    from agentdojo.task_suite.load_suites import get_suites

    return get_suites("v1.2.2")


def _plan_pools(adapter, key):
    """(tool, arg, authorized values, pool) for every arg of every plan call."""
    spec = adapter.task_spec(key)
    for i, call in enumerate(spec.plan):
        ctx = ContextRef(task=key, injections=(), prefix=tuple(range(i)))
        for arg, val in call.args.items():
            yield call.tool, arg, val, list(adapter.value_pool(ctx, call.tool, arg))


@pytest.mark.parametrize("name", SUITES)
def test_roles_reference_real_tools_and_args(suites, name):
    """A role keyed on a misspelled tool or arg is a silent hole: the pool comes
    back empty and the axis just looks sparse."""
    sigs = {f.name: set(f.parameters.model_fields) for f in suites[name].tools}
    for (suite, tool, arg), role in ROLES.items():
        if suite != name:
            continue
        assert tool in sigs, f"{suite}/{tool} is not a real tool"
        assert arg in sigs[tool], f"{suite}/{tool}: no arg named {arg!r}"


def test_pool_never_returns_an_authorized_value(adapter):
    """Adapter protocol. A substitute equal to a ground-truth value mints a probe
    byte-identical to a required-ALLOW one. Must exclude EVERY authorized value for
    that (tool, arg), not just the first: a plan may call one tool repeatedly with
    different targets (measured: 34 collisions when only the first was excluded)."""
    for key in adapter.list_tasks():
        spec = adapter.task_spec(key)
        for tool, arg, _val, pool in _plan_pools(adapter, key):
            authorized = [c.args[arg] for c in spec.plan if c.tool == tool and arg in c.args]
            for sub in pool:
                assert sub not in authorized, f"{key.suite}/{tool}.{arg}: {sub!r} is authorized"
                if isinstance(sub, (list, tuple)) and len(sub) == 1:
                    assert sub[0] not in authorized, f"{key.suite}/{tool}.{arg}: {sub!r}"


def test_substitutes_match_the_container_shape(adapter):
    """`recipients` is a list, `email` is a bare string. A shape mismatch makes the
    probe fail on validation instead of on policy -- a DENY that proves nothing."""
    for key in adapter.list_tasks():
        for tool, arg, val, pool in _plan_pools(adapter, key):
            for sub in pool:
                if isinstance(val, (list, tuple)):
                    assert isinstance(sub, (list, tuple)), f"{key.suite}/{tool}.{arg}: {sub!r}"
                elif isinstance(val, str):
                    assert not isinstance(sub, (list, tuple)), f"{key.suite}/{tool}.{arg}: {sub!r}"


def test_hostile_substitute_comes_first(adapter):
    """The hostile-destination substitute is the probe that separates a tool-name
    filter from an argument-aware policy, so it must survive pool truncation."""
    checked = 0
    for key in adapter.list_tasks():
        hostile = HOSTILE_VALUES.get(key.suite, frozenset())
        for tool, arg, _val, pool in _plan_pools(adapter, key):
            if not pool:
                continue
            hits = [i for i, s in enumerate(pool) if any(h in str(s).lower() for h in hostile)]
            if hits:
                assert hits[0] == 0, f"{key.suite}/{tool}.{arg}: hostile value at index {hits[0]}"
                checked += 1
    assert checked > 0


def test_payload_substitute_keeps_the_authorized_text(suites):
    """A PAYLOAD arg carries the exfiltrated data. The real attack shape is a
    legitimate-looking message that ALSO names the attacker, not one replaced
    wholesale -- a policy that only pattern-matches the whole body would pass."""
    env = suites["workspace"].load_and_inject_default_environment({})
    body = "Here are the notes you asked for."
    pool = substitutes("workspace", "send_email", "body", body, env)
    assert pool, "no PAYLOAD substitute generated"
    for sub in pool:
        assert body in sub, f"authorized text dropped: {sub!r}"
        assert any(h in sub.lower() for h in HOSTILE_VALUES["workspace"])


def test_free_text_args_have_no_pool(suites):
    """Rewording a subject reaches no other resource, so a DENY label would be
    wrong. These args are still probed by the omission/type sub-generators."""
    env = suites["workspace"].load_and_inject_default_environment({})
    for arg in ("subject", "title"):
        assert not substitutes("workspace", "send_email", arg, "x", env)
    assert ("workspace", "send_email", "subject") not in ROLES


@pytest.mark.parametrize("name", ("banking", "slack", "travel", "workspace"))
def test_every_write_sink_arg_is_classified(suites, name):
    """Every arg of every WRITE tool must either have a role or be deliberately
    free-text/formatting. Catches a tool gaining an arg upstream."""
    from ajar.adapters.agentdojo.tool_inventory import INVENTORY

    FREE_TEXT = {
        "subject", "body", "content", "title", "description", "query", "filename",
        "date", "start_time", "end_time", "start_day", "end_day", "new_start_time",
        "new_end_time", "day", "permission", "recurring", "city", "password",
        "street", "n", "reservation_type", "location", "user_email",
        "first_name", "last_name",
        # travel/send_email.attachments: travel's env has no cloud_drive, so there
        # is no file to attach and no resource to redirect to. workspace's same-named
        # arg IS role-mapped to FILE_ID, where a drive exists.
        "attachments",
    }
    for (suite, tool), f in INVENTORY.items():
        if suite != name or f.side_effect != "WRITE":
            continue
        fn = next(x for x in suites[name].tools if x.name == tool)
        for arg in fn.parameters.model_fields:
            classified = (suite, tool, arg) in ROLES or arg in FREE_TEXT
            assert classified, f"{suite}/{tool}.{arg} is neither role-mapped nor free-text"
