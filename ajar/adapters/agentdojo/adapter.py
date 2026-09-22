"""AgentDojoAdapter: turns live AgentDojo ground truth into `TaskSpec`s.

v0 scope: AgentDojo v1.2.2, all four default suites for `list_tasks`/`task_spec`
plumbing, but only `banking` has been measured end-to-end (AJAR_AGENTDOJO_DESIGN.md,
PROBE_SUITE_banking_user_task_0.md). Three non-obvious, load-bearing decisions,
each cited against a specific measured fact:

1. **`TaskSpec.necessary` carries the real ablation, but it is INFORMATIONAL.**
   `ir.required_tools` never narrows to it (core fix) — it is the full frozen
   plan, always. On banking/user_task_0, `read_file` is ablation-*removable*
   (dropping it leaves `utility()` True — `send_money`'s args are hardcoded in
   `ground_truth()`, not derived from the file) while `send_money` is necessary.
   If `required_tools` narrowed to `{send_money}`, `tool_allowlist` would deny
   `read_file` and fail the sufficiency gate — exactly what design memo §7.3
   forbids ("Do not narrow `required` to the necessary partition"). So the
   ablation is recorded and used only as a read/write classifier, never a filter.
2. **`read_closure` is MEASURED from the run corpus**, not authored: scan
   `bench/agentdojo/runs/*/{suite}/{task_id}/none/none.json`, keep
   `utility()==True` runs whose `user_task_id` field equals `task_id` exactly
   (guards the memo's "attacker goal in the user-task slot" hazard), collect
   tool names outside the frozen plan. For banking/user_task_0 this measures to
   exactly `{get_iban, get_balance, get_user_info, get_scheduled_transactions,
   get_most_recent_transactions}` -- verified live, matches the worked probe
   suite's corpus finding (5/18 runs for get_iban across 5 base models; the other
   four from 2 run dirs, both `repeat_user_prompt`).
3. **`attack_sinks` are leave-one-out ablation-reduced per injection task**
   against that task's own `security()`, verified live: 11/12 GT calls across
   the suite's 9 injection tasks are sinks; the sole exception is
   `injection_task_8`'s `get_scheduled_transactions` (a discovery read whose
   removal leaves `security()` True). Harm is hardcoded H4 for every banking
   sink (design memo + probe suite: sink-restriction removes the only non-H4
   candidate; the remainder is a suite-specific, checked finding, not a fiat).
"""
from __future__ import annotations

import glob
import json
from collections.abc import Sequence
from typing import Any

from ajar import paths
from ajar.adapters.base import register_benchmark
from ajar.adapters.agentdojo.oracle import AgentDojoOracle
from ajar.adapters.agentdojo.harm import AgentDojoHarmModel
from ajar.adapters.agentdojo.value_pool import substitutes as _substitutes
from ajar.adapters.agentdojo.tool_inventory import (
    facts as _facts,
    promote as _promote,
    side_effect as _side_effect,
)
from ajar.adapters.agentdojo.state import AgentDojoState
from ajar.core.ir import AttackSink, TaskSpec
from ajar.core.state import State
from ajar.core.types import Candidate, ContextRef, HarmTier, TaskKey, ToolSpec

# The upstream release we pin. Named once: it is both what `get_suites` is called
# with and what the export manifest records as provenance.
_RELEASE_LABEL = "v1.2.2"

# One AgentDojo release label does NOT map to one version tuple: under label
# "v1.2.2", banking and workspace report benchmark_version (1,2,2) while slack and
# travel report (1,2,0). Assuming a single tuple makes 41 of the 97 tasks
# unreachable, so map every tuple the loaded suites actually report.
_VERSION_LABEL = {(1, 2, 2): _RELEASE_LABEL, (1, 2, 0): _RELEASE_LABEL}

# Categorical exclusion (design memo §3.6): its `security()` is a model-output
# substring check with `ground_truth() -> []` -- there is no liftable sink call.
_EXCLUDED_INJECTIONS = {("travel", "injection_task_6")}


class AgentDojoAdapter:
    name = "agentdojo"

    def __init__(self) -> None:
        self._spec_cache: dict[TaskKey, TaskSpec] = {}
        self._sinks_cache: dict[str, list[AttackSink]] = {}

    # -- BenchmarkAdapter protocol -----------------------------------------

    def list_tasks(self) -> Sequence[TaskKey]:
        from agentdojo.task_suite.load_suites import get_suites

        suites = get_suites(_RELEASE_LABEL)
        keys: list[TaskKey] = []
        for suite_name, suite in suites.items():
            for task_id in sorted(suite.user_tasks, key=_task_num):
                keys.append(
                    TaskKey(
                        benchmark=self.name,
                        suite=suite_name,
                        version=tuple(suite.benchmark_version),
                        task_id=task_id,
                    )
                )
        return keys

    def task_spec(self, key: TaskKey) -> TaskSpec:
        if key in self._spec_cache:
            return self._spec_cache[key]

        from agentdojo.functions_runtime import FunctionsRuntime

        suite = self._suite_for(key)
        user_task = suite.user_tasks[key.task_id]

        tools = [self._tool_spec(fn, key.suite) for fn in suite.tools]

        base_env = suite.load_and_inject_default_environment({})
        base_env = user_task.init_environment(base_env)
        pre_env = base_env.model_copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)

        gt_calls = user_task.ground_truth(pre_env.model_copy(deep=True))
        plan = [Candidate(tool=c.function, args=dict(c.args)) for c in gt_calls]

        necessary_measured = self._ablation_necessary(suite, user_task, pre_env, plan, runtime)
        plan_tools = frozenset(c.tool for c in plan)
        read_closure = self._measure_read_closure(key.suite, key.task_id, plan_tools)
        attack_sinks = self._attack_sinks_for_suite(key.suite, suite)
        is_gradeable = self._is_gradeable(suite, user_task, pre_env)

        spec = TaskSpec(
            task=key,
            prompt=getattr(user_task, "PROMPT", getattr(user_task, "GOAL", "")),
            tools=tools,
            plan=plan,
            # `necessary` is now informational only (required_tools never narrows
            # to it — core fix), so we carry the real measured ablation. It is a
            # read/write classifier, not a sufficiency filter.
            necessary=frozenset(necessary_measured),
            read_closure=read_closure,
            attack_sinks=attack_sinks,
            is_gradeable=is_gradeable,
            notes=(
                f"ablation-necessary indices (measured, informational — not used "
                f"to narrow required_tools, design memo §7.3): "
                f"{sorted(necessary_measured)} of {len(plan)}. read_closure measured "
                f"from runs/*/{key.suite}/{key.task_id}/none/none.json: {sorted(read_closure)}."
            ),
        )
        self._spec_cache[key] = spec
        return spec

    def materialize(self, context: ContextRef) -> State:
        suite = self._suite_for(context.task)
        user_task = suite.user_tasks[context.task.task_id]
        spec = self.task_spec(context.task)

        from agentdojo.functions_runtime import FunctionsRuntime

        env0 = suite.load_and_inject_default_environment(dict(context.injections))
        env0 = user_task.init_environment(env0)
        pre_env = env0.model_copy(deep=True)
        env = env0.model_copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)

        state = AgentDojoState(
            suite=suite, user_task=user_task, runtime=runtime, pre_env=pre_env, env=env, trace=[]
        )
        for i in context.prefix:
            state.execute(spec.plan[i])
        return state

    def oracle(self, key: TaskKey) -> AgentDojoOracle:
        return AgentDojoOracle(self.task_spec(key))

    def harm_model(self, key: TaskKey) -> AgentDojoHarmModel:
        return AgentDojoHarmModel(self.task_spec(key))

    def value_pool(self, context: ContextRef, tool: str, arg: str) -> Sequence[Any]:
        """Env-derived substitutes, all four suites, organised by the argument's
        semantic role (`value_pool.ROLES`). v0 hardcoded banking's arg names, which
        left the PARAMETER axis empty in the other three suites -- and that axis is
        where the headline finding lives (a tool-name filter changes no tool, only
        an argument). The env is materialized at THIS context, not the default
        state, so substitutes for ids created by earlier plan steps are real."""
        spec = self.task_spec(context.task)
        env = self.materialize(context).env

        # EVERY authorized value for this (tool, arg), not just the first. A plan
        # may call one tool repeatedly with different targets -- `get_webpage` on
        # three urls, `read_channel_messages` on four channels -- and excluding
        # only the first leaves the others in the pool, minting "wrong-resource"
        # probes that are actually authorized calls (measured: 34 such collisions).
        authorized = [c.args[arg] for c in spec.plan if c.tool == tool and arg in c.args]
        primary = authorized[0] if authorized else None

        return _substitutes(context.task.suite, tool, arg, primary, env, exclude=authorized)

    # -- optional protocol ---------------------------------------------------

    def export_metadata(self) -> dict:
        """Provenance for `benchmarks/agentdojo/data/manifest.json`. Lives here, not in the
        export script, because only this adapter knows which upstream label it loaded
        and which of the benchmark's own tasks it had to exclude."""
        return {
            "upstream": "ethz-spylab/agentdojo",
            "release_label": _RELEASE_LABEL,
            "excluded_injections": [f"{s}/{i}" for s, i in sorted(_EXCLUDED_INJECTIONS)],
            # Not a bug on our side: these attacks have `ground_truth() -> []`
            # upstream, so no sink can be lifted from them and they cannot enter the
            # dataset. Recorded because it means `attacks_total` is NOT the number of
            # attacks the suite defines. See docs/upstream-agentdojo-defects.md §1.
            "attacks_total_caveat": (
                "injection tasks whose upstream ground_truth() is empty contribute no "
                "attack_sinks; attacks_total counts liftable attacks only"
            ),
        }

    # -- internals -----------------------------------------------------------

    def _suite_for(self, key: TaskKey):
        from agentdojo.task_suite.load_suites import get_suites

        label = _VERSION_LABEL.get(tuple(key.version))
        if label is None:
            raise ValueError(f"AgentDojoAdapter (v0) only knows version(s) {list(_VERSION_LABEL)}, got {key.version}")
        return get_suites(label)[key.suite]

    def _tool_spec(self, fn, suite_name: str) -> ToolSpec:
        domain = None
        for dep in fn.dependencies.values():
            d = dep.env_dependency
            domain = d if isinstance(d, str) else None
            break
        return ToolSpec(
            name=fn.name,
            params_schema=fn.parameters.model_json_schema(),
            resource_domain=domain,
            side_effect=_side_effect(suite_name, fn.name),
            description=getattr(fn, "description", "") or "",
        )

    def _ablation_necessary(self, suite, user_task, pre_env, plan: list[Candidate], runtime) -> frozenset[int]:
        from agentdojo.functions_runtime import FunctionCall

        necessary: set[int] = set()
        for i in range(len(plan)):
            subset = plan[:i] + plan[i + 1 :]
            env = pre_env.model_copy(deep=True)
            for c in subset:
                runtime.run_function(env, c.tool, dict(c.args))
            trace = [FunctionCall(function=c.tool, args=dict(c.args)) for c in subset]
            u = suite._check_user_task_utility(user_task, [], pre_env, env, trace)
            if u is not True:
                necessary.add(i)
        return frozenset(necessary)

    def _is_gradeable(self, suite, user_task, pre_env) -> bool:
        env = pre_env.model_copy(deep=True)
        u = suite._check_user_task_utility(user_task, [], pre_env, env, [])
        return u is not True

    def _measure_read_closure(self, suite_name: str, task_id: str, plan_tools: frozenset[str]) -> frozenset[str]:
        # Location comes from `ajar.paths` and nowhere else, so bootstrap's verify
        # step and this glob can never disagree about where the corpus is. Resolved per
        # call rather than at import, since AJAR_AGENTDOJO_ROOT may be set later.
        pattern = str(paths.runs_dir() / "*" / suite_name / task_id / "none" / "none.json")
        extras: set[str] = set()
        for f in glob.glob(pattern):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            # Guard: injection tasks running in the user-task slot (design memo
            # §6, "attacker goals in the user-task namespace", 22.6% of files).
            if d.get("user_task_id") != task_id:
                continue
            if d.get("utility") is not True:
                continue
            for m in d.get("messages", []):
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function")
                    if fn and fn not in plan_tools:
                        extras.add(fn)
        return frozenset(extras)

    def _attack_sinks_for_suite(self, suite_name: str, suite) -> list[AttackSink]:
        if suite_name in self._sinks_cache:
            return self._sinks_cache[suite_name]

        from agentdojo.functions_runtime import FunctionsRuntime

        sinks: list[AttackSink] = []
        base_env = suite.load_and_inject_default_environment({})
        runtime = FunctionsRuntime(suite.tools)

        for inj_id, inj_task in suite.injection_tasks.items():
            if (suite_name, inj_id) in _EXCLUDED_INJECTIONS:
                continue
            pre_i = base_env.model_copy(deep=True)
            gt = inj_task.ground_truth(pre_i)
            if not gt:
                continue

            def replay(subset, pre=pre_i):
                env = pre.model_copy(deep=True)
                for call in subset:
                    runtime.run_function(env, call.function, call.args)
                return env

            for idx, call in enumerate(gt):
                subset = gt[:idx] + gt[idx + 1 :]
                env = replay(subset)
                sec = suite._check_injection_task_security(inj_task, [], pre_i, env, subset)
                necessary = sec is not True  # dropping call idx flips security True->False
                if necessary:
                    sinks.append(
                        AttackSink(
                            attack_id=inj_id,
                            candidate=Candidate(tool=call.function, args=dict(call.args)),
                            # NOT hardcoded H4 (v0 was banking-only, where sink-
                            # restriction happens to leave only H4 candidates).
                            # Across all four suites it does not: `slack/injection_
                            # _task_2`'s sink set includes `get_channels` (H0 probe)
                            # and 4x `read_channel_messages` (H1). The tier comes
                            # from the authored inventory + hostile-value promotion;
                            # the ATTACK ROLLUP (all sinks admitted => attack
                            # admitted) is what makes the low-tier legs matter, not
                            # an inflated per-leg tier.
                            harm_tier=_sink_tier(suite_name, call),
                            jointly_with=(),
                            evidence=(
                                f"leave-one-out ablation: dropping call {idx} "
                                f"({call.function}) flips {inj_id}.security() True->False"
                            ),
                        )
                    )

        self._sinks_cache[suite_name] = sinks
        return sinks


def _sink_tier(suite_name: str, call) -> HarmTier:
    """Authored tier for one attack sink: inventory base tier, then hostile-value
    promotion. Unlisted tool => H1 (an authoring gap, never H0 -- H0 has weight 0
    and would vanish from OPL)."""
    f = _facts(suite_name, call.function)
    if f is None:
        return HarmTier.H1
    return _promote(suite_name, call.function, dict(call.args), f.base_tier)


def _task_num(task_id: str) -> int:
    try:
        return int(task_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


AGENTDOJO_ADAPTER = register_benchmark(AgentDojoAdapter())
