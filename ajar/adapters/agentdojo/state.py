"""AgentDojoState: the executable, materialized state for one (task, prefix, injections).

Wraps a live AgentDojo `TaskEnvironment` plus a `FunctionsRuntime` bound to the
suite's granted tool list. Grading is routed through the suite's own
`_check_user_task_utility` / `_check_injection_task_security` (task_suite.py
~245-300), which try the `*_from_traces` variant first and fall back to the
plain `utility()` / `security()` predicates — see AJAR_AGENTDOJO_DESIGN.md §2.3
rule 2. `pre_env` is frozen once at materialization time (before any replay); it
is never mutated and is shared (not deep-copied) across `fork()`, matching
AgentDojo's own `run_task_with_pipeline` convention of a single `pre_environment`
snapshot per run.

Determinism note (design memo §6, "the clock field is not decoration"): the three
`datetime.datetime.now()` nondeterminism sites in AgentDojo are all in
workspace/email and workspace/cloud_drive. Banking has none, so no logical-clock
seeding is implemented here — this adapter is v0-scoped to banking and would need
that seeding before workspace/travel tasks could be trusted byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ajar.core.state import StateOutcome
from ajar.core.types import Candidate

if TYPE_CHECKING:
    from agentdojo.base_tasks import BaseUserTask
    from agentdojo.functions_runtime import FunctionsRuntime, TaskEnvironment
    from agentdojo.task_suite.task_suite import TaskSuite


@dataclass
class AgentDojoState:
    suite: "TaskSuite"
    user_task: "BaseUserTask"
    runtime: "FunctionsRuntime"
    pre_env: "TaskEnvironment"  # frozen at materialize() time; never mutated
    env: "TaskEnvironment"  # the working copy; mutated by execute()
    trace: list[Candidate] = field(default_factory=list)

    def fork(self) -> "AgentDojoState":
        # 64-81x cheaper than reloading from YAML (design memo §6). runtime and
        # pre_env are shared references — runtime is stateless (a name->Function
        # dict), pre_env is never mutated after materialize().
        return AgentDojoState(
            suite=self.suite,
            user_task=self.user_task,
            runtime=self.runtime,
            pre_env=self.pre_env,
            env=self.env.model_copy(deep=True),
            trace=list(self.trace),
        )

    def execute(self, candidate: Candidate) -> tuple[Any, str | None]:
        # raise_on_error=False (the default): a bad call returns an error string,
        # it never raises. The error string IS the signal, per the State contract.
        result, err = self.runtime.run_function(
            self.env, candidate.tool, dict(candidate.args), raise_on_error=False
        )
        self.trace.append(candidate)
        return result, err

    def outcome(self, against: str | None = None) -> StateOutcome:
        trace_calls = _to_function_calls(self.trace)
        utility = self.suite._check_user_task_utility(
            self.user_task, [], self.pre_env, self.env, trace_calls
        )
        security = None
        if against is not None:
            inj_task = self.suite.injection_tasks.get(against)
            if inj_task is not None:
                security = self.suite._check_injection_task_security(
                    inj_task, [], self.pre_env, self.env, trace_calls
                )
        return StateOutcome(utility=utility, security=security, trace=list(self.trace))

    def utility_with_output(self, model_output: str = "") -> bool:
        """Utility given the agent's final text answer — the same task-level logic
        AgentDojo's own re-scorer uses (utility_from_traces first, then utility).
        `outcome()` passes empty output, which is wrong for answer-graded tasks; this
        feeds the captured final answer so those tasks are scored correctly."""
        traces = _to_function_calls(self.trace)
        u = self.user_task.utility_from_traces(model_output, self.pre_env, self.env, traces)
        if u is None:
            u = self.user_task.utility(model_output, self.pre_env, self.env)
        return bool(u)

    def fingerprint(self) -> str:
        payload = json.dumps(self.env.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def raw(self) -> Any:
        return self.env


def _to_function_calls(trace: list[Candidate]):
    from agentdojo.functions_runtime import FunctionCall

    return [FunctionCall(function=c.tool, args=dict(c.args)) for c in trace]
