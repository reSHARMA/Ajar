"""Offline replay of AgentDojo's OpenAILLMToolFilter — a REAL defense, no API key.

The filter's decision is a pure function of the prompt (injections run at a later
pipeline stage and never reach it), so its policy is fully recoverable from the
replies already logged in the runs corpus. We reproduce the EXACT containment logic
(`tool_name in reply_text`, openai_llm.py:233-238) against the suite's tool names,
including its substring-collision behavior. Stateless, tool-name-only, so its
declared privilege_surface is ("TOOL",): it can DENY on the tool axis and only ever
ALLOW on the parameter/sequence axes — which is exactly what we want to measure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ajar import paths
from ajar.core.ir import TaskSpec
from ajar.core.types import Candidate, ContextRef, Verdict
from ajar.defenses.base import PolicyUnavailable, register_defense

# Which logged run supplies the replies. The corpus LOCATION comes from
# `ajar.paths` and nowhere else: this used to hand-build the path with `../../..`,
# which meant AJAR_AGENTDOJO_ROOT moved the adapter's read-closure glob but not
# this replay, and the two silently disagreed about where the corpus was.
_RUN_NAME = "gpt-4o-2024-05-13-tool_filter"
_FILTER_PROMPT_PREFIX = "Your task is to filter the list of tools"


def _default_runs_root() -> str:
    """Resolved per call, not at import: the env override may be set later."""
    return str(paths.runs_dir() / _RUN_NAME)


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(block.get("text") or block.get("content") or "")
        else:
            parts.append(str(block))
    return "".join(parts)


class ReplayedToolFilter:
    name = "tool_filter"
    privilege_surface = ("TOOL",)

    def __init__(self, runs_root: str | None = None):
        # None means "ask paths at prepare() time"; an explicit root (tests) wins.
        self._runs_root = runs_root

    @property
    def runs_root(self) -> str:
        return self._runs_root if self._runs_root is not None else _default_runs_root()

    def _reply_text(self, suite: str, task_id: str) -> str:
        path = Path(self.runs_root) / suite / task_id / "none" / "none.json"
        if not path.exists():
            raise PolicyUnavailable(f"no logged tool_filter run at {path}")
        msgs = json.loads(path.read_text()).get("messages", [])
        for i, m in enumerate(msgs):
            if m.get("role") == "user" and _as_text(m.get("content")).startswith(_FILTER_PROMPT_PREFIX):
                for nxt in msgs[i + 1:]:
                    if nxt.get("role") == "assistant":
                        return _as_text(nxt.get("content"))
        raise PolicyUnavailable(f"no filter reply found in {path}")

    def prepare(self, spec: TaskSpec, llm=None) -> frozenset[str]:
        reply = self._reply_text(spec.task.suite, spec.task.task_id)
        # openai_llm.py:235 exactly: keep a tool iff its name is a substring of the reply.
        return frozenset(t.name for t in spec.tools if t.name in reply)

    def decide(self, policy: frozenset[str], context: ContextRef, candidate: Candidate) -> Verdict:
        # Stateless and arg-blind: allow any call whose tool name was kept.
        return Verdict.ALLOW if candidate.tool in policy else Verdict.DENY


register_defense(ReplayedToolFilter())
