"""The Explorer — a bounded, least-privilege, agentic reader.

Given (task, read-only tools, ground truth), it calls read tools to build the
minimal task-relevant StateModel. It is the ONE genuinely agentic component: which
read to make next is open-ended and depends on what prior reads revealed, so it
cannot be a fixed script. Everything downstream (generation) is deterministic.

TERMINATION / SUFFICIENCY — the ground truth is the checkable target. The Explorer
reads until it can *explain every authorized argument* from read data (fill the
provenance map), or until a read budget is hit. "Explain the ground truth" is both
the stop condition and the proof exploration was sufficient; whatever it still
cannot explain becomes `unexplained_args` (a coverage hole, reported not hidden).

LEAST-PRIVILEGE READING — every read must be justified against the task. The
Explorer should not sweep the world; its read set is meant to be minimal-and-
sufficient, which is *why* that set is a defensible discovery-read closure.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ajar.connect.base import GroundTruth, ReadTool, TaskDescriptor
from ajar.explore.statemodel import StateModel
from ajar.llm.base import LLMClient


@runtime_checkable
class Explorer(Protocol):
    def explore(
        self,
        task: TaskDescriptor,
        read_tools: list[ReadTool],
        ground_truth: GroundTruth,
        llm: LLMClient,
        *,
        max_reads: int = 32,
    ) -> StateModel: ...


# ---------------------------------------------------------------------------
# Reference loop (the contract; a concrete impl is the next build step).
#
#   model = StateModel(task_id=task.task_id)
#   history = []
#   while budget remaining and provenance incomplete:
#       step = llm.complete_json(
#           EXPLORE_PROMPT(task, read_tool_schemas, ground_truth.authorized,
#                          discovered=model, history=history),
#           schema=STEP_SCHEMA)      # {"action":"read"|"done", "tool","args","why"}
#       if step.action == "done": break
#       assert step.tool in {t.name for t in read_tools}   # ENFORCE read-only
#       result, err = call_read_tool(step.tool, step.args)  # real read, creds bound
#       history.append((step, result, err))
#       model = ingest(model, step, result)   # extract Entities; append to read_trace
#   model.provenance      = reconcile(ground_truth.authorized, model)  # arg <- read
#   model.value_pools     = index_entities_by_type(model.entities)      # wrong-resource
#   model.unexplained_args = gt_args_with_no_source(model.provenance)   # coverage holes
#   return model
#
# Notes that make it sound, not just plausible:
#   * The LLM only PROPOSES which read to make and how to bind a discovered value to
#     a ground-truth arg; the read RESULTS are ground truth (real API output), and
#     entity extraction / provenance reconciliation are deterministic post-steps.
#   * Enforce the read-only whitelist in code — never trust the model to stay read-only.
#   * Freeze (seed + cache) the produced StateModel: exploration has variance, and a
#     suite is only comparable across defenses if every defense saw the same probes.
# ---------------------------------------------------------------------------
