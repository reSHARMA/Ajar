"""Serialize probes and task specs to plain dicts, and read them back.

Why this exists: generating the full 97-task probe suite takes ~9 minutes, because
every probe's label comes from actually executing the candidate against a
materialized AgentDojo environment. Nothing about that work is defense-specific, so
paying it again for every defense evaluated is waste. Export once, score many times.

Both directions live in ONE file on purpose. Split across an exporter and a loader,
adding a field means editing two places, and the failure mode of forgetting one is
silent: a missing key falls back to its dataclass default (`required=True`,
`pruned=False`, `Label.ALLOW`), which does not raise -- it just changes the score.
`_PROBE_FIELDS` below is the single list both directions walk, so a new field is one
edit and a forgotten one is a KeyError at export.

Two things deliberately NOT serialized:

  * `Probe.scored` and `Probe.penalty_weight` are derived (`types.py:158-169`) from
    `pruned` / `expected_label` / `fault_class` / `harm_tier` / `required`. Writing
    them down creates a second copy that can contradict its own inputs, and nothing
    would say which copy wins. Recompute instead.
  * `TaskSpec.necessary` and `.notes` are informational (`ir.py:36`, `:42`) -- no
    defense and no scorer reads them. They ARE exported for provenance, but a
    reader is free to drop them.

`TaskSpec.tools` carries every tool's full JSON schema, which is most of the on-disk
size, and it is not optional: `tool_filter.prepare` (`tool_filter.py:65`) needs the
suite-wide tool NAMES to reproduce its substring-containment policy, and dropping
them would silently shrink what that defense is measured against.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Iterator

from ajar.core.ir import AttackSink, TaskSpec
from ajar.core.types import (
    Axis,
    Candidate,
    ContextRef,
    FaultClass,
    HarmTier,
    Label,
    LabelSource,
    Probe,
    Provenance,
    RenderedInjection,
    TaskKey,
    ToolSpec,
)

SCHEMA_VERSION = 1

# --- leaf types -----------------------------------------------------------


def _task_key_to_dict(k: TaskKey) -> dict:
    return {
        "benchmark": k.benchmark,
        "suite": k.suite,
        "version": list(k.version),
        "task_id": k.task_id,
    }


def _task_key_from_dict(d: Mapping[str, Any]) -> TaskKey:
    return TaskKey(
        benchmark=d["benchmark"],
        suite=d["suite"],
        version=tuple(d["version"]),
        task_id=d["task_id"],
    )


def _candidate_to_dict(c: Candidate) -> dict:
    return {"tool": c.tool, "args": dict(c.args)}


def _candidate_from_dict(d: Mapping[str, Any]) -> Candidate:
    return Candidate(tool=d["tool"], args=dict(d["args"]))


def _context_to_dict(c: ContextRef) -> dict:
    return {
        "task": _task_key_to_dict(c.task),
        "injections": dict(c.injections),
        "prefix": list(c.prefix),
        "clock": c.clock,
        "label": c.label,
    }


def _context_from_dict(d: Mapping[str, Any]) -> ContextRef:
    # A ContextRef is a replayable recipe, so `prefix` MUST come back as a tuple:
    # `build_suite` dedupes on `(context.prefix, candidate.key())` and a list is
    # unhashable, so a list here turns dedup into a TypeError.
    return ContextRef(
        task=_task_key_from_dict(d["task"]),
        injections=dict(d["injections"]),
        prefix=tuple(d["prefix"]),
        clock=d["clock"],
        label=d["label"],
    )


# --- Probe ----------------------------------------------------------------

# (field name, to_json, from_json). Both directions walk this one list.
_PROBE_FIELDS: tuple[tuple[str, Any, Any], ...] = (
    ("probe_id", lambda v: v, lambda v: v),
    ("task", _task_key_to_dict, _task_key_from_dict),
    ("context", _context_to_dict, _context_from_dict),
    ("candidate", _candidate_to_dict, _candidate_from_dict),
    ("expected_label", lambda v: v.value, Label),
    ("label_source", lambda v: v.value, LabelSource),
    ("axis", lambda v: v.value, Axis),
    ("fault_class", lambda v: v.value, FaultClass),
    ("harm_tier", lambda v: v.name, lambda v: HarmTier[v]),
    ("rationale", lambda v: v, lambda v: v),
    ("oracle_evidence", lambda v: v, lambda v: v),
    (
        "provenance",
        lambda v: None if v is None else {"arg": v.arg, "required_source": v.required_source},
        lambda v: None if v is None else Provenance(arg=v["arg"], required_source=v["required_source"]),
    ),
    ("pruned", lambda v: v, lambda v: v),
    ("generated_by", lambda v: v, lambda v: v),
    ("required", lambda v: v, lambda v: v),
    # Deliverable form of `candidate`: what a live agent is actually told to do.
    # Exported rather than re-derived because it is generated (optionally by an LLM),
    # so re-rendering on load would not reproduce the string a run was scored against.
    ("prompt", lambda v: v, lambda v: v),
    ("ground_truth", dict, dict),
)

# Guard: if someone adds a field to Probe and not to _PROBE_FIELDS, the round trip
# would silently substitute that field's default. Fail at import instead. `scored`
# and `penalty_weight` never appear here -- they are properties, not dataclass
# fields, so they are recomputed on load rather than stored.
_declared = {name for name, _, _ in _PROBE_FIELDS}
_actual = set(Probe.__dataclass_fields__)
if _declared != _actual:
    raise RuntimeError(
        f"ajar.io is out of sync with Probe: "
        f"missing {sorted(_actual - _declared)}, stale {sorted(_declared - _actual)}. "
        f"Add the field to _PROBE_FIELDS (both directions) or it will silently "
        f"round-trip to its default."
    )


def probe_to_dict(p: Probe) -> dict:
    return {name: to_json(getattr(p, name)) for name, to_json, _ in _PROBE_FIELDS}


def probe_from_dict(d: Mapping[str, Any]) -> Probe:
    # Indexing, not .get(): a missing key is a corrupt export, and defaulting it
    # would produce a probe that scores differently without ever complaining.
    return Probe(**{name: from_json(d[name]) for name, _, from_json in _PROBE_FIELDS})


# --- RenderedInjection ----------------------------------------------------
# Suite-scoped, so `suite` is carried on every row rather than implied by a filename:
# one injections.jsonl holds all four suites and a reader filters.


def injection_to_dict(r: RenderedInjection) -> dict:
    return {
        "suite": r.suite,
        "injection_id": r.injection_id,
        "goal": r.goal,
        "prompt": r.prompt,
        "calls": [_candidate_to_dict(c) for c in r.calls],
    }


def injection_from_dict(d: Mapping[str, Any]) -> RenderedInjection:
    return RenderedInjection(
        suite=d["suite"],
        injection_id=d["injection_id"],
        goal=d["goal"],
        prompt=d["prompt"],
        calls=[_candidate_from_dict(c) for c in d["calls"]],
    )


_inj_keys = {"suite", "injection_id", "goal", "prompt", "calls"}
if _inj_keys != set(RenderedInjection.__dataclass_fields__):
    raise RuntimeError(
        f"ajar.io is out of sync with RenderedInjection: "
        f"missing {sorted(set(RenderedInjection.__dataclass_fields__) - _inj_keys)}, "
        f"stale {sorted(_inj_keys - set(RenderedInjection.__dataclass_fields__))}."
    )


# --- TaskSpec -------------------------------------------------------------


def _tool_spec_to_dict(t: ToolSpec) -> dict:
    return {
        "name": t.name,
        "params_schema": dict(t.params_schema),
        "resource_domain": t.resource_domain,
        "side_effect": t.side_effect,
    }


def _tool_spec_from_dict(d: Mapping[str, Any]) -> ToolSpec:
    return ToolSpec(
        name=d["name"],
        params_schema=dict(d["params_schema"]),
        resource_domain=d["resource_domain"],
        side_effect=d["side_effect"],
    )


def _sink_to_dict(s: AttackSink) -> dict:
    return {
        "attack_id": s.attack_id,
        "candidate": _candidate_to_dict(s.candidate),
        "harm_tier": s.harm_tier.name,
        "jointly_with": list(s.jointly_with),
        "evidence": s.evidence,
    }


def _sink_from_dict(d: Mapping[str, Any]) -> AttackSink:
    return AttackSink(
        attack_id=d["attack_id"],
        candidate=_candidate_from_dict(d["candidate"]),
        harm_tier=HarmTier[d["harm_tier"]],
        jointly_with=tuple(d["jointly_with"]),
        evidence=d["evidence"],
    )


def spec_to_dict(s: TaskSpec) -> dict:
    return {
        "task": _task_key_to_dict(s.task),
        "prompt": s.prompt,
        "tools": [_tool_spec_to_dict(t) for t in s.tools],
        "plan": [_candidate_to_dict(c) for c in s.plan],
        "necessary": sorted(s.necessary),
        "read_closure": sorted(s.read_closure),
        "permitted_write_paths": (
            None if s.permitted_write_paths is None else sorted(s.permitted_write_paths)
        ),
        "attack_sinks": [_sink_to_dict(a) for a in s.attack_sinks],
        "is_gradeable": s.is_gradeable,
        "notes": s.notes,
    }


def spec_from_dict(d: Mapping[str, Any]) -> TaskSpec:
    return TaskSpec(
        task=_task_key_from_dict(d["task"]),
        prompt=d["prompt"],
        tools=[_tool_spec_from_dict(t) for t in d["tools"]],
        plan=[_candidate_from_dict(c) for c in d["plan"]],
        necessary=frozenset(d["necessary"]),
        read_closure=frozenset(d["read_closure"]),
        permitted_write_paths=(
            None if d["permitted_write_paths"] is None else frozenset(d["permitted_write_paths"])
        ),
        attack_sinks=[_sink_from_dict(a) for a in d["attack_sinks"]],
        is_gradeable=d["is_gradeable"],
        notes=d["notes"],
    )


# Same drift guard as for Probe: TaskSpec gains a field, this raises at import.
_spec_keys = {
    "task", "prompt", "tools", "plan", "necessary", "read_closure",
    "permitted_write_paths", "attack_sinks", "is_gradeable", "notes",
}
if _spec_keys != set(TaskSpec.__dataclass_fields__):
    raise RuntimeError(
        f"ajar.io is out of sync with TaskSpec: "
        f"missing {sorted(set(TaskSpec.__dataclass_fields__) - _spec_keys)}, "
        f"stale {sorted(_spec_keys - set(TaskSpec.__dataclass_fields__))}."
    )


# --- JSONL ----------------------------------------------------------------
# sort_keys so two generation runs produce byte-identical files (a diff should mean
# the probes changed, not that a dict iterated differently). ensure_ascii=False
# keeps the non-ASCII arg values in the AgentDojo fixtures readable.


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(_dumps(row) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(obj) + "\n", encoding="utf-8")


def load_specs(path: Path) -> dict[TaskKey, TaskSpec]:
    return {(s := spec_from_dict(d)).task: s for d in read_jsonl(path)}


def load_probes(path: Path) -> dict[TaskKey, list[Probe]]:
    """Grouped by task, insertion-ordered, so a reader can score task by task."""
    out: dict[TaskKey, list[Probe]] = {}
    for d in read_jsonl(path):
        p = probe_from_dict(d)
        out.setdefault(p.task, []).append(p)
    return out


def load_task_probes(path: Path, key: TaskKey) -> list[Probe]:
    """One task's probes, without materializing the other 96 tasks'.

    The live-run harness runs one task at a time, and `load_probes` would build
    ~11.9k Probe objects to hand back the ~120 it wants. Filtering on the raw dict
    keeps the cost to a JSON parse per line.

    Empty list, not an error, for an absent task: the caller distinguishes "not in
    this export" from "no probes" (`run_eval_all` skips tasks with none), and a task
    excluded as non-gradeable is a normal state, not a corrupt file.
    """
    want = _task_key_to_dict(key)
    return [probe_from_dict(d) for d in read_jsonl(path) if d["task"] == want]


def load_task_keys(path: Path) -> set[TaskKey]:
    """Which tasks a probe export covers -- for callers that only need to know
    whether a task is present before deciding to run it."""
    return {_task_key_from_dict(d["task"]) for d in read_jsonl(path)}


def load_injections(path: Path, suite: str | None = None) -> list[RenderedInjection]:
    """All rendered injections, or one suite's. Empty list when the file is absent:
    injections are optional (only the pairing runner needs them), so a missing export
    is a normal state for the scoring path, not a corrupt one."""
    if not path.is_file():
        return []
    return [
        injection_from_dict(d)
        for d in read_jsonl(path)
        if suite is None or d["suite"] == suite
    ]
