"""The StateModel — the portable replacement for a materialized environment.

It is NOT the whole world. It is the minimal, task-relevant slice the Explorer
discovered by calling read-only tools, plus two artifacts that fall out of
exploration for free and that a raw environment never gives you:

  * read_trace   — the reads the Explorer had to make == the discovery-read closure
                   (solves the read-closure problem WITHOUT a run corpus).
  * provenance   — which read result each authorized argument came from == the
                   dataflow DAG the benchmark never records (feeds §3.4 sequence
                   and §3.5 provenance probes).

Generators consume this exactly where they previously consumed a live env: entities
become wrong-resource value pools, provenance seeds sequence/provenance probes, and
`unexplained_args` is an honest coverage signal (exploration didn't reach far enough).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ajar.core.types import Candidate


@dataclass(frozen=True)
class Entity:
    """A typed thing the Explorer discovered (an account, a file, a person, ...)."""

    kind: str  # "account" | "file" | "person" | "channel" | ...
    identity: str  # the id / handle / IBAN / path
    attributes: dict[str, Any] = field(default_factory=dict)
    source_read: str = ""  # which read call surfaced it


@dataclass(frozen=True)
class ProvenanceEdge:
    """An authorized argument, and the read whose result contained its value.
    `source_read == None` means the Explorer could NOT source this value from any
    read — either it is a literal constant, or exploration under-read."""

    gt_index: int  # index into ground_truth.authorized
    arg: str
    value: Any
    source_read: str | None


@dataclass
class StateModel:
    task_id: str
    read_trace: Sequence[Candidate] = ()  # == the discovery-read closure
    entities: Sequence[Entity] = ()
    provenance: Sequence[ProvenanceEdge] = ()
    # (tool, arg) -> discovered alternative values, for wrong-resource / partition gen
    value_pools: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    # authorized (gt_index, arg) pairs exploration could not source: a coverage hole
    unexplained_args: Sequence[tuple[int, str]] = ()

    @property
    def read_closure_tools(self) -> frozenset[str]:
        return frozenset(c.tool for c in self.read_trace)

    def pool(self, tool: str, arg: str) -> list[Any]:
        return self.value_pools.get((tool, arg), [])
