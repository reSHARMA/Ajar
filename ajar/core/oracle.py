"""The oracle: it disposes labels. Generators only propose."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ajar.core.state import State
from ajar.core.types import Candidate, Label, LabelSource


@runtime_checkable
class Oracle(Protocol):
    """Assigns the authoritative label to a candidate at a state.

    Contract (from the measured AgentDojo findings):
      * DENY is returned ORACLE_CONFIRMED only when a benchmark predicate
        actually witnesses the harm (e.g. executing the call flips an injection
        task's `security()` to True).
      * ALLOW is returned ORACLE_CONFIRMED only when replaying a required call
        provably preserves the success predicate.
      * Otherwise Label.UNKNOWN — the benchmark cannot witness this (s, c). The
        generator's proposed label may stand, but only as ASSERTED, and UNKNOWN
        probes are excluded from OPL and reported as a coverage hole.

    NEVER let a monotone `utility()` mint a DENY: on AgentDojo, appending a full
    account drain leaves utility True. Over-grant DENYs are ASSERTED + harm-tiered.
    """

    def label(self, state: State, candidate: Candidate) -> tuple[Label, LabelSource, str]:
        """Return (label, source, evidence)."""
        ...
