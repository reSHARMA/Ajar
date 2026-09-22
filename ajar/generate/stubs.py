"""Registered-but-unimplemented families, so the adequacy report shows an
explicit column (covered / applicable) rather than silently omitting them.

§3.4 sequence (premature / replay / reorder / abandoned-path), §3.5 provenance,
§3.7 cross-axis. Each needs adapter support that v0 does not wire (per-call
side-effect classification, a provenance carrier, a covering-array builder).
Filling one in is: implement the generator, register it, done — no core changes.
"""
from __future__ import annotations

from collections.abc import Iterable

from ajar.core.types import Probe
from ajar.generate.base import GenContext, register_generator

_STUB_FAMILIES = ("3.4-sequence", "3.5-provenance", "3.7-cross-axis")


def _make_stub(family: str):
    @register_generator(family)
    def _stub(gc: GenContext) -> Iterable[Probe]:  # noqa: ARG001
        return ()

    _stub.__name__ = f"stub_{family.replace('.', '_').replace('-', '_')}"
    return _stub


for _f in _STUB_FAMILIES:
    _make_stub(_f)
