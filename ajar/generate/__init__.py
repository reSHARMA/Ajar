"""Importing the generate package registers every probe family.

Generators self-register via `@register_generator` on import, so loading them here
means any consumer (`from ajar.generate import build_suite`) gets the full suite
without having to remember each family module.
"""
from ajar.generate import (  # noqa: F401
    attacker,
    llm_parameter,
    parameter,
    stubs,
    tool_identity,
    undergrant,
)
from ajar.generate.base import build_suite, registered_generators  # noqa: F401
