"""The LLM seam. YOU provide the client and the keys; lp-bench only calls it.

lp-bench never constructs an LLM client or reads an API key. It depends solely on
the `LLMClient` Protocol below. Provide an object with these methods — or a short
shim over your SDK (Anthropic Messages, OpenAI, a gateway, whatever) — and inject
it at the call sites (`Runner(..., llm=your_client)`).

CRUCIAL DESIGN RULE: the LLM only *proposes*. Every proposal is disposed by the
oracle before it becomes a label. So `llm=None` is fully supported — generators
fall back to mechanical enumeration and the suite is still correct, just narrower.
The LLM widens coverage (semantic partitions, adversarial mutations, harm hints);
it never decides a label.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        """Free-form completion. `messages` is [{'role','content'}, ...]."""
        ...

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Structured completion validated against a JSON schema. Implementations
        may use tool-calling / response_format / a retry loop — lp-bench does not
        care how, only that the return conforms to `schema`."""
        ...


class NullLLM:
    """Reference no-op client so the pipeline runs with zero API dependency.

    Every generator treats an empty proposal list as 'mechanical-only'. Swap this
    for your real client to enable LLM-proposed probes.
    """

    def complete(self, messages, **kwargs) -> str:  # noqa: D401, ANN001
        return ""

    def complete_json(self, messages, schema, **kwargs):  # noqa: ANN001
        # An empty container that satisfies "no proposals".
        return [] if (schema or {}).get("type") == "array" else {}
